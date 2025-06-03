import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import unquote
import asyncio
import time # Added for rate limiter
from collections import deque # Added for rate limiter

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from api.config import get_model_config
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.rag import RAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Rate Limiter Configuration ---
LLM_REQUEST_TIMESTAMPS = deque()
LLM_RPM_LIMIT = 100  # Requests Per Minute
LLM_RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_LOCK = asyncio.Lock()

async def enforce_llm_rate_limit():
    async with RATE_LIMIT_LOCK:
        now = time.monotonic()
        
        # Remove timestamps older than the window
        while LLM_REQUEST_TIMESTAMPS and LLM_REQUEST_TIMESTAMPS[0] <= now - LLM_RATE_LIMIT_WINDOW_SECONDS:
            LLM_REQUEST_TIMESTAMPS.popleft()
            
        if len(LLM_REQUEST_TIMESTAMPS) >= LLM_RPM_LIMIT:
            oldest_relevant_timestamp = LLM_REQUEST_TIMESTAMPS[0]
            wait_time = (oldest_relevant_timestamp + LLM_RATE_LIMIT_WINDOW_SECONDS) - now
            if wait_time > 0:
                logger.info(f"Rate limit reached ({LLM_RPM_LIMIT} RPM). Waiting for {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)
                # Re-clean after waiting
                now = time.monotonic() # Update current time
                while LLM_REQUEST_TIMESTAMPS and LLM_REQUEST_TIMESTAMPS[0] <= now - LLM_RATE_LIMIT_WINDOW_SECONDS:
                    LLM_REQUEST_TIMESTAMPS.popleft()

        LLM_REQUEST_TIMESTAMPS.append(now)
# --- End Rate Limiter ---

# Define a safety margin for token limits
# (e.g., Gemini 2.5 Pro has a 32k context window, but prompts can be large)
# Let's aim for a prompt significantly smaller than the absolute max.
# The previous check was at 8000, and warnings were for > 7500.
# Let's set a more conservative overall prompt limit.
# Considering the detailed system prompts, a 16k limit for the *entire assembled prompt*
# before sending to the LLM might be a safer starting point.
# The actual model context window is much larger (e.g., 32k for Gemini 2.5 Pro, or even 1M+ for others)
# but the issue arises from the prompt itself being too large for *this specific application's typical request structure*.
# The 7500 limit was for the user message content alone.
# Let's refine this. The core issue is the final assembled prompt for the LLM.
# The Google model previously errored when the *request size* (which seems to be just the last message) was 26858.
# This suggests that the other parts of the prompt (system message, RAG context) add to this.
# Let's set a target for the RAG context itself.
MAX_RAG_CONTEXT_TOKENS = 10000  # Max tokens for the RAG-retrieved context_text
MAX_OVERALL_PROMPT_TOKENS = 24000 # Revised based on observed issues with ~26k user message content.

# Token limits by provider
PROVIDER_TOKEN_LIMITS = {
    "google": {
        "gemini-2.0-flash": 1048576,  # 1M context window
        "gemini-2.0-flash-experimental": 1048576,
        "gemini-2.0-pro": 2097152,  # 2M context window
        "gemini-2.5-pro-preview-0409": 2097152,
        "gemini-2.5-pro-preview-05-06": 2097152,
        "gemini-2.5-flash": 1048576,
        "gemini-2.5-flash-preview-01-11": 1048576,
        "gemini-2.5-flash-preview-05-20": 1048576,
        "gemini-1.5-pro": 2097152,
        "gemini-1.5-flash": 1048576,
        "default": 32768  # Conservative default
    },
    "openai": {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16384,
        "default": 8192
    },
    "openrouter": {
        "default": 128000  # Most OpenRouter models support large contexts
    },
    "ollama": {
        "default": 8192  # Conservative default for local models
    }
}

def get_model_token_limit(provider: str, model: str) -> int:
    """Get the token limit for a specific provider and model."""
    provider_limits = PROVIDER_TOKEN_LIMITS.get(provider, {})
    return provider_limits.get(model, provider_limits.get("default", 32768))

def calculate_safe_prompt_limit(provider: str, model: str) -> int:
    """Calculate a safe prompt limit based on the model's context window."""
    model_limit = get_model_token_limit(provider, model)
    # Use 80% of the model's limit as a safety margin
    return int(model_limit * 0.8)

def truncate_text_by_tokens(text: str, max_tokens: int, is_ollama: bool = False) -> tuple[str, bool]:
    """Truncates text to a maximum number of tokens. Returns (truncated_text, was_truncated)."""
    original_tokens = count_tokens(text, is_ollama)
    if original_tokens <= max_tokens:
        return text, False

    # Simple truncation for now, could be made smarter (e.g., sentence boundaries)
    # For simplicity, let's try a character-based approximation for truncation.
    avg_chars_per_token = 4  # This is a rough estimate
    estimated_chars_to_keep = max_tokens * avg_chars_per_token
    
    truncated_text = text[:estimated_chars_to_keep]
    
    # Recalculate tokens and adjust if still over (e.g., due to UTF-8 or tokenization nuances)
    loop_count = 0 # Safety break for loop
    while count_tokens(truncated_text, is_ollama) > max_tokens and len(truncated_text) > 0 and loop_count < 100:
        truncated_text = truncated_text[:-100] # Reduce by a chunk
        if not truncated_text: # Safety break
            break
        loop_count += 1
            
    # Final check, if still over after rough truncation, do a more precise (but slower) word by word.
    if count_tokens(truncated_text, is_ollama) > max_tokens:
        words = text.split() # split by space, not ideal for all languages but a start
        truncated_words = []
        current_word_tokens = 0
        for word in words:
            word_with_space = word + " "
            # Check token count of current word + space, or word itself if it's the last one or too long
            word_tokens = count_tokens(word_with_space, is_ollama)
            if current_word_tokens + word_tokens <= max_tokens:
                truncated_words.append(word)
                current_word_tokens += word_tokens
            else:
                # If even a single word is too long, try to truncate it (very basic)
                if not truncated_words and count_tokens(word, is_ollama) > max_tokens:
                    truncated_word, _ = truncate_text_by_tokens(word, max_tokens, is_ollama) # Recursive call for a single word
                    truncated_words.append(truncated_word)
                    current_word_tokens += count_tokens(truncated_word, is_ollama)
                break
        truncated_text = " ".join(truncated_words)

    final_tokens = count_tokens(truncated_text, is_ollama)
    was_actually_truncated = final_tokens < original_tokens
    
    if was_actually_truncated:
        logger.warning(f"Truncated text from {original_tokens} to {final_tokens} tokens to fit token limit of {max_tokens}.")
        return truncated_text + "... (truncated)", True
    else:
        # This case implies original_tokens <= max_tokens, or truncation didn't reduce it (e.g. single very long token)
        return truncated_text, False

# Get API keys from environment variables
google_api_key = os.environ.get('GOOGLE_API_KEY')

# Configure Google Generative AI
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

# Models for the API
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Model for requesting a chat completion.
    """
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab', 'bitbucket')")

    # model parameters
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")

async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions.
    This replaces the HTTP streaming endpoint with a WebSocket connection.
    """
    await websocket.accept()

    try:
        # Receive and parse the request data
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        # Check if request contains very large input (basic check on last message)
        input_too_large = False
        last_message_content_tokens = 0
        is_ollama_provider = request.provider == "ollama"
        model_name = request.model or get_model_config(request.provider, request.model)["model_kwargs"]["model"]
        safe_prompt_limit = calculate_safe_prompt_limit(request.provider, model_name)
        
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                last_message_content_tokens = count_tokens(last_message.content, is_ollama_provider)
                logger.info(f"Last message content size: {last_message_content_tokens} tokens")
                logger.info(f"Safe prompt limit for {request.provider}/{model_name}: {safe_prompt_limit} tokens")
                
                # Adjust threshold based on model capabilities
                rag_skip_threshold = min(safe_prompt_limit * 0.5, 20000)  # Use 50% of safe limit or 20k, whichever is smaller
                
                if last_message_content_tokens > rag_skip_threshold:
                    logger.warning(f"Last message content ({last_message_content_tokens} tokens) exceeds RAG threshold ({rag_skip_threshold} tokens), RAG will be skipped.")
                    input_too_large = True

        # Create a new RAG instance for this request
        try:
            request_rag = RAG(provider=request.provider, model=request.model)

            excluded_dirs = [unquote(dir_path) for dir_path in (request.excluded_dirs or "").split('\\n') if dir_path.strip()]
            excluded_files = [unquote(file_pattern) for file_pattern in (request.excluded_files or "").split('\\n') if file_pattern.strip()]
            included_dirs = [unquote(dir_path) for dir_path in (request.included_dirs or "").split('\\n') if dir_path.strip()]
            included_files = [unquote(file_pattern) for file_pattern in (request.included_files or "").split('\\n') if file_pattern.strip()]
            
            if excluded_dirs: logger.info(f"Using custom excluded directories: {excluded_dirs}")
            if excluded_files: logger.info(f"Using custom excluded files: {excluded_files}")
            if included_dirs: logger.info(f"Using custom included directories: {included_dirs}")
            if included_files: logger.info(f"Using custom included files: {included_files}")

            request_rag.prepare_retriever(request.repo_url, request.type, request.token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"Retriever prepared for {request.repo_url}")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"No valid embeddings found: {str(e)}")
                await websocket.send_text("Error: No valid document embeddings found. This may be due to embedding size inconsistencies or API errors during document processing. Please try again or check your repository content.")
                await websocket.close()
                return
            else:
                logger.error(f"ValueError preparing retriever: {str(e)}")
                await websocket.send_text(f"Error preparing retriever: {str(e)}")
                await websocket.close()
                return
        except Exception as e:
            logger.error(f"Error preparing retriever: {str(e)}")
            if "All embeddings should be of the same size" in str(e):
                await websocket.send_text("Error: Inconsistent embedding sizes detected. Some documents may have failed to embed properly. Please try again.")
            else:
                await websocket.send_text(f"Error preparing retriever: {str(e)}")
            await websocket.close()
            return

        # Validate request
        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("Error: No messages provided")
            await websocket.close()
            return

        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            await websocket.close()
            return

        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=user_msg.content,
                        assistant_response=assistant_msg.content
                    )

        is_deep_research = False
        research_iteration = 1
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                if msg == request.messages[-1]:
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                original_topic = None
                for msg_hist in request.messages:
                    if msg_hist.role == "user" and "continue" not in msg_hist.content.lower():
                        original_topic = msg_hist.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break
                if original_topic:
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        query = last_message.content
        context_text = ""
        CONTEXT_START = "<START_OF_CONTEXT>"  # Define the context delimiter
        if not input_too_large:
            try:
                rag_query = query
                if request.filePath:
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(f"Modified RAG query to focus on file: {request.filePath}")
                
                retrieved_documents = request_rag(rag_query, language=request.language)
                if retrieved_documents and retrieved_documents[0].documents:
                    documents = retrieved_documents[0].documents
                    logger.info(f"Retrieved {len(documents)} documents via RAG.")
                    docs_by_file = {}
                    for doc in documents:
                        file_path_meta = doc.meta_data.get('file_path', 'unknown')
                        if file_path_meta not in docs_by_file:
                            docs_by_file[file_path_meta] = []
                        docs_by_file[file_path_meta].append(doc)
                    context_parts = [f"## File Path: {fp}\\n\\n" + "\\n\\n".join([d.text for d in docs_in_file]) for fp, docs_in_file in docs_by_file.items()]
                    context_text = "\\n\\n" + "----------\\n\\n".join(context_parts)
                    
                    # Calculate available token budget for RAG context
                    # We need to reserve tokens for: system prompt, conversation history, file content, and user query
                    estimated_system_prompt_tokens = 1500  # Conservative estimate
                    conversation_history = ""
                    for turn_id, turn in request_rag.memory().items():
                        if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                            conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"
                    conversation_history_tokens = count_tokens(conversation_history, is_ollama_provider) if conversation_history else 0
                    file_content_tokens = 0
                    if request.filePath:
                        try:
                            temp_file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                            file_content_tokens = count_tokens(temp_file_content, is_ollama_provider)
                        except:
                            file_content_tokens = 0
                    
                    # Calculate remaining budget for RAG context
                    used_tokens = (estimated_system_prompt_tokens + 
                                  conversation_history_tokens + 
                                  file_content_tokens + 
                                  last_message_content_tokens + 
                                  500)  # Buffer for formatting
                    
                    available_rag_tokens = safe_prompt_limit - used_tokens
                    max_rag_tokens = min(available_rag_tokens, MAX_RAG_CONTEXT_TOKENS)
                    
                    logger.info(f"Token budget calculation: safe_limit={safe_prompt_limit}, used={used_tokens}, available_for_rag={available_rag_tokens}, max_rag={max_rag_tokens}")
                    
                    # Truncate RAG context if necessary
                    rag_context_tokens = count_tokens(context_text, is_ollama_provider)
                    if rag_context_tokens > max_rag_tokens and max_rag_tokens > 0:
                        logger.warning(f"RAG context ({rag_context_tokens} tokens) exceeds budget ({max_rag_tokens} tokens). Truncating.")
                        context_text, was_truncated = truncate_text_by_tokens(context_text, max_rag_tokens, is_ollama_provider)
                        if was_truncated:
                            context_text += "\\n\\n(Note: Context was truncated to fit within token limits)"
                    elif max_rag_tokens <= 0:
                        logger.warning("No token budget available for RAG context. Skipping RAG.")
                        context_text = ""
                else:
                    logger.warning("No documents retrieved from RAG")
            except Exception as e:
                logger.error(f"Error in RAG retrieval: {str(e)}")
        
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
        repo_type = request.type
        language_code = request.language or "en"
        language_name = {"en": "English", "ja": "Japanese (日本語)", "zh": "Mandarin Chinese (中文)", "es": "Spanish (Español)", "kr": "Korean (한국어)", "vi": "Vietnamese (Tiếng Việt)"}.get(language_code, "English")

        # Create system prompt
        if is_deep_research:
            is_first_iteration = research_iteration == 1
            is_final_iteration = research_iteration >= 5
            if is_first_iteration:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are conducting a multi-turn Deep Research process to thoroughly investigate the specific topic in the user's query.
Your goal is to provide detailed, focused information EXCLUSIVELY about this topic.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the first iteration of a multi-turn research process focused EXCLUSIVELY on the user's query
- Start your response with "## Research Plan"
- Outline your approach to investigating this specific topic
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- Clearly state the specific topic you're researching to maintain focus throughout all iterations
- Identify the key aspects you'll need to research
- Provide initial findings based on the information available
- End with "## Next Steps" indicating what you'll investigate in the next iteration
- Do NOT provide a final conclusion yet - this is just the beginning of the research
- Do NOT include general repository information unless directly relevant to the query
- Focus EXCLUSIVELY on the specific topic being researched - do not drift to related topics
- Your research MUST directly address the original question
- NEVER respond with just "Continue the research" as an answer - always provide substantive research findings
- Remember that this topic will be maintained across all research iterations
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
</style>"""
            elif is_final_iteration:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are in the final iteration of a Deep Research process focused EXCLUSIVELY on the latest user query.
Your goal is to synthesize all previous findings and provide a comprehensive conclusion that directly addresses this specific topic and ONLY this topic.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the final iteration of the research process
- CAREFULLY review the entire conversation history to understand all previous findings
- Synthesize ALL findings from previous iterations into a comprehensive conclusion
- Start with "## Final Conclusion"
- Your conclusion MUST directly address the original question
- Stay STRICTLY focused on the specific topic - do not drift to related topics
- Include specific code references and implementation details related to the topic
- Highlight the most important discoveries and insights about this specific functionality
- Provide a complete and definitive answer to the original question
- Do NOT include general repository information unless directly relevant to the query
- Focus exclusively on the specific topic being researched
- NEVER respond with "Continue the research" as an answer - always provide a complete conclusion
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- Ensure your conclusion builds on and references key findings from previous iterations
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
- Structure your response with clear headings
- End with actionable insights or recommendations when appropriate
</style>"""
            else:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are currently in iteration {research_iteration} of a Deep Research process focused EXCLUSIVELY on the latest user query.
Your goal is to build upon previous research iterations and go deeper into this specific topic without deviating from it.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- CAREFULLY review the conversation history to understand what has been researched so far
- Your response MUST build on previous research iterations - do not repeat information already covered
- Identify gaps or areas that need further exploration related to this specific topic
- Focus on one specific aspect that needs deeper investigation in this iteration
- Start your response with "## Research Update {research_iteration}"
- Clearly explain what you're investigating in this iteration
- Provide new insights that weren't covered in previous iterations
- If this is iteration 3, prepare for a final conclusion in the next iteration
- Do NOT include general repository information unless directly relevant to the query
- Focus EXCLUSIVELY on the specific topic being researched - do not drift to related topics
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- NEVER respond with just "Continue the research" as an answer - always provide substantive research findings
- Your research MUST directly address the original question
- Maintain continuity with previous research iterations - this is a continuous investigation
</guidelines>

<style>
- Be concise but thorough
- Focus on providing new information, not repeating what's already been covered
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
</style>"""
        else:
            system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You provide direct, concise, and accurate information about code repositories.
You NEVER start responses with markdown headers or code fences.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- Answer the user's question directly without ANY preamble or filler phrases
- DO NOT include any rationale, explanation, or extra comments.
- DO NOT start with preambles like "Okay, here's a breakdown" or "Here's an explanation"
- DO NOT start with markdown headers like "## Analysis of..." or any file path references
- DO NOT start with ```markdown code fences
- DO NOT end your response with ``` closing fences
- DO NOT start by repeating or acknowledging the question
- JUST START with the direct answer to the question

<example_of_what_not_to_do>
```markdown
## Analysis of `adalflow/adalflow/datasets/gsm8k.py`

This file contains...
```
</example_of_what_not_to_do>
- Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer
- For code analysis, organize your response with clear sections
- Think step by step and structure your answer logically
- Start with the most relevant information that directly addresses the user's query
- Be precise and technical when discussing code
- Your response language should be in the same language as the user's query
</guidelines>

<style>
- Use concise, direct language
- Prioritize accuracy over verbosity
- When showing code, include line numbers and file paths when relevant
- Use markdown formatting to improve readability
</style>"""

        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")

        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"
        
        # Determine the base of the prompt: if the query looks like a large instructional prompt, use it as the base.
        # Otherwise, use the system_prompt.
        # This is a simpler heuristic than the full token-based switching previously.
        # A "master instructional query" is likely long and contains specific phrasing.
        prompt_base = ""
        if last_message_content_tokens > 3000 and ("expert technical writer" in query.lower() or "[WIKI_PAGE_TOPIC]" in query.lower()):
            logger.info("Using user query as the primary prompt base due to its size and content.")
            prompt_base = query # The user's detailed instructions
        else:
            prompt_base = system_prompt # Backend-defined system prompt

        # Assemble the prompt
        prompt_parts = [prompt_base]

        if conversation_history:
            prompt_parts.append(f"<conversation_history>\\n{conversation_history}</conversation_history>")

        if file_content:
            prompt_parts.append(f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{file_content}\\n</currentFileContent>")
        
        if context_text.strip():
            prompt_parts.append(f"<START_OF_CONTEXT>\\n{context_text}\\n<END_OF_CONTEXT>")
        elif not input_too_large: # RAG was attempted but yielded no results
             prompt_parts.append("<note>No relevant context found via retrieval. Answering based on general knowledge and provided query.</note>")
        elif input_too_large: # RAG was skipped
            prompt_parts.append("<note>Retrieval augmentation was skipped due to the large size of the input query.</note>")

        if not is_deep_research: # if not a Deep Research query, add the user's query explicitly if it wasn't the base
             prompt_parts.append(f"<query>\\n{query}\\n</query>")
        
        prompt = "\\n\\n".join(prompt_parts)
        prompt += "\\n\\nAssistant:"
        
        # Calculate and validate final prompt size
        final_prompt_token_count = count_tokens(prompt, is_ollama_provider)
        logger.info(f"Final assembled prompt token count: {final_prompt_token_count}")
        
        if final_prompt_token_count > safe_prompt_limit:
            logger.warning(f"Final prompt ({final_prompt_token_count} tokens) exceeds safe limit ({safe_prompt_limit} tokens)")
            
            # Attempt to reduce prompt size by removing less critical components
            # Priority order: 1) Keep user query, 2) Keep system prompt, 3) Reduce context, 4) Reduce conversation history
            
            # First, try removing/reducing context
            if context_text.strip():
                logger.info("Attempting to reduce prompt size by removing RAG context")
                prompt_parts_reduced = [p for p in prompt_parts if not (CONTEXT_START in p or context_text in p)]
                prompt_reduced = "\\n\\n".join(prompt_parts_reduced) + "\\n\\nAssistant:"
                reduced_token_count = count_tokens(prompt_reduced, is_ollama_provider)
                
                if reduced_token_count <= safe_prompt_limit:
                    prompt = prompt_reduced
                    final_prompt_token_count = reduced_token_count
                    logger.info(f"Reduced prompt to {final_prompt_token_count} tokens by removing context")
                else:
                    # If still too large, try truncating conversation history
                    if conversation_history:
                        logger.info("Still too large, attempting to truncate conversation history")
                        # Keep only the most recent exchanges
                        conv_parts = conversation_history.split("<turn>")
                        if len(conv_parts) > 3:  # Keep only last 2 turns
                            recent_conv = "<turn>".join(conv_parts[-3:])
                            prompt_parts_minimal = [prompt_base]
                            if recent_conv:
                                prompt_parts_minimal.append(f"<conversation_history>\\n{recent_conv}</conversation_history>")
                            if file_content:
                                prompt_parts_minimal.append(f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{file_content}\\n</currentFileContent>")
                            if not is_deep_research and query != prompt_base:
                                prompt_parts_minimal.append(f"<query>\\n{query}\\n</query>")
                            
                            prompt_minimal = "\\n\\n".join(prompt_parts_minimal) + "\\n\\nAssistant:"
                            minimal_token_count = count_tokens(prompt_minimal, is_ollama_provider)
                            
                            if minimal_token_count <= safe_prompt_limit:
                                prompt = prompt_minimal
                                final_prompt_token_count = minimal_token_count
                                logger.info(f"Reduced prompt to {final_prompt_token_count} tokens by truncating conversation history")

        model_config = get_model_config(request.provider, request.model)["model_kwargs"]

        if request.provider == "ollama":
            # prompt += " /no_think" # Ollama specific - handled by client if needed
            model = OllamaClient()
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "options": {
                    "temperature": model_config["temperature"],
                    "top_p": model_config["top_p"],
                    "num_ctx": model_config["num_ctx"]
                }
            }
            api_kwargs = model.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        
        elif request.provider == "openrouter":
            logger.info(f"Using OpenRouter with model: {request.model}")
            if not os.environ.get("OPENROUTER_API_KEY"):
                logger.warning("OPENROUTER_API_KEY environment variable is not set, but continuing with request")
            model = OpenRouterClient()
            model_kwargs = {"model": request.model, "stream": True, "temperature": model_config["temperature"], "top_p": model_config["top_p"]}
            api_kwargs = model.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)

        elif request.provider == "openai":
            logger.info(f"Using Openai protocol with model: {request.model}")
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY environment variable is not set, but continuing with request")
            model = OpenAIClient()
            model_kwargs = {"model": request.model, "stream": True, "temperature": model_config["temperature"], "top_p": model_config["top_p"]}
            api_kwargs = model.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        
        else: # Default to Google
            model = genai.GenerativeModel(
                model_name=model_config["model"],
                generation_config={
                    "temperature": model_config["temperature"],
                    "top_p": model_config["top_p"],
                    "top_k": model_config["top_k"]
                }
            )
            # For Google, api_kwargs is not used in the same way; prompt is passed directly to generate_content

        # Process the response based on the provider
        try:
            await enforce_llm_rate_limit() # Enforce rate limit before LLM call
            logger.info(f"Making LLM call for {request.provider} after rate limit check...")

            if request.provider == "ollama":
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                async for chunk in response:
                    text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                    if text and not text.startswith('model=') and not text.startswith('created_at='):
                        text = text.replace('<think>', '').replace('</think>', '')
                        await websocket.send_text(text)
                await websocket.close()
            elif request.provider == "openrouter":
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                async for chunk in response:
                    await websocket.send_text(chunk)
                await websocket.close()
            elif request.provider == "openai":
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                async for chunk in response:
                    choices = getattr(chunk, "choices", [])
                    if len(choices) > 0:
                        delta = getattr(choices[0], "delta", None)
                        if delta is not None:
                            text = getattr(delta, "content", None)
                            if text is not None:
                                await websocket.send_text(text)
                await websocket.close()
            else: # Google
                response = model.generate_content(prompt, stream=True)
                for chunk in response:
                    # Check if the chunk has parts and iterate through them
                    if chunk.parts:
                        for part in chunk.parts:
                            if hasattr(part, 'text') and part.text:
                                await websocket.send_text(part.text)
                    # Handle cases where generation might have stopped without valid parts (e.g. safety reasons)
                    elif chunk.candidates and chunk.candidates[0].finish_reason.name != 'STOP' and chunk.candidates[0].finish_reason.name != 'UNSPECIFIED':
                        logger.warning(f"Chunk processing stopped. Finish Reason: {chunk.candidates[0].finish_reason.name}")
                        # Optionally send a message to client or break, depending on desired behavior
                        # For now, we just log and continue, but might be better to break or send an error marker.
                await websocket.close()

        except Exception as e_outer:
            logger.error(f"Error in streaming response or LLM call: {str(e_outer)}", exc_info=True)
            error_message = str(e_outer)
            
            # Check if it's a Google API specific error that might indicate content policy violation / safety
            is_google_api_error = False
            google_response_text = None
            if hasattr(e_outer, 'response') and hasattr(e_outer.response, 'prompt_feedback'):
                 # This structure is typical for google.generativeai.types. génération.GenerateContentResponse errors
                if e_outer.response.prompt_feedback.block_reason:
                    is_google_api_error = True
                    google_response_text = f"Content generation blocked. Reason: {e_outer.response.prompt_feedback.block_reason}."
                    if e_outer.response.candidates and e_outer.response.candidates[0].finish_reason.name == 'SAFETY':
                         google_response_text += f" Finish Reason: SAFETY. Safety Ratings: {e_outer.response.prompt_feedback.safety_ratings}"

            if google_response_text:
                 await websocket.send_text(f"\\nError: {google_response_text}")
            elif "maximum context length" in error_message.lower() or "token limit" in error_message.lower() or "too many tokens" in error_message.lower() or "prompt is too long" in error_message.lower() or "user input is too long" in error_message.lower() :
                logger.warning("Token limit exceeded, attempting fallback without RAG/file context.")
                try:
                    # Simplified prompt: System prompt + original query ONLY.
                    simplified_prompt_base = ""
                    if is_deep_research and last_message_content_tokens < 30000 : # If Deep Research query was just a bit too big with context
                        simplified_prompt_base = query # Try with just the Deep Research query
                        logger.info("Fallback: Using Deep Research query as base due to its large size.")
                    else: # For regular queries or extremely large Deep Research queries, fall back to system_prompt + query
                        simplified_prompt_base = f"{system_prompt}\\n\\n<query>\\n{query}\\n</query>"

                    simplified_prompt = f"{simplified_prompt_base}\\n\\n<note>Answering with reduced context due to original request size.</note>\\n\\nAssistant:"
                    
                    logger.info(f"Fallback simplified prompt token count: {count_tokens(simplified_prompt, request.provider == 'ollama')}")

                    await enforce_llm_rate_limit() # Enforce rate limit for fallback call too
                    logger.info(f"Making FALLBACK LLM call for {request.provider} after rate limit check...")

                    if request.provider == "ollama":
                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(input=simplified_prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)
                        async for chunk in fallback_response:
                            text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                            if text and not text.startswith('model=') and not text.startswith('created_at='):
                                text = text.replace('<think>', '').replace('</think>', '')
                                await websocket.send_text(text)
                    elif request.provider == "openrouter":
                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(input=simplified_prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)
                        async for chunk in fallback_response: await websocket.send_text(chunk)
                    elif request.provider == "openai":
                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(input=simplified_prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)
                        async for chunk in fallback_response:
                            choices = getattr(chunk, "choices", [])
                            if len(choices) > 0:
                                delta = getattr(choices[0], "delta", None)
                                if delta is not None:
                                    text = getattr(delta, "content", None)
                                    if text is not None: await websocket.send_text(text)
                    else: # Google
                        fallback_model = genai.GenerativeModel( model_name=model_config["model"], generation_config=model.generation_config ) # Re-use config
                        fallback_response = fallback_model.generate_content(simplified_prompt, stream=True)
                        for chunk in fallback_response:
                            if hasattr(chunk, 'text'): await websocket.send_text(chunk.text)
                except Exception as e2:
                    logger.error(f"Error in fallback streaming response: {str(e2)}", exc_info=True)
                    await websocket.send_text(f"\\nI apologize, but your request is too large for me to process, even after simplification. Please try a shorter query or break it into smaller parts.")
            else:
                await websocket.send_text(f"\\nError: {error_message}")
            
            await websocket.close() # Ensure close on error

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}", exc_info=True)
        try:
            await websocket.send_text(f"Error: {str(e)}")
            await websocket.close()
        except: # If sending error itself fails
            pass
