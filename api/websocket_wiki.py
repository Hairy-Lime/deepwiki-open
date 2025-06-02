import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import unquote
import asyncio

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

        # Check if request contains very large input
        input_too_large = False
        last_message_content_tokens = 0
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                # Use the provider from the request to determine if it's ollama for token counting
                is_ollama_provider = request.provider == "ollama"
                last_message_content_tokens = count_tokens(last_message.content, is_ollama_provider)
                logger.info(f"Last message content size: {last_message_content_tokens} tokens")
                # This 7500 was a general threshold. Let's keep it for the user message part.
                if last_message_content_tokens > 7500: # Adjusted from 8000 to align with warning
                    logger.warning(f"Last message content exceeds recommended token limit ({last_message_content_tokens} > 7500)")
                    input_too_large = True # This flag mainly affects RAG bypass

        # Create a new RAG instance for this request
        try:
            request_rag = RAG(provider=request.provider, model=request.model)

            # Extract custom file filter parameters if provided
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom excluded directories: {excluded_dirs}")
            if request.excluded_files:
                excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom excluded files: {excluded_files}")
            if request.included_dirs:
                included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom included directories: {included_dirs}")
            if request.included_files:
                included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom included files: {included_files}")

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
            # Check for specific embedding-related errors
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

        # Check if this is a Deep Research request
        is_deep_research = False
        research_iteration = 1

        # Process messages to detect Deep Research requests
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                # Only remove the tag from the last message
                if msg == request.messages[-1]:
                    # Remove the Deep Research tag
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        # Count research iterations if this is a Deep Research request
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")

            # Check if this is a continuation request
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                # Find the original topic from the first user message
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break

                if original_topic:
                    # Replace the continuation message with the original topic
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        # Get the query from the last message
        query = last_message.content

        # Only retrieve documents if input is not too large (based on last message content)
        context_text = ""
        retrieved_documents = None
        retrieved_documents_count = 0

        if not input_too_large:
            try:
                # If filePath exists, modify the query for RAG to focus on the file
                rag_query = query
                if request.filePath:
                    # Use the file path to get relevant context about the file
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(f"Modified RAG query to focus on file: {request.filePath}")

                # Try to perform RAG retrieval
                try:
                    # This will use the actual RAG implementation
                    retrieved_documents = request_rag(rag_query, language=request.language)

                    if retrieved_documents and retrieved_documents[0].documents:
                        # Format context for the prompt in a more structured way
                        documents = retrieved_documents[0].documents
                        logger.info(f"Retrieved {len(documents)} documents")
                        retrieved_documents_count = len(documents)

                        # Group documents by file path
                        docs_by_file = {}
                        for doc in documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(doc)

                        # Format context text with file path grouping
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            # Add file header with metadata
                            header = f"## File Path: {file_path}\n\n"
                            # Add document content
                            content = "\n\n".join([doc.text for doc in docs])

                            context_parts.append(f"{header}{content}")

                        # Join all parts with clear separation
                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                        
                        # Proactively truncate context_text if it's too large
                        context_text, _ = truncate_text_by_tokens(context_text, MAX_RAG_CONTEXT_TOKENS, is_ollama_provider)
                        logger.info(f"RAG context text token count after initial truncation: {count_tokens(context_text, is_ollama_provider)}")
                    else:
                        logger.warning("No documents retrieved from RAG")
                except Exception as e:
                    logger.error(f"Error in RAG retrieval: {str(e)}")
                    # Continue without RAG if there's an error

            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                context_text = ""

        # Get repository information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # Determine repository type
        repo_type = request.type

        # Get language information
        language_code = request.language or "en"
        language_name = {
            "en": "English",
            "ja": "Japanese (日本語)",
            "zh": "Mandarin Chinese (中文)",
            "es": "Spanish (Español)",
            "kr": "Korean (한국어)",
            "vi": "Vietnamese (Tiếng Việt)"
        }.get(language_code, "English")

        # Create system prompt
        if is_deep_research:
            # Check if this is the first iteration
            is_first_iteration = research_iteration == 1

            # Check if this is the final iteration
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

        # Fetch file content if provided
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")
                # Continue without file content if there's an error

        # Format conversation history
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

        # Create the prompt with context
        prompt = "" # Initialize empty prompt
        current_prompt_tokens = 0

        # Determine if the query is a large instructional prompt (e.g., wiki generation template)
        is_master_instructional_query = False
        if last_message_content_tokens > 3000 and \
           ("expert technical writer" in query.lower() or \
            "[wiki_page_topic]" in query.lower() or \
            "<details>" in query.lower() or \
            "relevant source files" in query.lower()):
            is_master_instructional_query = True
            logger.info("Master instructional query detected. Using it as the primary prompt base.")
            
        # The /no_think prefix appears to be for Adalflow/Ollama. Add it conditionally or ensure client handles it.
        # For now, let's assume it's handled by the client if needed based on provider.
        # Update: It's added specifically for ollama later.

        # Base prompt material
        if is_master_instructional_query:
            # The query (last_message.content) is the main set of instructions
            prompt_base_text = query
        else:
            # For regular chat, use the backend-defined system_prompt and then the query
            prompt_base_text = f"{system_prompt}\\n\\n<query>\\n{query}\\n</query>"
        
        prompt = f"{prompt_base_text}\\n\\n" # Add trailing newlines for separation
        current_prompt_tokens = count_tokens(prompt, is_ollama_provider)

        # Reserve space for "Assistant: " and a small buffer
        RESERVED_FOR_ASSISTANT_TAG = count_tokens("Assistant: ", is_ollama_provider) + 50

        # Add conversation history if it fits
        if conversation_history:
            conv_history_full_tag = f"<conversation_history>\\n{conversation_history}</conversation_history>\\n\\n"
            conv_history_tokens = count_tokens(conv_history_full_tag, is_ollama_provider)
            if current_prompt_tokens + conv_history_tokens < MAX_OVERALL_PROMPT_TOKENS - RESERVED_FOR_ASSISTANT_TAG:
                prompt = conv_history_full_tag + prompt # Prepend history
                current_prompt_tokens += conv_history_tokens
            else:
                logger.warning(f"Conversation history ({conv_history_tokens} tokens) too large for overall prompt, omitting.")

        # Add file content if it fits (prepend before RAG context, but after history)
        # This order might need adjustment based on how instructions in a master_query expect file content.
        # For now, prepending contextual items to the main query/instruction block.
        temp_prompt_after_history = prompt
        prompt_after_file_content = ""

        if file_content:
            file_content_full_tag = f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{file_content}\\n</currentFileContent>\\n\\n"
            file_content_tokens = count_tokens(file_content_full_tag, is_ollama_provider)
            
            if current_prompt_tokens + file_content_tokens < MAX_OVERALL_PROMPT_TOKENS - RESERVED_FOR_ASSISTANT_TAG:
                prompt_after_file_content = file_content_full_tag + temp_prompt_after_history
                current_prompt_tokens += file_content_tokens
            else:
                remaining_for_file = MAX_OVERALL_PROMPT_TOKENS - current_prompt_tokens - RESERVED_FOR_ASSISTANT_TAG
                if remaining_for_file > 200: # Only add if meaningful space
                    truncated_file_text, _ = truncate_text_by_tokens(file_content, remaining_for_file - count_tokens(f"<currentFileContent path=\\\"{request.filePath}\\\">\\n\\n</currentFileContent>\\n\\n", is_ollama_provider), is_ollama_provider)
                    if truncated_file_text.strip():
                        file_content_full_tag_truncated = f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{truncated_file_text}\\n</currentFileContent>\\n\\n"
                        prompt_after_file_content = file_content_full_tag_truncated + temp_prompt_after_history
                        current_prompt_tokens += count_tokens(file_content_full_tag_truncated, is_ollama_provider)
                        logger.warning(f"Truncated file content for {request.filePath} to fit overall prompt limit.")
                    else:
                        prompt_after_file_content = temp_prompt_after_history # File content was too big or truncated to empty
                        logger.warning(f"File content for {request.filePath} too large or truncated to empty, omitting.")
                else:
                    prompt_after_file_content = temp_prompt_after_history
                    logger.warning(f"File content for {request.filePath} ({file_content_tokens} tokens) too large, omitting due to insufficient remaining space.")
            prompt = prompt_after_file_content
        else:
            prompt = temp_prompt_after_history

        # Add RAG context (already truncated by MAX_RAG_CONTEXT_TOKENS) if it fits
        # Prepend RAG context so it appears before the main query/instructions if not a master query,
        # or before the main master_query block.
        temp_prompt_before_rag = prompt
        prompt_after_rag = ""

        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"

        if context_text.strip(): 
            context_full_tag = f"{CONTEXT_START}\\n{context_text}\\n{CONTEXT_END}\\n\\n"
            context_tokens = count_tokens(context_full_tag, is_ollama_provider)

            if current_prompt_tokens + context_tokens < MAX_OVERALL_PROMPT_TOKENS - RESERVED_FOR_ASSISTANT_TAG:
                prompt_after_rag = context_full_tag + temp_prompt_before_rag
                current_prompt_tokens += context_tokens
            else:
                remaining_for_context = MAX_OVERALL_PROMPT_TOKENS - current_prompt_tokens - RESERVED_FOR_ASSISTANT_TAG
                if remaining_for_context > 200: 
                    # context_text was already truncated once by MAX_RAG_CONTEXT_TOKENS.
                    # This further truncation is for the overall limit.
                    # We need to subtract the token size of the context tags themselves.
                    tag_tokens = count_tokens(f"{CONTEXT_START}\\n\\n{CONTEXT_END}\\n\\n", is_ollama_provider)
                    further_truncated_context, _ = truncate_text_by_tokens(context_text, remaining_for_context - tag_tokens, is_ollama_provider)
                    if further_truncated_context.strip():
                        context_full_tag_further_truncated = f"{CONTEXT_START}\\n{further_truncated_context}\\n{CONTEXT_END}\\n\\n"
                        prompt_after_rag = context_full_tag_further_truncated + temp_prompt_before_rag
                        current_prompt_tokens += count_tokens(context_full_tag_further_truncated, is_ollama_provider)
                        logger.warning("Further truncated RAG context to fit overall prompt limit.")
                    else:
                        prompt_after_rag = temp_prompt_before_rag
                        logger.warning("RAG context too large or truncated to empty for overall limit, omitting.")
                        prompt_after_rag = "<note>Retrieval augmentation omitted due to overall prompt size constraints.</note>\\n\\n" + prompt_after_rag

                else:
                    prompt_after_rag = temp_prompt_before_rag
                    logger.warning("RAG context too large for overall prompt limit (insufficient space), omitting.")
                    prompt_after_rag = "<note>Retrieval augmentation omitted due to overall prompt size constraints.</note>\\n\\n" + prompt_after_rag
            prompt = prompt_after_rag
        elif not input_too_large : 
             logger.info("No documents retrieved from RAG.")
             prompt = "<note>Answering without retrieval augmentation (no relevant documents found or input was too large for RAG).</note>\\n\\n" + prompt
        elif input_too_large:
            logger.info("RAG skipped because initial user message content was too large.")
            prompt = "<note>Answering without retrieval augmentation (initial query was too large to support RAG).</note>\\n\\n" + prompt
        
        # Add the /no_think prefix now if it wasn't part of the base material (e.g. master query)
        # This needs to be handled carefully. Adalflow examples suggest it's part of the input string.
        # If it's already in `prompt` because `query` (master prompt) contained it, fine.
        # If not, and needed by a specific client, it should be added. The Ollama client adds it later.
        # For Gemini, it's not standard. Let's ensure it's NOT added for Gemini here unless query had it.
        # The existing code adds it specifically for Ollama later, which is fine.
        # The initial f"/no_think {base_prompt_material}..." was removed.
        # If base_prompt_material was `query` and `query` started with /no_think, it would be there.
        # If base_prompt_material was system_prompt + query, system_prompt needs to include it or not.
        # The original `system_prompt` variables do NOT include /no_think.
        # The old code: `prompt = f"/no_think {system_prompt}\n\n"`
        # Let's assume `/no_think` is an Adalflow specific convention which is handled by the Adalflow client adapter code if provider is ollama etc.
        # So, we don't need to add it universally here.

        prompt += "Assistant: " # Final mandatory part
        
        final_prompt_tokens = count_tokens(prompt, is_ollama_provider)
        logger.info(f"Final assembled prompt token count: {final_prompt_tokens}")
        if final_prompt_tokens > MAX_OVERALL_PROMPT_TOKENS:
            logger.error(f"CRITICAL: Final prompt still exceeds MAX_OVERALL_PROMPT_TOKENS ({final_prompt_tokens} > {MAX_OVERALL_PROMPT_TOKENS}) despite truncation efforts. This may lead to API errors.")
            # Potentially, we could send an error message back to the user here immediately if the prompt is catastrophically large.
            # For now, we let it proceed to the LLM, which will likely error out, and the existing fallback logic might catch it.

        model_config = get_model_config(request.provider, request.model)["model_kwargs"]

        if request.provider == "ollama":
            prompt += " /no_think"

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

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "openrouter":
            logger.info(f"Using OpenRouter with model: {request.model}")

            # Check if OpenRouter API key is set
            if not os.environ.get("OPENROUTER_API_KEY"):
                logger.warning("OPENROUTER_API_KEY environment variable is not set, but continuing with request")
                # We'll let the OpenRouterClient handle this and return a friendly error message

            model = OpenRouterClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"]
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "openai":
            logger.info(f"Using Openai protocol with model: {request.model}")

            # Check if an API key is set for Openai
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY environment variable is not set, but continuing with request")
                # We'll let the OpenAIClient handle this and return an error message

            # Initialize Openai client
            model = OpenAIClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"]
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        else:
            # Initialize Google Generative AI model
            model = genai.GenerativeModel(
                model_name=model_config["model"],
                generation_config={
                    "temperature": model_config["temperature"],
                    "top_p": model_config["top_p"],
                    "top_k": model_config["top_k"]
                }
            )

        # Process the response based on the provider
        try:
            # Introduce a delay to help manage potential RPM limits
            # 4 seconds delay aims for ~15 RPM if calls are back-to-back.
            # This is a server-side safeguard; frontend pacing is also crucial.
            logger.info(f"Introducing a 4-second delay before LLM call for {request.provider}...")
            await asyncio.sleep(4) 

            if request.provider == "ollama":
                # Get the response and handle it properly using the previously created api_kwargs
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                # Handle streaming response from Ollama
                async for chunk in response:
                    text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                    if text and not text.startswith('model=') and not text.startswith('created_at='):
                        text = text.replace('<think>', '').replace('</think>', '')
                        await websocket.send_text(text)
                # Explicitly close the WebSocket connection after the response is complete
                await websocket.close()
            elif request.provider == "openrouter":
                try:
                    # Get the response and handle it properly using the previously created api_kwargs
                    logger.info("Making OpenRouter API call")
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # Handle streaming response from OpenRouter
                    async for chunk in response:
                        await websocket.send_text(chunk)
                    # Explicitly close the WebSocket connection after the response is complete
                    await websocket.close()
                except Exception as e_openrouter:
                    logger.error(f"Error with OpenRouter API: {str(e_openrouter)}")
                    error_msg = f"\nError with OpenRouter API: {str(e_openrouter)}\n\nPlease check that you have set the OPENROUTER_API_KEY environment variable with a valid API key."
                    await websocket.send_text(error_msg)
                    # Close the WebSocket connection after sending the error message
                    await websocket.close()
            elif request.provider == "openai":
                try:
                    # Get the response and handle it properly using the previously created api_kwargs
                    logger.info("Making Openai API call")
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # Handle streaming response from Openai
                    async for chunk in response:
                        choices = getattr(chunk, "choices", [])
                        if len(choices) > 0:
                            delta = getattr(choices[0], "delta", None)
                            if delta is not None:
                                text = getattr(delta, "content", None)
                                if text is not None:
                                    await websocket.send_text(text)
                    # Explicitly close the WebSocket connection after the response is complete
                    await websocket.close()
                except Exception as e_openai:
                    logger.error(f"Error with Openai API: {str(e_openai)}")
                    error_msg = f"\nError with Openai API: {str(e_openai)}\n\nPlease check that you have set the OPENAI_API_KEY environment variable with a valid API key."
                    await websocket.send_text(error_msg)
                    # Close the WebSocket connection after sending the error message
                    await websocket.close()
            else:
                # Generate streaming response
                response = model.generate_content(prompt, stream=True)
                # Stream the response
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        await websocket.send_text(chunk.text)
                # Explicitly close the WebSocket connection after the response is complete
                await websocket.close()

        except Exception as e_outer:
            logger.error(f"Error in streaming response: {str(e_outer)}")
            error_message = str(e_outer)

            # Check for token limit errors
            if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                # If we hit a token limit error, try again without context
                logger.warning("Token limit exceeded, retrying without context")
                try:
                    # Create a simplified prompt without context
                    simplified_prompt = ""
                    base_simplified_prompt_tokens = 0
                    
                    # Use system_prompt for fallback, not the (potentially huge) original query if it was a master_instructional_query
                    simplified_prompt_base_text = f"{system_prompt}\\n\\n<query>\\n{query}\\n</query>"
                    simplified_prompt = f"{simplified_prompt_base_text}\\n\\n"
                    base_simplified_prompt_tokens = count_tokens(simplified_prompt, is_ollama_provider)

                    if conversation_history:
                        conv_history_full_tag_simplified = f"<conversation_history>\\n{conversation_history}</conversation_history>\\n\\n"
                        conv_history_tokens_simplified = count_tokens(conv_history_full_tag_simplified, is_ollama_provider)
                        if base_simplified_prompt_tokens + conv_history_tokens_simplified < MAX_OVERALL_PROMPT_TOKENS - RESERVED_FOR_ASSISTANT_TAG:
                             simplified_prompt = conv_history_full_tag_simplified + simplified_prompt # Prepend
                             base_simplified_prompt_tokens += conv_history_tokens_simplified
                        else:
                            logger.warning("Conversation history too large for simplified fallback, omitting.")

                    if request.filePath and file_content: # Use original file_content, truncate as needed
                        file_content_full_tag_simplified = f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{file_content}\\n</currentFileContent>\\n\\n"
                        file_content_tokens_simplified = count_tokens(file_content_full_tag_simplified, is_ollama_provider)
                        
                        if base_simplified_prompt_tokens + file_content_tokens_simplified < MAX_OVERALL_PROMPT_TOKENS - RESERVED_FOR_ASSISTANT_TAG:
                            simplified_prompt = file_content_full_tag_simplified + simplified_prompt # Prepend
                            base_simplified_prompt_tokens += file_content_tokens_simplified
                        else:
                            remaining_for_file_simplified = MAX_OVERALL_PROMPT_TOKENS - base_simplified_prompt_tokens - RESERVED_FOR_ASSISTANT_TAG
                            if remaining_for_file_simplified > 200:
                                tags_size = count_tokens(f"<currentFileContent path=\\\"{request.filePath}\\\">\\n\\n</currentFileContent>\\n\\n", is_ollama_provider)
                                truncated_file_text_simplified, _ = truncate_text_by_tokens(file_content, remaining_for_file_simplified - tags_size, is_ollama_provider)
                                if truncated_file_text_simplified.strip():
                                    file_content_full_tag_trunc_simplified = f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{truncated_file_text_simplified}\\n</currentFileContent>\\n\\n"
                                    simplified_prompt = file_content_full_tag_trunc_simplified + simplified_prompt # Prepend
                                    base_simplified_prompt_tokens += count_tokens(file_content_full_tag_trunc_simplified, is_ollama_provider)
                                    logger.info("Truncated file content for simplified fallback prompt.")
                                else:
                                     logger.warning("File content too large or truncated to empty for simplified fallback, omitting.")
                            else:
                                logger.warning("File content too large for simplified fallback (insufficient space), omitting.")

                    simplified_prompt += "<note>Answering without retrieval augmentation due to input size constraints.</note>\\n\\n"
                    simplified_prompt += "Assistant: " # Final part for simplified prompt
                    
                    logger.info(f"Fallback simplified prompt token count: {count_tokens(simplified_prompt, is_ollama_provider)}")

                    if request.provider == "ollama":
                        # Create new api_kwargs with the simplified prompt
                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                            input=simplified_prompt,
                            model_kwargs=model_kwargs,
                            model_type=ModelType.LLM
                        )

                        # Get the response using the simplified prompt
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                        # Handle streaming fallback_response from Ollama
                        async for chunk in fallback_response:
                            text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                            if text and not text.startswith('model=') and not text.startswith('created_at='):
                                text = text.replace('<think>', '').replace('</think>', '')
                                await websocket.send_text(text)
                    elif request.provider == "openrouter":
                        try:
                            # Create new api_kwargs with the simplified prompt
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # Get the response using the simplified prompt
                            logger.info("Making fallback OpenRouter API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # Handle streaming fallback_response from OpenRouter
                            async for chunk in fallback_response:
                                await websocket.send_text(chunk)
                        except Exception as e_fallback:
                            logger.error(f"Error with OpenRouter API fallback: {str(e_fallback)}")
                            error_msg = f"\nError with OpenRouter API fallback: {str(e_fallback)}\n\nPlease check that you have set the OPENROUTER_API_KEY environment variable with a valid API key."
                            await websocket.send_text(error_msg)
                    elif request.provider == "openai":
                        try:
                            # Create new api_kwargs with the simplified prompt
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # Get the response using the simplified prompt
                            logger.info("Making fallback Openai API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # Handle streaming fallback_response from Openai
                            async for chunk in fallback_response:
                                text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                                await websocket.send_text(text)
                        except Exception as e_fallback:
                            logger.error(f"Error with Openai API fallback: {str(e_fallback)}")
                            error_msg = f"\nError with Openai API fallback: {str(e_fallback)}\n\nPlease check that you have set the OPENAI_API_KEY environment variable with a valid API key."
                            await websocket.send_text(error_msg)
                    else:
                        # Initialize Google Generative AI model
                        model_config = get_model_config(request.provider, request.model)
                        fallback_model = genai.GenerativeModel(
                            model_name=model_config["model"],
                            generation_config={
                                "temperature": model_config["model_kwargs"].get("temperature", 0.7),
                                "top_p": model_config["model_kwargs"].get("top_p", 0.8),
                                "top_k": model_config["model_kwargs"].get("top_k", 40)
                            }
                        )

                        # Get streaming response using simplified prompt
                        fallback_response = fallback_model.generate_content(simplified_prompt, stream=True)
                        # Stream the fallback response
                        for chunk in fallback_response:
                            if hasattr(chunk, 'text'):
                                await websocket.send_text(chunk.text)
                except Exception as e2:
                    logger.error(f"Error in fallback streaming response: {str(e2)}")
                    await websocket.send_text(f"\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts.")
                    # Close the WebSocket connection after sending the error message
                    await websocket.close()
            else:
                # For other errors, return the error message
                await websocket.send_text(f"\nError: {error_message}")
                # Close the WebSocket connection after sending the error message
                await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        try:
            await websocket.send_text(f"Error: {str(e)}")
            await websocket.close()
        except:
            pass
