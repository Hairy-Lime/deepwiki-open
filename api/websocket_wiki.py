import logging
import os
from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator
from urllib.parse import unquote
import asyncio
import time # Added for rate limiter
from collections import deque # Added for rate limiter

import google.generativeai as genai
from google.generativeai.types import BlockedPromptException, generation_types
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, WebSocketState
from pydantic import BaseModel, Field

import fastapi # Debug import
print(f"[DEBUG] FastAPI version in websocket_wiki.py: {fastapi.__version__}") # Debug print

from api.config import get_model_config as get_llm_config # Alias to avoid confusion
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.rag import RAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(lineno)d %(filename)s:%(funcName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Rate Limiter Configuration ---
LLM_RPM_LIMITS: Dict[str, int] = {
    "google": 150,
    "openai": 60, 
    "openrouter": 60, 
    "ollama": 1000, 
    "default": 30
}
LLM_REQUEST_TIMESTAMPS: Dict[str, deque] = {provider: deque() for provider in LLM_RPM_LIMITS.keys()}
LLM_REQUEST_TIMESTAMPS["default"] = deque()
LLM_RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_LOCK = asyncio.Lock()

async def enforce_llm_rate_limit(provider: str):
    async with RATE_LIMIT_LOCK:
        now = time.monotonic()
        current_rpm_limit = LLM_RPM_LIMITS.get(provider, LLM_RPM_LIMITS["default"])
        provider_timestamps = LLM_REQUEST_TIMESTAMPS.get(provider, LLM_REQUEST_TIMESTAMPS["default"])
        while provider_timestamps and provider_timestamps[0] <= now - LLM_RATE_LIMIT_WINDOW_SECONDS:
            provider_timestamps.popleft()
        if len(provider_timestamps) >= current_rpm_limit:
            oldest_relevant_timestamp = provider_timestamps[0]
            wait_time = (oldest_relevant_timestamp + LLM_RATE_LIMIT_WINDOW_SECONDS) - now
            if wait_time > 0:
                logger.info(f"Rate limit for {provider} reached ({current_rpm_limit} RPM). Waiting for {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)
                now = time.monotonic()
                while provider_timestamps and provider_timestamps[0] <= now - LLM_RATE_LIMIT_WINDOW_SECONDS:
                    provider_timestamps.popleft()
        provider_timestamps.append(now)

# --- Token Budget Configuration ---
MODEL_MAX_CONTEXT_WINDOW_TOKENS = 32000 
DESIRED_MAX_OUTPUT_TOKENS = 4096    
MAX_EFFECTIVE_PROMPT_TOKENS = MODEL_MAX_CONTEXT_WINDOW_TOKENS - DESIRED_MAX_OUTPUT_TOKENS

BUDGET_SYSTEM_PROMPT_TARGET = 4000
BUDGET_QUERY_TARGET = 8000 
BUDGET_FILE_CONTENT_TARGET = 7000
BUDGET_SUMMARIZED_FILE_CONTENT_TARGET = 2000
BUDGET_HISTORY_TARGET = 5000
BUDGET_RAG_CONTEXT_TARGET = 10000
MIN_TOKENS_FOR_QUERY = 100
MIN_TOKENS_FOR_CONTEXT = 200

# --- Helper Functions ---
def get_token_count_for_provider(text: str, provider: str, model_name: Optional[str] = None) -> int:
    is_ollama = provider == "ollama"
    return count_tokens(text, is_ollama_provider=is_ollama)

def truncate_text_by_tokens(text: str, max_tokens: int, provider: str, model_name: Optional[str] = None) -> Tuple[str, bool, int]:
    if not text: return "", False, 0
    original_tokens = get_token_count_for_provider(text, provider, model_name)
    if original_tokens <= max_tokens: return text, False, original_tokens
    estimated_chars_per_token = 3.5 
    chars_to_keep = int(max_tokens * estimated_chars_per_token)
    truncated_text = text[:chars_to_keep]
    current_tokens = get_token_count_for_provider(truncated_text, provider, model_name)
    loop_safety = 0
    while current_tokens > max_tokens and len(truncated_text) > 10 and loop_safety < 20:
        excess_chars = int(len(truncated_text) * 0.1)
        truncated_text = truncated_text[:-max(10, excess_chars)] 
        current_tokens = get_token_count_for_provider(truncated_text, provider, model_name)
        loop_safety += 1
    if current_tokens > max_tokens:
        words = text.split()
        truncated_words = []
        current_word_tokens = 0
        for word in words:
            word_plus_space = word + " "
            word_tokens = get_token_count_for_provider(word_plus_space, provider, model_name)
            if current_word_tokens + word_tokens <= max_tokens:
                truncated_words.append(word)
                current_word_tokens += word_tokens
            else: break
        truncated_text = " ".join(truncated_words)
        current_tokens = get_token_count_for_provider(truncated_text, provider, model_name)
    was_truncated = current_tokens < original_tokens
    suffix = "... (truncated)" if was_truncated else ""
    if was_truncated: logger.warning(f"Truncated text from {original_tokens} to {current_tokens} tokens (limit: {max_tokens}).")
    return truncated_text + suffix, was_truncated, current_tokens

async def summarize_content_with_llm(
    content_to_summarize: str, 
    contextual_query: str, 
    file_path_hint: str,
    target_token_count: int,
    request_provider: str, 
    request_model_name: Optional[str],
    language_name: str
) -> Tuple[Optional[str], int]:
    logger.info(f"Attempting to summarize content from '{file_path_hint}' for query context, target tokens: {target_token_count}")
    summarization_system_prompt = f"""You are an expert text summarizer. Your task is to concisely summarize the following content from the file '{file_path_hint}'. 
The summary MUST be relevant to this user query: '{contextual_query}'. 
Focus on extracting essential details, function definitions, configurations, or explanations that directly help address or understand the query. 
Aim for a summary of approximately {target_token_count // 2} to {target_token_count} tokens. 
Respond ONLY with the summary text, without any preamble. Respond in {language_name}."""
    summarization_prompt = f"{summarization_system_prompt}\n\n<CONTENT_TO_SUMMARIZE>\n{content_to_summarize}\n</CONTENT_TO_SUMMARIZE>\n\nSummary:"
    summary_provider = request_provider
    summary_model_name = request_model_name
    try:
        summary_config = get_llm_config(summary_provider, summary_model_name)["model_kwargs"]
        api_kwargs_summary: Dict[str, Any] = {}
        llm_client_summary: Any = None
        if summary_provider == "google":
            if not google_api_key: return None, 0
            llm_client_summary = genai.GenerativeModel(
                model_name=summary_config.get("model", "gemini-1.5-flash-latest"),
                generation_config={"temperature": 0.3, "top_p": 0.9, "max_output_tokens": target_token_count + 500}
            )
            await enforce_llm_rate_limit(summary_provider)
            response = await llm_client_summary.generate_content_async(summarization_prompt)
            summarized_text = response.text if hasattr(response, 'text') else ""
        elif summary_provider in ["openai", "openrouter", "ollama"]:
            if summary_provider == "openai": llm_client_summary = OpenAIClient()
            elif summary_provider == "openrouter": llm_client_summary = OpenRouterClient()
            else: llm_client_summary = OllamaClient()
            model_kwargs_summary = {
                "model": summary_config.get("model"), 
                "stream": False, 
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": target_token_count + 500
            }
            if summary_provider == "ollama":
                model_kwargs_summary["options"] = {"temperature": 0.3, "top_p": 0.9, "num_ctx": summary_config.get("options", {}).get("num_ctx", 4096)}
            api_kwargs_summary = llm_client_summary.convert_inputs_to_api_kwargs(input=summarization_prompt, model_kwargs=model_kwargs_summary, model_type=ModelType.LLM)
            await enforce_llm_rate_limit(summary_provider)
            response = await llm_client_summary.acall(api_kwargs=api_kwargs_summary, model_type=ModelType.LLM)
            summarized_text = response.choices[0].message.content if summary_provider != "ollama" and response.choices else getattr(response, 'response', "")
        else:
            logger.error(f"Unsupported provider for summarization: {summary_provider}")
            return None, 0
        if summarized_text:
            final_summary_tokens = get_token_count_for_provider(summarized_text, summary_provider, summary_model_name)
            logger.info(f"Successfully summarized content from '{file_path_hint}'. Original approx tokens: {get_token_count_for_provider(content_to_summarize, summary_provider, summary_model_name)}, Summary tokens: {final_summary_tokens}")
            return summarized_text, final_summary_tokens
        else:
            logger.warning(f"LLM summarization for '{file_path_hint}' returned empty content.")
            return None, 0
    except Exception as e_summary:
        logger.error(f"Error during LLM summarization for '{file_path_hint}': {str(e_summary)}", exc_info=True)
        return None, 0

google_api_key = os.environ.get('GOOGLE_API_KEY')
if google_api_key: genai.configure(api_key=google_api_key)
else: logger.warning("GOOGLE_API_KEY not found")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab', 'bitbucket')")
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")
    language: Optional[str] = Field("en", description="Language for content generation")
    excluded_dirs: Optional[str] = Field(None)
    excluded_files: Optional[str] = Field(None)
    included_dirs: Optional[str] = Field(None)
    included_files: Optional[str] = Field(None)

async def _process_chat_request_logic(request_model: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Core logic for processing chat requests and streaming LLM responses."""
    try:
        logger.info(f"Processing chat request logic for repo: {request_model.repo_url}, provider: {request_model.provider}, model: {request_model.model}")
        request_rag = RAG(provider=request_model.provider, model=request_model.model)
        excluded_dirs_list = [unquote(d) for d in (request_model.excluded_dirs or "").split('\\n') if d.strip()]
        excluded_files_list = [unquote(f) for f in (request_model.excluded_files or "").split('\\n') if f.strip()]
        included_dirs_list = [unquote(d) for d in (request_model.included_dirs or "").split('\\n') if d.strip()]
        included_files_list = [unquote(f) for f in (request_model.included_files or "").split('\\n') if f.strip()]
        request_rag.prepare_retriever(
            request_model.repo_url, request_model.type, request_model.token,
            excluded_dirs_list, excluded_files_list, included_dirs_list, included_files_list
        )
        logger.info(f"Retriever prepared for {request_model.repo_url}")

        if not request_model.messages or len(request_model.messages) == 0:
            yield "Error: No messages provided"
            return
        last_message = request_model.messages[-1]
        if last_message.role != "user":
            yield "Error: Last message must be from the user"
            return
        query = last_message.content

        is_deep_research = False
        research_iteration = 1
        for msg in request_model.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                if msg == request_model.messages[-1]:
                    query = query.replace("[DEEP RESEARCH]", "").strip()
        if is_deep_research:
            research_iteration = sum(1 for msg in request_model.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")
            if "continue" in query.lower() and "research" in query.lower():
                original_topic = None
                for msg_hist in request_model.messages:
                    if msg_hist.role == "user" and "continue" not in msg_hist.content.lower():
                        original_topic = msg_hist.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic for continuation: {original_topic}")
                        break
                if original_topic: query = original_topic; logger.info(f"Using original topic for continued research: {query}")
        
        assembled_prompt_parts = []
        current_total_tokens = 0
        provider = request_model.provider
        model_name = request_model.model
        repo_info_str = f"the {request_model.type} repository: {request_model.repo_url} ({request_model.repo_url.split('/')[-1] if '/' in request_model.repo_url else request_model.repo_url})"
        language_name = {"en": "English", "ja": "Japanese (日本語)", "zh": "Mandarin Chinese (中文)", "es": "Spanish (Español)", "kr": "Korean (한국어)", "vi": "Vietnamese (Tiếng Việt)"}.get(request_model.language or "en", "English")
        
        system_prompt = "" 
        if is_deep_research:
            is_first_iteration = research_iteration == 1
            is_final_iteration = research_iteration >= 5 
            if is_first_iteration:
                system_prompt = f"""<role>
You are an expert code analyst examining {repo_info_str}.
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
You are an expert code analyst examining {repo_info_str}.
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
You are an expert code analyst examining {repo_info_str}.
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
You are an expert code analyst examining {repo_info_str}.
You provide direct, concise, and accurate information about code repositories.
You NEVER start responses with markdown headers or code fences.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- Answer the user's question directly without ANY preamble or filler phrases.
- DO NOT include any rationale, explanation, or extra comments.
- DO NOT start with preambles like "Okay, here's a breakdown" or "Here's an explanation".
- DO NOT start with markdown headers like "## Analysis of..." or any file path references.
- DO NOT start with ```markdown code fences.
- DO NOT end your response with ``` closing fences.
- DO NOT start by repeating or acknowledging the question.
- JUST START with the direct answer to the question.
- Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer.
- For code analysis, organize your response with clear sections.
- Think step by step and structure your answer logically.
- Start with the most relevant information that directly addresses the user's query.
- Be precise and technical when discussing code.
</guidelines>

<style>
- Use concise, direct language.
- Prioritize accuracy over verbosity.
- When showing code, include line numbers and file paths when relevant.
- Use markdown formatting to improve readability.
</style>"""

        system_prompt_tokens = get_token_count_for_provider(system_prompt, provider, model_name)
        if system_prompt_tokens > min(BUDGET_SYSTEM_PROMPT_TARGET, MAX_EFFECTIVE_PROMPT_TOKENS):
            logger.error(f"System prompt ({system_prompt_tokens} tokens) is too large.")
            yield "Error: System prompt configuration is too large."
            return
        assembled_prompt_parts.append(system_prompt)
        current_total_tokens += system_prompt_tokens

        query_original_tokens = get_token_count_for_provider(query, provider, model_name)
        remaining_budget_for_query = MAX_EFFECTIVE_PROMPT_TOKENS - current_total_tokens
        query_budget_to_use = min(BUDGET_QUERY_TARGET, remaining_budget_for_query)
        truncated_query_text, query_was_truncated, query_final_tokens = truncate_text_by_tokens(query, query_budget_to_use, provider, model_name)
        if query_was_truncated and query_final_tokens < MIN_TOKENS_FOR_QUERY:
            logger.error(f"User query severely truncated (to {query_final_tokens} from {query_original_tokens}).")
            yield "Error: Query too long. Please shorten."
            return
        current_total_tokens += query_final_tokens
        if query_was_truncated:
            logger.info(f"User query was truncated from {query_original_tokens} to {query_final_tokens} tokens.")

        conversation_history_str = ""
        if request_model.messages and len(request_model.messages) > 1:
            temp_history_turns = []
            for i in range(0, len(request_model.messages) - 1, 2):
                if i + 1 < len(request_model.messages):
                    user_msg, assistant_msg = request_model.messages[i], request_model.messages[i+1]
                    if user_msg.role == "user" and assistant_msg.role == "assistant":
                        temp_history_turns.append(f"<turn>\n<user>{user_msg.content}</user>\n<assistant>{assistant_msg.content}</assistant>\n</turn>")
            temp_conv_history_tokens = 0
            final_history_parts = []
            remaining_budget_for_history = MAX_EFFECTIVE_PROMPT_TOKENS - current_total_tokens
            history_budget_to_use = min(BUDGET_HISTORY_TARGET, remaining_budget_for_history)
            for turn_str in reversed(temp_history_turns):
                turn_tokens = get_token_count_for_provider(turn_str, provider, model_name)
                if temp_conv_history_tokens + turn_tokens <= history_budget_to_use:
                    final_history_parts.insert(0, turn_str)
                    temp_conv_history_tokens += turn_tokens
                else:
                    logger.info("Truncating conversation history for prompt."); break
            if final_history_parts:
                conversation_history_str = "\n".join(final_history_parts)
                assembled_prompt_parts.append(f"\n\n<conversation_history>\n{conversation_history_str}</conversation_history>")
                current_total_tokens += temp_conv_history_tokens
                logger.info(f"Added history ({temp_conv_history_tokens} tokens).")

        file_content_str = ""
        file_tokens_to_add = 0
        if request_model.filePath:
            try:
                file_content_raw = get_file_content(request_model.repo_url, request_model.filePath, request_model.type, request_model.token)
                raw_file_tokens = get_token_count_for_provider(file_content_raw, provider, model_name)
                logger.info(f"Raw file content for {request_model.filePath}: {raw_file_tokens} tokens.")

                remaining_budget_for_file = MAX_EFFECTIVE_PROMPT_TOKENS - current_total_tokens
                target_file_budget = min(BUDGET_FILE_CONTENT_TARGET, remaining_budget_for_file)

                if raw_file_tokens > target_file_budget + 200: # Summarization threshold
                    logger.info(f"File {request_model.filePath} ({raw_file_tokens} tokens) too large. Summarizing.")
                    summarized_content, summarized_tokens = await summarize_content_with_llm(
                        file_content_raw, truncated_query_text, request_model.filePath,
                        min(BUDGET_SUMMARIZED_FILE_CONTENT_TARGET, remaining_budget_for_file), 
                        provider, model_name, language_name
                    )
                    if summarized_content and summarized_tokens > 0:
                        file_content_str, file_tokens_to_add = summarized_content, summarized_tokens
                    else: # Summarization failed or empty, fallback to truncation
                        file_content_str, _, file_tokens_to_add = truncate_text_by_tokens(file_content_raw, target_file_budget, provider, model_name)
                else: # Small enough for direct inclusion (or truncation)
                    file_content_str, _, file_tokens_to_add = truncate_text_by_tokens(file_content_raw, target_file_budget, provider, model_name)
                
                if file_tokens_to_add > 0:
                    assembled_prompt_parts.append(f"\n\n<currentFileContent path=\"{request_model.filePath}\">\n{file_content_str}\n</currentFileContent>")
                    current_total_tokens += file_tokens_to_add
                logger.info(f"Added file content for {request_model.filePath} ({file_tokens_to_add} tokens).")
            except Exception as e_file: logger.error(f"Error with file content for {request_model.filePath}: {e_file}", exc_info=True)

        context_text_str = ""
        last_message_initial_tokens = get_token_count_for_provider(request_model.messages[-1].content, provider, model_name)
        if last_message_initial_tokens <= 20000:
            try:
                rag_query_for_retrieval = truncated_query_text 
                if request_model.filePath: rag_query_for_retrieval = f"Context for {request_model.filePath}. Query: {truncated_query_text}"
                retrieved_data = request_rag(rag_query_for_retrieval, language=request_model.language)
                if retrieved_data and retrieved_data[0].documents:
                    docs = retrieved_data[0].documents; logger.info(f"RAG: {len(docs)} docs.")
                    temp_parts, temp_tokens = [], 0
                    remaining_rag_budget = MAX_EFFECTIVE_PROMPT_TOKENS - current_total_tokens
                    rag_budget = min(BUDGET_RAG_CONTEXT_TARGET, remaining_rag_budget)
                    for doc in docs:
                        doc_fmt = f"--- Source: {doc.meta_data.get('file_path', 'unknown')} ---\n{doc.text}\n--- End Source ---"
                        doc_t = get_token_count_for_provider(doc_fmt, provider, model_name)
                        if temp_tokens + doc_t <= rag_budget:
                            temp_parts.append(doc_fmt); temp_tokens += doc_t
                        else:
                            if rag_budget - temp_tokens > MIN_TOKENS_FOR_CONTEXT:
                                trunc_doc, _, trunc_t = truncate_text_by_tokens(doc_fmt, rag_budget - temp_tokens, provider, model_name)
                                if trunc_t > 0: temp_parts.append(trunc_doc); temp_tokens += trunc_t
                            logger.info(f"RAG budget hit. Added {len(temp_parts)} docs/parts."); break
                    if temp_parts: 
                        context_text_str = "\n\n".join(temp_parts)
                        assembled_prompt_parts.append(f"\n\n<START_OF_CONTEXT>\n{context_text_str}\n<END_OF_CONTEXT>")
                        current_total_tokens += temp_tokens; logger.info(f"Added RAG context ({temp_tokens} tokens).")
                    else: assembled_prompt_parts.append("\n\n<note>No RAG context fit budget.</note>")
                else: assembled_prompt_parts.append("\n\n<note>No RAG docs retrieved.</note>")
            except Exception as e_rag: logger.error(f"RAG error: {e_rag}", exc_info=True); assembled_prompt_parts.append(f"\n\n<note>RAG error: {e_rag}</note>")
        else: assembled_prompt_parts.append("\n\n<note>RAG skipped: large query.</note>")
        
        assembled_prompt_parts.append(f"\n\n<query>\n{truncated_query_text}\n</query>\n\nAssistant:")
        final_prompt = "".join(assembled_prompt_parts)
        final_calculated_tokens = get_token_count_for_provider(final_prompt, provider, model_name)
        logger.info(f"Final prompt: Budgeted tokens {current_total_tokens}, Recalculated {final_calculated_tokens} (Target: {MAX_EFFECTIVE_PROMPT_TOKENS})")
        if final_calculated_tokens > MODEL_MAX_CONTEXT_WINDOW_TOKENS: # Hard limit check
            logger.error(f"FATAL: Final prompt ({final_calculated_tokens}) exceeds model max window ({MODEL_MAX_CONTEXT_WINDOW_TOKENS}).")
            yield "Error: Assembled request is too large for the AI model. Please simplify significantly."
            return
        elif final_calculated_tokens > MAX_EFFECTIVE_PROMPT_TOKENS * 1.05:
             logger.warning(f"Final prompt ({final_calculated_tokens}) exceeds effective target by >5%.")

        model_config_details = get_llm_config(request_model.provider, request_model.model)
        llm_model_kwargs = model_config_details["model_kwargs"]
        api_kwargs: Dict[str, Any] = {}
        llm_client: Any = None
        
        if provider == "ollama":
            llm_client = OllamaClient()
            api_kwargs = llm_client.convert_inputs_to_api_kwargs(input=final_prompt, model_kwargs={"model": llm_model_kwargs["model"], "stream": True, "options": llm_model_kwargs.get("options", {})}, model_type=ModelType.LLM)
        elif provider == "openrouter":
            llm_client = OpenRouterClient()
            api_kwargs = llm_client.convert_inputs_to_api_kwargs(input=final_prompt, model_kwargs={"model": request_model.model, "stream": True, **llm_model_kwargs}, model_type=ModelType.LLM)
        elif provider == "openai":
            llm_client = OpenAIClient()
            api_kwargs = llm_client.convert_inputs_to_api_kwargs(input=final_prompt, model_kwargs={"model": request_model.model, "stream": True, **llm_model_kwargs, "max_tokens": DESIRED_MAX_OUTPUT_TOKENS}, model_type=ModelType.LLM)
        elif provider == "google":
            if not google_api_key: yield "Error: Google API key not configured."; return
            llm_client = genai.GenerativeModel(model_name=llm_model_kwargs["model"], generation_config={**llm_model_kwargs, "max_output_tokens": DESIRED_MAX_OUTPUT_TOKENS})
        else:
            yield f"Error: Unsupported provider '{provider}'"; return

        logger.info(f"Making LLM call for {provider} ({request_model.model}) with prompt tokens: {final_calculated_tokens}")
        await enforce_llm_rate_limit(provider)
        stream_had_content = False
        full_response_text = ""

        try:
            if provider == "google":
                response_stream = await llm_client.generate_content_async(final_prompt, stream=True)
                async for chunk in response_stream:
                    stream_had_content = True; chunk_text = ""
                    try:
                        candidate_obj = chunk.candidates[0] if chunk.candidates else None 
                        if candidate_obj and candidate_obj.finish_reason != generation_types.Candidate.FinishReason.UNSPECIFIED:
                            if candidate_obj.finish_reason == generation_types.Candidate.FinishReason.MAX_TOKENS:
                                logger.warning("Google API response stream truncated: MAX_TOKENS.")
                                yield "\n[Warning: Response truncated by AI model due to length limits.]"; 
                                break 
                            elif candidate_obj.finish_reason != generation_types.Candidate.FinishReason.STOP:
                                safety_ratings_str = str(candidate_obj.safety_ratings) if hasattr(candidate_obj, 'safety_ratings') else 'N/A'
                                logger.error(f"Google API stream stopped. Reason: {candidate_obj.finish_reason.name}. Details: {safety_ratings_str}")
                                yield f"\n[Error: AI model stopped generation - {candidate_obj.finish_reason.name}]"; 
                                stream_ended_by_error = True; break
                        
                        if chunk.parts:
                            for part in chunk.parts: chunk_text += getattr(part, 'text', '')
                        elif hasattr(chunk, 'text'): 
                            chunk_text = chunk.text
                        
                        if chunk_text: 
                            yield chunk_text
                            full_response_text += chunk_text

                    except BlockedPromptException as bpe:
                        logger.error(f"Google API blocked prompt during streaming: {bpe}", exc_info=True)
                        yield f"\n[Error: AI request blocked by Google's content safety filter. {bpe}]"; 
                        stream_ended_by_error = True; break
                    except ValueError as ve: 
                        logger.error(f"ValueError processing Google API chunk: {ve}. Chunk: {getattr(chunk, '__dict__', str(chunk))}. Prompt tokens: {final_calculated_tokens}", exc_info=True)
                        if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                            block_reason = chunk.prompt_feedback.block_reason.name
                            block_message = chunk.prompt_feedback.block_reason_message or ""
                            logger.error(f"Google API blocked prompt (from chunk feedback). Reason: {block_reason}, Message: {block_message}")
                            yield f"\n[Error: AI request possibly blocked by content safety filter ({block_reason}). {block_message}]"; 
                        else:
                            yield "\n[Error: Incomplete or malformed response from AI. The request might have been too long or hit a content filter.]"; 
                        stream_ended_by_error = True; break
                    except Exception as e_gc: 
                        logger.error(f"Unexpected error processing Google API chunk: {e_gc}", exc_info=True)
                        yield "\n[Error: Server error while processing AI response stream.]"; 
                        stream_ended_by_error = True; break
                
                if not stream_had_content and not stream_ended_by_error: # Check after loop if stream didn't error but had no content
                    logger.warning("Google stream yielded no content and did not error during iteration. Checking for upfront block if possible (though direct exception is more likely).")

            elif provider in ["ollama", "openrouter", "openai"]:
                response_stream = await llm_client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                async for chunk_data in response_stream:
                    stream_had_content = True; text_to_send = ""
                    if provider == "ollama": text_to_send = chunk_data.get('response', '') if isinstance(chunk_data, dict) else str(chunk_data)
                    elif provider == "openrouter": text_to_send = str(chunk_data) 
                    elif provider == "openai":
                        choices = getattr(chunk_data, "choices", [])
                        if choices and hasattr(choices[0], "delta") and choices[0].delta: text_to_send = choices[0].delta.content or ""
                    if text_to_send: yield text_to_send.replace('<think>', '').replace('</think>', ''); full_response_text += text_to_send
            
            if not stream_had_content: yield "\n[Note: AI model returned an empty response.]"
        
        except BlockedPromptException as e_bpe: yield f"\n[Error: Request blocked by content safety filter ({provider}). Details: {e_bpe}]"; logger.error(f'{provider} blocked: {e_bpe}', exc_info=True)
        except generation_types.StopCandidateException as e_sce: yield f"\n[Error: AI stopped unexpectedly ({provider}). Reason: {e_sce.finish_reason.name if e_sce.finish_reason else 'Unknown'}]"; logger.error(f'{provider} stopped: {e_sce}', exc_info=True)
        except Exception as e_llm: 
            err_msg = f"Error with {provider} AI model."
            if "context length" in str(e_llm).lower() or "token limit" in str(e_llm).lower(): err_msg += " Exceeded token limit."
            yield err_msg; logger.error(f'{provider} LLM error: {e_llm}', exc_info=True)
        finally:
            if full_response_text and len(full_response_text) < 500: logger.info(f"LLM {provider} full captured response (on completion/error): {full_response_text[:500]}")
            elif full_response_text: logger.info(f"LLM {provider} full captured response (on completion/error) was long, showing start: {full_response_text[:200]}...")

    except Exception as e_process_logic:
        logger.error(f"Unhandled error in _process_chat_request_logic: {str(e_process_logic)}", exc_info=True)
        yield f"Error: Server-side processing failed: {str(e_process_logic)}"


async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions.
    Uses the refactored _process_chat_request_logic for core operations.
    """
    await websocket.accept()
    request_model_ws: Optional[ChatCompletionRequest] = None
    try:
        request_data = await websocket.receive_json()
        request_model_ws = ChatCompletionRequest(**request_data)
        
        async for content_chunk in _process_chat_request_logic(request_model_ws):
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(content_chunk)
            else:
                logger.warning("WebSocket disconnected during streaming, stopping send.")
                break # Stop trying to send if client disconnected

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
    except HTTPException as http_exc:
        logger.error(f"HTTPException in WebSocket handler shell: {http_exc.detail}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.send_text(f"Error: {http_exc.detail}"); await websocket.close() 
            except: pass
    except Exception as e_ws_shell:
        logger.error(f"Unexpected error in WebSocket shell: {str(e_ws_shell)}", exc_info=True)
        if request_model_ws:
             logger.error(f"Failing request context (WS shell): Provider={request_model_ws.provider}, Model={request_model_ws.model}, Repo={request_model_ws.repo_url}")
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.send_text(f"An unexpected server error occurred. Please check server logs."); await websocket.close()
            except: pass
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
            logger.info("WebSocket connection explicitly closed by server at end of handler.")
