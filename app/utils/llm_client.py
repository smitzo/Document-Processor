"""
LLM Client Utilities
====================
Shared helpers for calling the Google Generative AI (Gemini) API.
Used by all extraction agents and the segregator.
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Any, Union

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configure API key
api_key = settings.google_api_key or os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    logger.info("[LLM] Google Generative AI configured successfully")
else:
    logger.warning("[LLM] No GOOGLE_API_KEY found. Please set it in .env or environment variables.")

# ---------------------------------------------------------------------------
# Vision message builder
# ---------------------------------------------------------------------------

def build_vision_message(prompt: str, images: list[str]) -> list[Union[str, dict]]:
    """
    Build a Google Generative AI content list with base64 PNG images + a text prompt.

    Parameters
    ----------
    prompt : str
        The text instruction for the model.
    images : list[str]
        Ordered list of base64-encoded PNG strings (one per page).

    Returns
    -------
    list — a content list ready to pass to Gemini API.
    """
    content = []
    
    # Add images first
    for b64 in images:
        content.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": b64,
            }
        })
    
    # Add text prompt
    content.append(prompt)
    
    return content


def build_simple_message(prompt: str) -> str:
    """Build a simple text-only message."""
    return prompt


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """
    Robustly extract JSON from model output.
    Handles markdown fences and leading/trailing noise.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = re.sub(r"```", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first balanced JSON object or array inside the text
    candidate = _find_balanced_json(text)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract valid JSON from model response. "
        f"Response preview:\n{text[:500]}"
    )


def _find_balanced_json(text: str) -> str | None:
    """Return the first balanced JSON object/array substring, if any."""
    start = None
    opening = ""
    closing = ""
    stack: list[str] = []
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                opening = "{"
                closing = "}"
                stack = [closing]
            elif ch == "[":
                start = idx
                opening = "["
                closing = "]"
                stack = [closing]
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opening:
            stack.append(closing)
            continue

        if ch == "{":
            stack.append("}")
            continue

        if ch == "[":
            stack.append("]")
            continue

        if stack and ch == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                return text[start:idx + 1]

    return None


# ---------------------------------------------------------------------------
# Main call wrapper
# ---------------------------------------------------------------------------

def call_llm_json(
    system_prompt: str,
    messages: list[Any],
    model: str = "gemini-3-flash-preview",
    max_tokens: int = 2048,
) -> Any:
    """
    Call the Google Generative AI (Gemini) API and return a parsed JSON object/array.

    Parameters
    ----------
    system_prompt : str
    messages      : list — already-formatted Gemini content list
    model         : str
    max_tokens    : int

    Returns
    -------
    Parsed Python object (dict or list).

    Raises
    ------
    ValueError if the response cannot be parsed as JSON.
    """
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not configured. Please set it in .env file.")
    
    logger.debug("[LLM] Calling model=%s max_tokens=%d", model, max_tokens)
    
    # Prepend system instruction to messages
    full_messages = [system_prompt] + messages
    
    # Get the model
    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for consistent JSON extraction
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    # Generate content
    try:
        response = gemini_model.generate_content(full_messages)
        raw_text = response.text
        logger.debug("[LLM] Raw response length: %d chars", len(raw_text))
        logger.debug("[LLM] Raw response preview: %s", raw_text[:1000])
        return _extract_json(raw_text)
    except Exception as e:
        logger.exception("[LLM] API call failed")
        raise ValueError(f"Gemini API call failed: {e}")


def call_llm_json_text_only(
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-3-flash-preview",
    max_tokens: int = 2048,
) -> Any:
    """
    Call the Google Generative AI (Gemini) API with text-only input and return parsed JSON.

    Parameters
    ----------
    system_prompt : str
    user_prompt   : str
    model         : str
    max_tokens    : int

    Returns
    -------
    Parsed Python object (dict or list).
    """
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not configured. Please set it in .env file.")
    
    logger.debug("[LLM] Calling text-only model=%s max_tokens=%d", model, max_tokens)
    
    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": 0.1,
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    try:
        # Prepend system prompt to user prompt
        full_message = f"{system_prompt}\n\n{user_prompt}"
        response = gemini_model.generate_content(full_message)
        raw_text = response.text
        logger.debug("[LLM] Raw text-only response preview: %s", raw_text[:1000])
        return _extract_json(raw_text)
    except Exception as e:
        logger.exception("[LLM] API call failed")
        raise ValueError(f"Gemini API call failed: {e}")
