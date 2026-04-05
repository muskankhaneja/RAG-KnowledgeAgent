import os
import openai
from typing import Optional


def call_openai_chat(system: str, user_prompt: str, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> str:
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    resp = openai.ChatCompletion.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()
