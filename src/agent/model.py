import os
from typing import Optional


def call_hf_chat(system: str, user_prompt: str, access_token: Optional[str] = None, model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    from huggingface_hub import InferenceClient

    if access_token is None:
        access_token = os.environ.get("HF_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("HF_ACCESS_TOKEN not set")

    hf_model = os.environ.get("HF_MODEL", model)
    # Seq2seq models don't support chat completions — fall back to default
    _SEQ2SEQ_PREFIXES = ("google/flan", "google/t5", "t5-", "facebook/bart")
    if any(hf_model.lower().startswith(p) for p in _SEQ2SEQ_PREFIXES):
        hf_model = model

    try:
        timeout_s = float(os.environ.get("HF_TIMEOUT_SECONDS", "40"))
        max_tokens = int(os.environ.get("HF_MAX_TOKENS", "256"))
        client = InferenceClient(model=hf_model, token=access_token, timeout=timeout_s)
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Hugging Face inference error: {e}")
    if isinstance(data, dict):
        if "error" in data:
            raise RuntimeError(data["error"])
        if "generated_text" in data:
            return data["generated_text"].strip()

    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"].strip()
        if isinstance(first, str):
            return first.strip()

    if isinstance(data, str):
        return data.strip()

    raise RuntimeError(f"Unexpected response from Hugging Face API: {data}")
