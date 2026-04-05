import os
import json
from src.agent.retriever import query

try:
    from src.agent.model import call_openai_chat
    _HAS_OPENAI_CLIENT = True
except Exception:
    call_openai_chat = None
    _HAS_OPENAI_CLIENT = False


def main():
    q = "What is this repository about?"
    res = query('demo', q, top_k=5)
    # build context
    context_texts = []
    for proj, hits in res.items():
        for h in hits:
            context_texts.append(f"Project: {proj}\nSource: {h.get('source')}\nText: {h.get('text')}\nScore: {h.get('score')}\n---\n")
    context = "\n".join(context_texts)

    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key and _HAS_OPENAI_CLIENT:
        system = "You are an assistant that uses the provided context to answer the user's question concisely."
        prompt = f"Context:\n{context}\nUser question:\n{q}\nProvide a concise answer and cite sources if helpful."
        try:
            answer = call_openai_chat(system, prompt, api_key)
            out = {"retrieved": res, "answer": answer}
        except Exception as e:
            out = {"retrieved": res, "llm_error": str(e)}
    else:
        # simple fallback summarizer
        joined = " \n ".join([t for t in context_texts])
        summary = joined.strip()[:400]
        out = {"note": "OPENAI_API_KEY not set or client unavailable; returning fallback summary.", "retrieved": res, "summary": summary}

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
