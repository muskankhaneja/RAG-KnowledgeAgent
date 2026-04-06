import os
import json
from src.agent.retriever import query
from src.agent.model import call_hf_chat


def main():
    q = "What is this repository about?"
    res = query('demo', q, top_k=5)
    # build context
    context_texts = []
    for proj, hits in res.items():
        for h in hits:
            context_texts.append(f"Project: {proj}\nSource: {h.get('source')}\nText: {h.get('text')}\nScore: {h.get('score')}\n---\n")
    context = "\n".join(context_texts)

    api_key = os.environ.get('HF_ACCESS_TOKEN')
    if api_key:
        system = "You are an assistant that uses the provided context to answer the user's question concisely."
        prompt = f"Context:\n{context}\nUser question:\n{q}\nProvide a concise answer and cite sources if helpful."
        model = os.environ.get('HF_MODEL', 'google/flan-t5-large')
        try:
            answer = call_hf_chat(system, prompt, api_key, model)
            out = {"retrieved": res, "answer": answer}
        except Exception as e:
            out = {"retrieved": res, "llm_error": str(e)}
    else:
        # simple fallback summarizer
        joined = " \n ".join([t for t in context_texts])
        summary = joined.strip()[:400]
        out = {"note": "HF_ACCESS_TOKEN not set; returning fallback summary.", "retrieved": res, "summary": summary}

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
