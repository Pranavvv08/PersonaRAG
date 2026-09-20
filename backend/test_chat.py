"""
Quick manual test loop — no FastAPI, no frontend needed.

    uv run python test_chat.py

Type a question, see the retrieved chunks and the generated answer.
Try a follow-up like "tell me about the RAG one" after a list query.
Type 'quit' to exit.
"""

from dotenv import load_dotenv
load_dotenv()

from rag.generation import answer_query
from rag.retrieval import fetch_all, infer_type, is_list_query, search


def main():
    print("RAG test console. Type 'quit' to exit.\n")
    history = []

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        print("\n--- Retrieved ---")
        if is_list_query(query):
            type_ = infer_type(query)
            hits = fetch_all(type_)
            print(f"(list mode, type={type_}, {len(hits)} items)")
            for doc_id, text in hits:
                print(f"  {doc_id}: {text[:100]}")
        else:
            hits = search(query)
            if not hits:
                print("(no matches)")
            for doc_id, text, sim in hits:
                print(f"  [{sim:.4f}] {doc_id}: {text[:150]}")

        print("\n--- Answer ---")
        answer = answer_query(query, history=history)
        print(answer)
        print()

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
