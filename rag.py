import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


# ========== 1) Embedding + VectorDB ==========
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="bge_knowledge_base",
)


# ========== 2) Optional reranker ==========
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_reranker = None


def _load_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from FlagEmbedding import FlagReranker

        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
    except Exception:
        _reranker = False
    return _reranker


def rerank_docs(question: str, docs, top_n: int = 5):
    reranker = _load_reranker()
    if reranker is False:
        return docs[:top_n]

    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked[:top_n]]


# ========== 3) Qwen3-8B ==========
model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ========== 4) Ask with metadata filtering + reranker ==========
def ask(
    question: str,
    k: int = 12,
    top_n: int = 5,
    doc_type: Optional[str] = None,
    filename: Optional[str] = None,
):
    where = {}
    if doc_type:
        where["doc_type"] = doc_type
    if filename:
        where["filename"] = filename

    if where:
        docs = vector_db.similarity_search(question, k=k, filter=where)
    else:
        docs = vector_db.similarity_search(question, k=k)

    docs = rerank_docs(question, docs, top_n=top_n)

    seen_ids = set()
    unique_docs = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        unique_key = chunk_id or doc.page_content.strip()
        if unique_key in seen_ids:
            continue
        seen_ids.add(unique_key)
        unique_docs.append(doc)

    context = "\n\n".join(doc.page_content for doc in unique_docs)
    sources = []
    for doc in unique_docs:
        meta = doc.metadata
        sources.append(
            f"{meta.get('filename', '未知文件')} | {meta.get('doc_type', 'general')} | chunk_id={meta.get('chunk_id', 'N/A')}"
        )

    system_prompt = (
        "你是严谨的法律文书助手。"
        "必须严格依据给定参考资料作答；若证据不足，明确说“资料中未提供”。"
    )
    user_prompt = f"参考资料：\n{context}\n\n问题：{question}\n\n请给出结构化、简明回答。"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return {"answer": generated, "sources": sources, "retrieved_docs": unique_docs}


if __name__ == "__main__":
    print("RAG 系统已启动，输入问题（输入 exit 退出）")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        result = ask(q)
        print("\n回答：", result["answer"])
        print("参考来源：")
        for s in result["sources"]:
            print("-", s)
