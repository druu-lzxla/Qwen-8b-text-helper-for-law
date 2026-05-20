import hashlib
from typing import Iterable

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

client = chromadb.PersistentClient(path="./chroma_db")

vector_db = Chroma(
    client=client,
    collection_name="bge_knowledge_base",
    embedding_function=embedding_model,
)


def stable_chunk_id(chunk: Document) -> str:
    """为每个 chunk 生成稳定 ID：相同内容 -> 相同 ID。"""
    content = chunk.page_content.strip()
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def dedupe_chunks(chunks: Iterable[Document]) -> list[Document]:
    """按稳定 ID 去重，避免重复写入。"""
    unique: dict[str, Document] = {}
    for chunk in chunks:
        chunk_id = stable_chunk_id(chunk)
        chunk.metadata["chunk_id"] = chunk_id
        unique.setdefault(chunk_id, chunk)
    return list(unique.values())


if __name__ == "__main__":
    # 示例：真实项目中请替换为你的法律文书原文列表
    raw_documents = [
        Document(
            page_content="合同成立的要件包括当事人、意思表示一致与合法标的。",
            metadata={"filename": "民法总则摘要", "doc_type": "civil_law", "source": "demo"},
        ),
        Document(
            page_content="合同无效的常见情形包括违反法律强制性规定。",
            metadata={"filename": "合同编摘要", "doc_type": "civil_law", "source": "demo"},
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(raw_documents)
    chunks = dedupe_chunks(chunks)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    vector_db.add_documents(chunks, ids=ids)

    print(f"数据库构建完成，写入 chunk 数量：{len(chunks)}")
