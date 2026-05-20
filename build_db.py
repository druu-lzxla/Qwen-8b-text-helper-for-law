import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

client = chromadb.PersistentClient(path="./chroma_db")

vector_db = Chroma(
    client=client,
    collection_name="bge_knowledge_base",
    embedding_function=embedding_model
)

documents = [
    "Qwen3-8B 是一个强大的开源大语言模型。",
    "OpenVINO 是英特尔推出的 AI 推理框架。",
    "RAG 可以提升大模型回答准确率。"
]

vector_db.add_texts(documents)

print("数据库构建完成")