import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 1. 使用专门的 embedding 模型
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 2. 创建 Chroma 客户端
client = chromadb.Client()

# 3. 创建 collection
collection = client.create_collection(
    name="knowledge_base",
    embedding_function=embedding_function
)

# 4. 添加文档
documents = [
    "Qwen3-8B 是一个强大的开源大语言模型。",
    "OpenVINO 是英特尔推出的用于加速 AI 推理的工具套件。",
    "ChromaDB 是一款轻量级、专为 AI 应用设计的向量数据库。",
    "RAG（检索增强生成）通过结合外部知识库来提升大模型的回答准确性。"
]

ids = [f"id{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    ids=ids
)

print("数据库文档数：", collection.count())

# 5. 检索
query = "怎么让大模型回答更准确？"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print("\n检索结果：")

for doc in results["documents"][0]:
    print(doc)