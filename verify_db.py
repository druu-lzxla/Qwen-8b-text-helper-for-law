from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# embedding必须一致
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

# 连接数据库
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="bge_knowledge_base"
)

# 查看总文档数
collection = vector_db._collection

count = collection.count()

print(f"\n当前向量数量：{count}")

# 抽查前几个metadata
results = collection.peek(limit=3)

print("\n示例数据：")

for i in range(len(results["documents"])):

    print("\n===================")

    print("Document:")
    print(results["documents"][i][:200])

    print("\nMetadata:")
    print(results["metadatas"][i])