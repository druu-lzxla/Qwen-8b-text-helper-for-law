import chromadb

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

# ==========================================
# 1. 加载Embedding模型
# ==========================================

print("加载 BGE-M3...")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

# ==========================================
# 2. 连接Chroma数据库
# ==========================================

print("连接 ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

vector_db = Chroma(
    client=client,
    collection_name="bge_knowledge_base",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k":2}
)

# ==========================================
# 3. 加载Qwen3 OpenVINO
# ==========================================

print("加载 Qwen3 OpenVINO...")

model_path = "./Qwen3-8B-int4-ov"

model = OVModelForCausalLM.from_pretrained(
    model_path,
    device='CPU'
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path
)

# ==========================================
# 4. RAG函数
# ==========================================

def ask(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    messages = [

        {
            "role":"system",
            "content":
            "你是一个RAG助手。"
            "只能基于提供的上下文回答。"
            "若上下文没有答案，明确说明不知道。"
        },

        {
            "role":"user",
            "content":
            f"""
上下文：

{context}

问题：

{question}
"""
        }

    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    answer = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    return answer

# ==========================================
# 5. CLI循环
# ==========================================

print("\nRAG系统启动成功。")
print("输入 quit 退出。\n")

while True:

    question = input("问题：")

    if question.lower() == "quit":
        break

    try:

        answer = ask(question)

        print("\n回答：")
        print(answer)
        print()

    except Exception as e:

        print("\n错误：",e)