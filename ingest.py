# rag.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ========== 1. 加载向量数据库（与 ingest.py 保持配置一致） ==========
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},      # 检索模型通常放 CPU 即可
    encode_kwargs={'normalize_embeddings': True}
)

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="bge_knowledge_base"
)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})   # 检索 5 个相关块

# ========== 2. 加载本地 Qwen3-8B 模型 ==========
model_name = "Qwen/Qwen3-8B"   # 请根据实际路径或 HF 名称修改

# 可选：4-bit 量化（推荐，显存降至 ~6GB）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           # 自动分配到 GPU
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,   # 如果不想量化，去掉这行并取消注释下一行
    # load_in_8bit=False,
    trust_remote_code=True
)

# 若需要，设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== 3. 核心 ask 函数 ==========
def ask(question: str, k: int = 5):
    """
    检索 + 生成回答
    """
    # 3.1 检索相关文档块
    docs = retriever.invoke(question)
    
    # 3.2 构建上下文和来源列表
    context = ""
    sources = []
    for doc in docs:
        context += doc.page_content + "\n\n"
        meta = doc.metadata
        filename = meta.get("filename", "未知文件")
        page = meta.get("page", "?")
        course = meta.get("course", "")
        doc_type = meta.get("doc_type", "general")
        
        source_info = f"{filename}"
        if page != "?":
            source_info += f" 第{page}页"
        if course:
            source_info += f" [课程: {course}]"
        if doc_type != "general":
            source_info += f" [{doc_type}]"
        sources.append(source_info)
    
    # 3.3 构建 prompt（Qwen3 对话模板，使用 system + user）
    system_prompt = (
        "你是一个知识助手，基于以下参考资料回答用户的问题。"
        "如果参考资料中没有明确答案，请直接说不知道，不要编造。"
    )
    user_prompt = f"""参考资料：
{context}

问题：{question}

请基于以上参考资料给出准确、简洁的回答。"""
    
    # Qwen3 的聊天格式（使用 tokenizer.apply_chat_template 更规范）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 3.4 生成回答
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # 解码并去除输入部分
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 3.5 打印结果
    print("\n" + "="*60)
    print(f"问题：{question}")
    print("="*60)
    print(f"回答：{generated}")
    print("\n参考来源：")
    for idx, src in enumerate(sources, 1):
        print(f"{idx}. {src}")
    print("="*60)
    
    return {
        "answer": generated,
        "sources": sources,
        "retrieved_docs": docs
    }

# ========== 4. 命令行交互 ==========
if __name__ == "__main__":
    print("RAG 系统已启动，输入问题（输入 exit 退出）")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        ask(q)

def ask(question: str, k: int = 5):

    # =====================================
    # 1. 检索
    # =====================================

    docs = retriever.invoke(question)

    # =====================================
    # 2. 去重
    # =====================================

    seen = set()

    unique_docs = []

    for doc in docs:

        content = doc.page_content.strip()

        if content not in seen:

            seen.add(content)

            unique_docs.append(doc)

    # =====================================
    # 3. 构建context
    # =====================================

    context_parts = []

    sources = []

    max_context_chars = 6000

    current_length = 0

    for idx, doc in enumerate(unique_docs):

        content = doc.page_content.strip()

        meta = doc.metadata

        filename = meta.get("filename", "未知文件")

        page = meta.get("page", "?")

        course = meta.get("course", "")

        doc_type = meta.get("doc_type", "general")

        chunk_id = meta.get("chunk_id", idx)

        source_tag = f"[文档{idx+1}]"

        source_info = (
            f"{source_tag} "
            f"{filename}"
        )

        if page != "?":
            source_info += f" 第{page}页"

        if course:
            source_info += f" | {course}"

        if doc_type != "general":
            source_info += f" | {doc_type}"

        sources.append(source_info)

        chunk_text = (
            f"{source_tag}\n"
            f"{content}\n"
        )

        # context长度控制
        if current_length + len(chunk_text) > max_context_chars:
            break

        context_parts.append(chunk_text)

        current_length += len(chunk_text)

    context = "\n\n".join(context_parts)

    # =====================================
    # 4. Prompt强化
    # =====================================

    system_prompt = """
你是一个严谨的知识库助手。

你必须严格依据提供的参考资料回答。

规则：

1. 不允许编造不存在的信息
2. 若资料不足，明确说明“资料中未提供”
3. 优先引用参考资料中的定义、概念、论证
4. 回答应结构化、清晰
5. 法律问题优先：
   - 定义
   - 构成要件
   - 学说争议
   - 实务观点
6. 若引用资料，请使用 [文档x] 标记来源
"""

    user_prompt = f"""
参考资料：

{context}

问题：

{question}

请基于参考资料回答。
"""

    messages = [

        {
            "role":"system",
            "content":system_prompt
        },

        {
            "role":"user",
            "content":user_prompt
        }

    ]

    # =====================================
    # 5. Qwen Chat Template
    # =====================================

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # =====================================
    # 6. 推理
    # =====================================

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=768,

            temperature=0.3,

            top_p=0.8,

            repetition_penalty=1.1,

            do_sample=True,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # =====================================
    # 7. 输出
    # =====================================

    print("\n" + "="*80)

    print(f"问题：{question}")

    print("="*80)

    print("\n回答：\n")

    print(generated)

    print("\n" + "-"*80)

    print("参考来源：\n")

    for s in sources:

        print("-", s)

    print("="*80)

    return {

        "answer": generated,

        "sources": sources,

        "retrieved_docs": unique_docs

    }