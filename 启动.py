import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================== 1. 准备 BGE-M3 嵌入模型 (负责向量化) ==================
print("正在加载 BGE-M3 嵌入模型...")
# 指定 BGE-M3 模型，并加上中文检索专属的指令前缀，能大幅提升检索准确率
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}, # 如果你的电脑有NVIDIA显卡，可以改成 'cuda'
    encode_kwargs={'normalize_embeddings': True},
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

# ================== 2. 准备 Chroma 向量数据库 (负责检索) ==================
# 这里我们直接复用你之前跑通的 Chroma 客户端
client = chromadb.Client()
# 假设你之前跑的代码已经创建并填充了名为 "qwen_ov_knowledge_base" 的集合
# 如果没有，记得先运行上一次的 test_chroma.py 往里面塞点数据
collection = client.get_collection(name="qwen_ov_knowledge_base")

# 使用 LangChain 包装 Chroma，并指定用 BGE-M3 来处理用户的提问
vector_db = Chroma(
    client=client,
    collection_name="qwen_ov_knowledge_base",
    embedding_function=embedding_model
)
# 设置检索器，每次提问时从数据库里捞出最相关的 2 段内容
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# ================== 3. 准备 Qwen3-8B OpenVINO (负责生成回答) ==================
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

print("正在加载 Qwen3-8B OpenVINO 模型...")
model_path = "Qwen3-8B-int4-ov" # 换成你本地转换好的 OpenVINO 模型路径
ov_model = OVModelForCausalLM.from_pretrained(model_path, device='CPU')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 自定义一个 LangChain 兼容的 LLM 类，把 OpenVINO 模型包起来
from langchain_core.language_models.llms import LLM
class OVLLM(LLM):
    model: OVModelForCausalLM
    tokenizer: AutoTokenizer

    @property
    def _llm_type(self):
        return "openvino_qwen3"

    def _call(self, prompt, stop=None):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        # 截取模型新生成的文字部分
        output_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return output_text

qwen_llm = OVLLM(model=ov_model, tokenizer=tokenizer)

# ================== 4. 用 LangChain 串联成 RAG 链 ==================
# 定义一个把检索到的文档拼成字符串的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 按照 Qwen3 的专属对话格式来写提示词模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个 helpful 的助手。请严格基于以下提供的上下文来回答用户的问题。\n\n上下文信息：\n{context}"),
    ("user", "{question}")
])

# 搭建 RAG 流水线：检索 -> 格式化文档 -> 填入提示词 -> Qwen生成 -> 解析输出
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | qwen_llm
    | StrOutputParser()
)

# ================== 5. 开始测试问答！ ==================
if __name__ == "__main__":
    print("\n" + "="*40)
    print("RAG 系统已就绪！(输入 'quit' 退出)")
    print("="*40)
    while True:
        question = input("\n请输入你的问题：")
        if question.lower() == 'quit':
            break
        
        print("\n正在检索知识库并由 Qwen3-8B 生成回答...\n")
        try:
            answer = rag_chain.invoke(question)
            print("Qwen 回答：", answer)
        except Exception as e:
            print("出错了：", e)