import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 使用 WebBaseLoader 从网页加载内容，并仅保留标题、标题头和文章内容
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# 使用 RecursiveCharacterTextSplitter 将文档分割成块，每块1000字符，重叠200字符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型，将分割的文档块嵌入并存储
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings()
)

# 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 使用 hub 模块拉取 rag 提示词模板
prompt = hub.pull("rlm/rag-prompt")

# 为 context 和 question 填充样例数据，并生成 ChatModel 可用的 Messages
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

# 定义 RAG 链，将用户问题与检索到的文档结合并生成答案
llm = ChatOpenAI(model="gpt-4o-mini")

# 使用 LCEL 构建 RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 流式生成回答
questions = [
    "What are adversarial attacks on large language models?",
    "How do adversarial attacks work in machine learning?",
    "What are some common defenses against adversarial attacks?"
]

# 测试召回并输出回答
for question in questions:
    try:
        # 使用 stream() 逐步处理生成的答案
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n" + "=" * 50 + "\n")
    except Exception as e:
        print(f"Error processing question '{question}': {e}")


# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 定义评估函数
def evaluate_recall_and_quality(question, retrieved_docs, model_response):
    # 召回率评估
    relevant_docs = [doc for doc in retrieved_docs if question.lower() in doc.page_content.lower()]
    recall = len(relevant_docs) / len(retrieved_docs) if retrieved_docs else 0

    # 生成质量评估：基于生成的文本的长度
    quality_score = 0
    if model_response:
        quality_score = len(model_response.split()) / 100  # 质量得分基于生成的字数

    return recall, quality_score

# 测试召回和生成质量
for question in questions:
    try:
        # 使用检索器从向量数据库中检索与问题相关的文档
        retrieved_docs = retriever.invoke(question)
        
        # 获取模型的回答
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = hub.pull("rlm/rag-prompt")
        example_messages = prompt.invoke(
            {"context": format_docs(retrieved_docs), "question": question}
        ).to_messages()
        
        # 从 AIMessage 中提取文本响应
        model_response = example_messages[0].content.strip() if example_messages else ""

        # 评估召回率和生成质量
        recall, quality_score = evaluate_recall_and_quality(question, retrieved_docs, model_response)

        # 打印评估结果
        print(f"Question: {question}")
        print(f"Recall: {recall:.2f}")
        print(f"Quality Score: {quality_score:.2f}")
        print("="*50)
    
    except Exception as e:
        print(f"Error processing question '{question}': {e}")

