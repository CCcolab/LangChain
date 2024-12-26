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
from langchain.chains import RetrievalQA

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

# 使用 OpenAIEmbeddings 和 Chroma 向量存储将分割文档块嵌入并存储
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # 使用更强的嵌入模型
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings
)

# 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# 提出问题进行评估
questions = [
    "What are the main types of adversarial attacks on LLMs and their impact?",
    "How are adversarial examples generated in machine learning models?",
    "What are the common defenses against adversarial attacks and how can they improve model robustness?"
]

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