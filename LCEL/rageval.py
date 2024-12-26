# 导入必要的库
import bs4
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever

# 使用 WebBaseLoader 从网页加载内容，并仅保留标题、标题头和文章内容
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# 检查加载的文档内容长度
#print(len(docs[0].page_content))  # 打印第一个文档内容的长度
# 查看第一个文档（前100字符）
#print(docs[0].page_content[:100])

# 使用 RecursiveCharacterTextSplitter 文档分��成块，每块1000字符，重叠200字符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 减小chunk大小，使文档分割更精细
    chunk_overlap=150,  # 增加重叠部分，提高上下文连贯性
    add_start_index=True,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "],  # 保持分隔符
    length_function=len
)
all_splits = text_splitter.split_documents(docs)

# 检查分割后的块数量和内容
#print(len(all_splits))  # 打印分割后的文档块数量
#print(len(all_splits[0].page_content))  # 打印第一个块的字符数
#print(all_splits[0].page_content)  # 打印第一个块的内容
#print(all_splits[0].metadata)  # 打印第一个块的元数据

# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型，将分割的文档块嵌入并存储
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings()
)

# 查看 vectorstore 数据类型
#type(vectorstore) 

# 创建BM25检索器
bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = 30  # 增加检索文档数量

# 创建向量检索器
vector_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 30,  # 增加检索文档数量
        "score_threshold": 0.5  # 降低相似度阈值，增加召回范围
    }
)

# 自定义混合检索函数代 EnsembleRetriever
def hybrid_retriever(query, weight_bm25=0.6):  # 增加BM25权重
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    vector_docs = vector_retriever.get_relevant_documents(query)
    
    # 改进文档相关性评分
    def score_document(doc, query):
        query_terms = set(query.lower().split())
        doc_terms = set(doc.page_content.lower().split())
        term_overlap = len(query_terms.intersection(doc_terms))
        return term_overlap / len(query_terms) if query_terms else 0
    
    # 为所有文档评分
    scored_docs = []
    seen_contents = set()
    
    # 处理BM25结果
    for doc in bm25_docs:
        if doc.page_content not in seen_contents:
            score = score_document(doc, query) * weight_bm25
            scored_docs.append((doc, score))
            seen_contents.add(doc.page_content)
    
    # 处理向量检索结果
    for doc in vector_docs:
        if doc.page_content not in seen_contents:
            score = score_document(doc, query) * (1 - weight_bm25)
            scored_docs.append((doc, score))
            seen_contents.add(doc.page_content)
    
    # 按分数排序
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:30]]  # 返回前30个最相关文档

# 3. 优化检索结果的重排序
def rerank_documents(docs, query):
    def calculate_relevance_score(doc, query):
        # 使用更复杂的相关性评分
        query_terms = query.lower().split()
        doc_content = doc.page_content.lower()
        
        # 计算词频
        term_frequency = sum(doc_content.count(term) for term in query_terms)
        
        # 计算词的覆盖率
        terms_present = sum(1 for term in query_terms if term in doc_content)
        coverage_ratio = terms_present / len(query_terms)
        
        # 考虑文档长度的惩罚项
        length_penalty = min(1.0, 1000 / len(doc_content))
        
        # 综合分数
        return (term_frequency * coverage_ratio * length_penalty)
    
    scored_docs = [(doc, calculate_relevance_score(doc, query)) for doc in docs]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs]

# 修改检索流程
def enhanced_retriever(query):
    # 获取初始检索结果
    initial_docs = hybrid_retriever(query)
    # 重排序
    reranked_docs = rerank_documents(initial_docs, query)
    return reranked_docs[:20]  # 返回重排序后的前20个文档

# 定义 RAG 链，将用户问题与检索到的文档结合并生成答案
llm = ChatOpenAI(model="gpt-4o-mini")

# 使用 hub 模块拉取 rag 提示词模板
prompt = hub.pull("rlm/rag-prompt")

# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 更新RAG链中的检索器
rag_chain = (
    {
        "context": lambda x: format_docs(enhanced_retriever(x)), 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 流式生成回答
#for chunk in rag_chain.stream("What are the main types of adversarial attacks on LLMs and their impact?"):
#   print(chunk, end="", flush=True)
#print("\n")
# 流式生成回答
#for chunk in rag_chain.stream("How are adversarial examples generated in machine learning models?"):
#    print(chunk, end="", flush=True)
#print("\n")
#for chunk in rag_chain.stream("What are the common defenses against adversarial attacks and how can they improve model robustness?"):
#    print(chunk, end="", flush=True)
#print("\n")

# 提出问题进行评估
questions = [
    "What are the main types of adversarial attacks on LLMs and their impact?",
    "How are adversarial examples generated in machine learning models?",
    "What are the common defenses against adversarial attacks and how can they improve model robustness?"
]


# 定义评估函数
def evaluate_recall_and_quality(question, retrieved_docs, model_response):
    # 改进召回率评估
    query_terms = set(question.lower().split())
    relevant_docs = []
    
    for doc in retrieved_docs:
        doc_terms = set(doc.page_content.lower().split())
        # 如果文档包含至少30%的查询词，则认为相关
        overlap = len(query_terms.intersection(doc_terms))
        if overlap / len(query_terms) >= 0.3:
            relevant_docs.append(doc)
    
    recall = len(relevant_docs) / len(retrieved_docs) if retrieved_docs else 0
    
    # 改进质量评估
    quality_score = 0
    if model_response:
        # 考虑回答的长度和相关词的出现
        response_terms = set(model_response.lower().split())
        term_overlap = len(query_terms.intersection(response_terms))
        response_length = len(model_response.split())
        
        # 综合考虑长度和相关性
        quality_score = (term_overlap / len(query_terms) * 0.6 + 
                       min(response_length / 200, 1.0) * 0.4)
    
    return recall, quality_score

# 测试召回和生成质量
for question in questions:
    try:
        # 使用增强检索器从向量数据库中检索与问题相关的文档
        retrieved_docs = enhanced_retriever(question)
        
        # 获取模型的回答
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")
        
        # 使用 rag_chain 获取完整响应用于评估
        model_response = rag_chain.invoke(question)

        # 评估召回率和生成质量
        recall, quality_score = evaluate_recall_and_quality(question, retrieved_docs, model_response)

        # 打印评估结果
        print(f"Question: {question}")
        print(f"Recall: {recall:.2f}")
        print(f"Quality Score: {quality_score:.2f}")
        print("="*50)
    
    except Exception as e:
        print(f"Error processing question '{question}': {e}")

# 评估改进后的召回率
def evaluate_enhanced_recall(question, retrieved_docs):
    # 计算相关文档数量
    relevant_docs = [
        doc for doc in retrieved_docs 
        if any(keyword in doc.page_content.lower() 
               for keyword in question.lower().split())
    ]
    recall = len(relevant_docs) / len(retrieved_docs) if retrieved_docs else 0
    return recall

# 测���改进效果
test_question = "What are the main types of adversarial attacks on LLMs?"
retrieved_docs = enhanced_retriever(test_question)
recall = evaluate_enhanced_recall(test_question, retrieved_docs)
print(f"Enhanced Recall: {recall:.2f}")
