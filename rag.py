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

# 使用 WebBaseLoader 从网页加载内容，并仅保留标题、标题头和文章内容
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# 检查加载的文档内容长度
print(len(docs[0].page_content))  # 打印第一个文档内容的长度

# 查看第一个文档（前100字符）
print(docs[0].page_content[:100])

# 使用 RecursiveCharacterTextSplitter 将文档分割成块，每块1000字符，重叠200字符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 检查分割后的块数量和内容
print(len(all_splits))  # 打印分割后的文档块数量

print(len(all_splits[0].page_content))  # 打印第一个块的字符数

print(all_splits[0].page_content)  # 打印第一个块的内容

print(all_splits[0].metadata)  # 打印第一个块的元数据

# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型，将分割的文档块嵌入并存储
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings()
)

# 查看 vectorstore 数据类型
type(vectorstore) 

# 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
type(retriever)

retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

# 检查检索到的文档内容
print(len(retrieved_docs))  # 打印检索到的文档数量

print(retrieved_docs[0].page_content)  # 打印第一个检索到的文档内容

print(retrieved_docs[1].page_content)  # 打印第一个检索到的文档内容

# 定义 RAG 链，将用户问题与检索到的文档结合并生成答案
llm = ChatOpenAI(model="gpt-4o-mini")

# 使用 hub 模块拉取 rag 提示词模板
prompt = hub.pull("rlm/rag-prompt")

# 打印模板
print(prompt.messages)

# 为 context 和 question 填充样例数据，并生成 ChatModel 可用的 Messages
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

# 查看提示词
print(example_messages[0].content)

# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 使用 LCEL 构建 RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 流式生成回答
for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)

# 流式生成回答
for chunk in rag_chain.stream("What is ToT?"):
    print(chunk, end="", flush=True)