from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
model = OpenAI()

template = """Answer the quedstion based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# create a vectorstore
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"],
    OpenAIEmbeddings(),
)

# # querying the vectorstore
query = "Where did harrison work?"
# docs = vectorstore.similarity_search(query, fetch_k=1, k=1)
# print(docs[0].page_content)

# querying as retriever
retriever = vectorstore.as_retriever()
# docs = retriever.invoke(query, fetch_k=1, k=1)
# print(docs[0].page_content)


retrieval_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model.with_config(run_name="retrieval_chain")
    | StrOutputParser()
)

response = retrieval_chain.invoke(query)
print(response)