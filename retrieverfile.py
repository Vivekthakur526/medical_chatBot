from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from dotenv import load_dotenv

from flask import Flask, request, render_template,jsonify
app=Flask(__name__)

import os

from pinecone_plugins import assistant
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  
)

llm=ChatOpenAI(
    openai_api_base="https://router.huggingface.co/v1",
    openai_api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model="meta-llama/Llama-3.1-8B-Instruct",
)

parser=StrOutputParser()





vectordb=PineconeVectorStore.from_existing_index(
       index_name="medicalchatbot",
         embedding=embedding
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", """You are a strict medical assistant.

Rules:
1. Answer ONLY from the given context.
2. Do NOT use your own knowledge.
3. If the answer is NOT clearly present in the context, reply EXACTLY:
   "I don't know based on the provided context."

Language Rule:
- Reply in the same language as the user (Hindi, English, or Hinglish).

Do not explain beyond the context.
Do not guess.
"""),
        ("user", "Context: {context}\nQuestion: {query}")
    ]
)

retriever=vectordb.as_retriever(search_kwargs={"k":3})

def stringify_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

chain= ({
    "context": retriever | stringify_docs,
    "query": RunnablePassthrough()
    }| prompt | llm|parser)
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print("DATA:", data)
    user_query = data.get("query")
    print("USER QUERY:", user_query)
    
    result = chain.invoke(user_query)
    print("result:", result)
    
    answer = str(result)
    print("answer:", answer)

   

    return jsonify({"reply": answer})


@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)




