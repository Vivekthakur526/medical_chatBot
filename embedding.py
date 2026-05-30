from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
llm=ChatOpenAI(
    openai_api_base="https://router.huggingface.co/v1",
    openai_api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model="meta-llama/Llama-3.1-8B-Instruct",
)

print("right18")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  
)



load=PyPDFLoader("Medical.pdf")
file_content=load.load()


splitter=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=120)


chunks=splitter.split_documents(file_content)

print(chunks[0])


vectordb=PineconeVectorStore.from_documents(
    chunks,
     embedding,
     index_name="medicalchatbot"
        )

