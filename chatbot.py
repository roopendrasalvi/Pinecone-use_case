import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings # pyright: ignore[reportMissingImports]
from langchain.vectorstores import Pinecone
# from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
 
#load keys
load_dotenv()
 
#connect Pinecone
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index("career-guidance-bot")
 
#vector store +embeddings

with open("aarti_proj/data/career_docs.json","r") as f:
    text=json.load(f)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

embeddings=AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    model="text-embedding-3-small",
    openai_api_type="azure",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2023-05-15",
    chunk_size=2048
    )

doc = text_splitter.create_documents([text[0]['text']])

vectorstore = PineconeVectorStore.from_documents(
    documents= doc,
    index_name="career-guidance-bot",
    embedding=embeddings
)
retriever=vectorstore.as_retriever(search_kwargs={"k":2})
 
#Azure OpenAI LLM
 
llm=AzureChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    openai_api_type="azure"
)
 
# Retrieval QA chain
 
qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
 
def get_answer(query:str)->str:
    """ ASK the AI Career Chatbot"""
    response=qa_chain.run(query)
    return response
 
#Optional console testing
 
if __name__=="__main__":
    print("AI career chatboat(type 'exit' to stop)")
    while True:
        user_input=input("you:")
        if user_input.lower() in ["exit","quit"]:
            print("bot:Goodbye!")
            break
        answer=get_answer(user_input)
        print("bot",answer)