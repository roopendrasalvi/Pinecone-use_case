import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings # pyright: ignore[reportMissingImports]
 
#load keys
load_dotenv()
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name="career-guidance-bot"
 
# Create index if not exists
if index_name not in[i['name']for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
 
    )
    print("created Pinecone index")
index=pc.Index(index_name)
 
#load json data
with open("aarti_proj/data/career_docs.json","r") as f:
    docs=json.load(f)
 
#Embeddings
embeddings=AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    model="text-embedding-3-small",
    openai_api_type="azure",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2023-05-15",
    chunk_size=2048
    )

#upload to pinecone
 
vectors=[]
for i,doc in enumerate(docs):
    vector=embeddings.embed_query(doc["text"])
    vectors.append({
        "id":str(i),
        "values":vector,
        "metadata":{"role":doc["role"],"text":doc["text"]}
    })
index.upsert(vectors=vectors)
print("career data uploaded to pinecone")