import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
import matplotlib.pyplot as plt

load_dotenv()

# Load Upstage embeddings
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "samsungds"
txt_path = "SamsungDS_24.txt"  # 텍스트 파일 경로

# Delete existing index if it exists
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"Index '{index_name}' deleted.")

# Create a new Pinecone index
pc.create_index(
    name=index_name,
    dimension=4096,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print(f"New index '{index_name}' created.")

print("Start processing text...")

# Load the text data from the .txt file
with open(txt_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
splits = text_splitter.create_documents([text_data])

split_lengths = [len(split.page_content) for split in splits]

# Create a bar graph
plt.bar(range(len(split_lengths)), split_lengths)
plt.title("RecursiveCharacterTextSplitter")
plt.xlabel("Split Index")
plt.ylabel("Split Content Length")
plt.xticks(range(len(split_lengths)), [])
plt.show()

# Embed the splits and save to Pinecone

PineconeVectorStore.from_documents(
    splits, embedding_upstage, index_name=index_name
)

print("End processing.")
