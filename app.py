import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

load_dotenv()

# upstage models
chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "samsungds"

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 3}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # 사용자 정의 프롬프트 템플릿 정의
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            """ 
            당신은 삼성전자의 채용 공고와 채용 절차에 대한 질문에 답변하는 지능형 도우미입니다. 
            사용자 질문에 대해 제공된 문서를 기반으로 정확하고 간결한 답변을 작성하세요. 
            관련 문서가 검색되었다면, 문서의 세부 정보를 답변에 포함하여 신뢰성을 높이세요. 
            만약 질문이 채용 공고와 관련이 없거나, 제공된 정보로 답변하기 어려운 경우 정중히 안내하세요.
            그리고 마지막으로 말 끝마다 "감사합니다!"를 붙여주세요.
            """
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    # RetrievalQA를 맞춤형 프롬프트로 초기화
    qa = RetrievalQA.from_chain_type(
        llm=chat_upstage,
        chain_type="stuff",
        retriever=pinecone_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}  # 사용자 정의 프롬프트 전달
    )

    print(f"User message: {req.message}")
    result = qa(req.message)

    # Retrieve source documents
    source_docs = result['source_documents']

    # Extract content from the source documents
    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in source_docs]

    # Log or inspect the retrieved documents
    for i, source in enumerate(sources):
        print(f"Source {i + 1}: {source['content']}")
        print(f"Metadata: {source['metadata']}")

    return {
        "reply": result['result'],
        "sources": sources  # Include source documents in the response
    }

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)