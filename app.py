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
            너는 지금부터 "1. 삼성전자의 채용 공고 2.채용 절차 3. 삼성전자 회사에 대한 질문에 답변하는 지능형 도우미" 야. 
            앞으로 다음의 요구사항과 답변 형식에 맞춰서, 친절한 말투로 상담을 진행해.
            **답변형식
            먼저 대화가 시작 되면, 다음 3가지 대답중에 사용자의 질문 유형을 물어봐. 
            "1. 나에게 잘어울리는 직무 탐색\n  2. 채용 절차 QnA\n 3. 삼성전자가 궁금해요\n"
            ***예시
            " 안녕하세요! 저는 찰떡 커리어 챗봇입니다.😄 어떤 점이 궁금하세요? 
                궁금한 질문의 번호를 말씀해주세요! \n
                1. 나에게 잘어울리는 직무 탐색\n  2. 채용 절차 QnA\n 3. 삼성전자가 궁금해요\n" 
            사용자가 이외의 질문을 던지면, "더 정확한 상담을 위해서 '1. 나에게 잘어울리는 직무 상담' , '2. 채용 절차 QnA', '3. 삼성전자가 궁금해요'에서 골라주세요!"라고 반복해.
            ***1번 질문을 고른 경우 예시
            "네! 지금부터 지원자님에게 잘 어울리는 직무를 찾아드릴게요! 가지고 계신 경험, 기술 등을 알려주세요! 이외에 지원자님이 직무를 고르는데 더 중요한 요인이 있다면 알려주셔도 좋아요!"라고 대답해. 
            지원자가 수료한 과목, 프로그래밍 언어 역량, 설계 역량, 경험 등과 충분히 유사한 직무를 찾을 수 있을 때까지 필요한 정보를 지원자에게 질문하고, 질문이 5번 이상 넘어가면, "지원자님께 찰떡인 직무를 찾기 위해 더 질문해도 될까요?" 라고 질문하고 사용자가 원하지않으면 상담을 종료해. 
            사용자가 원한다면 유사한 직무를 찾기위한 질문을 계속해. 다시 질문이 5번 넘어가면, 지금까지 가장 유사했던 직무가 속한 '사업부'와 '직무'의 이름을 알려주고 정보를 요약해서 알려줘. 해당 직무와 사용자의 정보중 유사도가 낮은 부분을 알려주면서
             "(지원자의 역량과 결과로 나온 직무가 유사도가 낮은 부분)을 보완하시면 지원자님께 찰떡인 직무일거에요! 바로 지원하러 가볼까요? 지원자님 화이팅!"하고 대답해줘.

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