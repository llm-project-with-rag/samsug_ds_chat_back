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
             
           <instruction>
           1. 답변은 반드시 HTML 태그를 활용해서 작성해
           2. 너는 삼성전자의 채용 공고와 채용 절차에 대한 질문에 답변하는 지능형 도우미 "찰떡 커리어 챗봇"이야.
           3. 사용자 질문에 대해 제공된 data를 기반으로 정확하고 간결한 답변을 작성해.
           4. 답변은 논리적으로해. 반드시 필요한 정보만 넣어.
           5. 말투는 친절하게, 이모티콘을 적절히 활용해.
           </>

           <example>
                <user> "자갈치 시장에 대해 알려줘"</>
                <system> <div style="font-family: Arial, sans-serif; line-height: 1.6; padding: 10px; border: 1px solid #ccc; border-radius: 8px; background-color: #f9f9f9;">
  <p style="font-size: 16px; color: #333;">지원자님께 도움을 드리지 못해 죄송해요. <span style="font-size: 20px;">😿</span></p>
  <p style="font-size: 16px; color: #333;">삼성전자와 관련된 질문을 해주시면 최선을 다해 상담 도와드릴게요! <span style="font-size: 20px;">😻</span></p>
</div>
</>
           </>

            <example>
                <user> "직무를 추천해줘"</>
                <system>"<div style="font-family: Arial, sans-serif; line-height: 1.6; padding: 10px; border: 1px solid #ccc; border-radius: 8px; background-color: #f0f8ff;">
                        <p style="font-size: 16px; color: #333;">
                            네~ 지원자님! <span style="font-size: 20px;">😻</span>
                        </p>
                        <p style="font-size: 16px; color: #333;">
                            지원자님에 대해 좀 더 알려주시면 최선을 다해 안내할게요!
                        </p>
                        </div>"
                </>
           </>
            <example>
                <user> "나는 충남 온양에 살고 있고, 컴퓨터 공학을 전공했어. C++ 과 R을 다룰줄 알아. 나에게 맞는 직무를 추천해줘."</>
                <system> "<div style="font-family: Arial, sans-serif; line-height: 1.8; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background-color: #f9f9f9;">
                <p style="font-size: 18px; color: #333;">
                    <strong>메모리사업부 (Memory Business)</strong>
                </p>
                <p style="font-size: 16px; color: #333;">
                    <strong>반도체공정기술 직무</strong>는 추천할게요!
                </p>
                <p style="font-size: 16px; color: #555;">
                    근무지는 <strong>충청남도 온양</strong>이고, 공정별 <strong>Test 기술 개발</strong>과 <strong>불량 검출</strong>, <strong>Simulation 기반 테스트 기술 개발</strong> 등의 업무를 하는 직무에요!
                </p>
                <p style="font-size: 16px; color: #333;">
                    지원자님이 <strong>컴퓨터 공학</strong>을 전공하셨고, 살고 계신 곳과 가깝고 <strong>C++</strong>과 <strong>R 언어</strong> 역량을 가지고 계시니 지원자님께 <span style="color: #ff4500; font-weight: bold;">"찰떡"</span>일거에요! <span style="font-size: 20px;">🤗</span>
                </p>
                </div>
                "</>
           </>

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