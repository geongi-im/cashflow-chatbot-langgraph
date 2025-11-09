import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

# 상수 정의
SOURCE_PATH = "./data/source"
VECTORSTORE_PATH = "./data/vectorstore"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ============================================================
# Functions
# ============================================================
def loaderPdf(source_path):
    """source 폴더에 있는 PDF 파일들 모두 로드하여 Docs 반환"""
    docs = []
    for file in os.listdir(source_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(source_path, file))
            docs.extend(loader.load())
    return docs

def splitText(docs, chunk_size=1000, chunk_overlap=50):
    """Docs를 분할하여 split_documents 반환"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def createEmbeddings():
    """임베딩 생성"""
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def createVectorstore(vectorstore_path):
    """SOURCE 폴더에 있는 PDF 파일을 임베딩하여 vectorstore 반환"""
    docs = loaderPdf(SOURCE_PATH)
    split_documents = splitText(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = createEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )
    return vectorstore

def loadVectorstore(vectorstore_path):
    """vectorstore_path에 있는 vectorstore를 반환하고 vectorstore가 없으면 생성해서 반환"""
    if os.path.exists(vectorstore_path):
        embeddings = createEmbeddings()
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embeddings
        )
        return vectorstore
    else:
        return createVectorstore(vectorstore_path)

def parseDocuments(documents):
    """Document 객체들을 텍스트로 변환"""
    result = ""
    for document in documents:
        data = {
            "page_content": document.page_content,
            "source": f"{document.metadata['source']} - {document.metadata['page']} / {document.metadata['total_pages']}"
        }
        result += f"{data}\n\n"
    return result

def getLLM():
    """LLM 반환"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return llm

# ============================================================
# LangGraph State & Nodes
# ============================================================
class routeQuestionResult(BaseModel):
    """LLM으로 부터 구조화된 답변을 받기 위한 output parser"""
    result: Literal["deny", "answer", "retriever"] = Field(description="사용자 질문에 대해 판단하여 어떤 노드를 호출할지 결정해주세요.")

class GraphState(TypedDict):
    context: Annotated[str, "Context"]  # Retrieval 검색 결과
    question: Annotated[str, "Question"]  # 사용자 질문
    answer: Annotated[str, "Answer"]  # 답변
    history: Annotated[list, add_messages]  # 대화 기록

def routeQuestion(state: GraphState):
    """사용자 질문에 대해 처음 입력으로 들어가 판단하는 조건부 엣지 함수"""
    llm = getLLM()
    structured_llm = llm.with_structured_output(routeQuestionResult)
    prompt = PromptTemplate(
        template="""당신은 캐시플로우 보드게임에 대해 답변을 도와주는 도우미입니다.
    사용자 질문에 대해 판단하여 어떤 노드를 호출할지 결정해주세요.
    - "deny": 사용자 질문이 캐시플로우 보드게임과 관련없는 질문
    - "answer": 이전 대화내용을 참고하여 사용자 질문에 바로 답변이 가능한 질문
    - "retriever": 사용자 질문에 대해 캐시플로우 보드게임과 관련된 문서를 검색이 필요한 질문

    Question: {question}
    History: {history}

    Result:
    """, input_variables=["question", "history"])

    chain = prompt | structured_llm
    response = chain.invoke({"question": state["question"], "history": state["history"]})
    return response.result

def denyNode(state: GraphState):
    """사용자 질문이 캐시플로우 보드 게임과 관련이 없는 경우 대화를 종료하는 노드"""
    return {"answer": "캐시플로우 보드게임과 관련있는 질문만 가능합니다."}

def retrieveNode(state: GraphState):
    """사용자 질문을 바탕으로 관련 문서를 찾는 노드"""
    retriever = st.session_state.retriever
    documents = retriever.invoke(state["question"])
    context = parseDocuments(documents)
    return {"context": context}

def answerNode(state: GraphState):
    """답변을 생성하는 노드"""
    llm = getLLM()
    prompt = PromptTemplate(
        template="""당신은 캐시플로우 보드게임에 대해 답변을 하는 도우미입니다.
    사용자 질문에 대해 주어진 context를 참고하여 답변해주세요.
    만약 답을 모른다면 반드시 답을 모르겠다고 답변해야합니다.
    답변은 항상 한국어로 해야합니다.

    Context: {context}
    Question: {question}
    History: {history}

    Result:
    """, input_variables=["context", "question", "history"])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": state["context"], "question": state["question"], "history": state["history"]})
    return {"answer": response}

# ============================================================
# LangGraph 초기화
# ============================================================

@st.cache_resource
def initialize_chatbot():
    """벡터 스토어 및 LangGraph 앱 초기화 (캐싱)"""
    # 벡터 스토어 로드
    vectorstore = loadVectorstore(VECTORSTORE_PATH)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # LangGraph 워크플로우 구축
    workflow = StateGraph(GraphState)
    workflow.add_node("deny", denyNode)
    workflow.add_node("retriever", retrieveNode)
    workflow.add_node("answer", answerNode)

    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        START,
        routeQuestion,
        {
            "deny": "deny",
            "retriever": "retriever",
            "answer": "answer"
        }
    )

    # 일반 엣지 추가
    workflow.add_edge("deny", END)
    workflow.add_edge("retriever", "answer")
    workflow.add_edge("answer", END)

    # 그래프 컴파일
    app = workflow.compile()
    return app, retriever

# ============================================================
# Streamlit 
# ============================================================
st.set_page_config(
    page_title="캐시플로우 보드게임 챗봇",
    layout="centered"  # 페이지 레이아웃 (wide = 전체 너비 사용, centered = 중앙 정렬)
)
st.title("캐시플로우 보드게임 챗봇")
st.info("캐시플로우 보드게임 규칙서를 기반으로 대화를 진행하는 챗봇입니다.")

try:
    with st.spinner("챗봇을 초기화하는 중..."):
        app, retriever = initialize_chatbot()

    # 세션 상태 초기화
    if "app" not in st.session_state:
        st.session_state.app = app
    if "retriever" not in st.session_state:
        st.session_state.retriever = retriever
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

except Exception as e:
    st.error(f"챗봇 초기화 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# 저장된 대화 히스토리를 화면에 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 예시 프롬프트 버튼 클릭 처리
if "selected_prompt" in st.session_state:
    user_input = st.session_state.selected_prompt
    del st.session_state.selected_prompt
else:
    user_input = st.chat_input("질문을 입력하세요.")

if user_input:
    # 사용자 메시지를 세션 상태에 저장
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("생각 중..."):
        try:
            state = GraphState(
                question=user_input,
                history=st.session_state.messages.copy(),
                context="",
                answer=""
            )
            result = st.session_state.app.invoke(state)
            response = result.get("answer", "답변을 생성할 수 없습니다.")

            # 챗봇 응답을 세션 상태에 저장
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # 페이지 재실행하여 새 메시지 표시
    st.rerun()