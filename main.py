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

SOURCE_PATH = "./data/source"
VECTORSTORE_PATH = "./data/vectorstore"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def loaderPdf(source_path):
    """ source 폴더에 있는 PDF 파일들 모두 로드하여 Docs 반환 """
    docs = []
    for file in os.listdir(source_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(source_path, file))
            docs.extend(loader.load())
    return docs

def splitText(docs, chunk_size=1000, chunk_overlap=50):
    """ Docs를 분할하여 split_documents 반환 """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def createEmbeddings():
    """ 임베딩 생성 """
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def createVectorstore(vectorstore_path):
    """ SOURCE 폴더에 있는 PDF 파일을 임베딩하여 vectorstore 반환 """
    print("--- 벡터스토어 파일을 생성합니다 ---")
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
    """ vectorstore_path에 있는 vectorstore를 반환하고 vectorstore가 없으면 생성해서 반환 """
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
    """ Document 객체들을 텍스트로 변환 """
    result = ""
    for document in documents:
        data = {
            "page_content": document.page_content,
            "source": f"{document.metadata["source"]} - {document.metadata["page"]} / {document.metadata["total_pages"]}"
        }
        result += f"{data}\n\n"

    return result

def getPrompt():
    """ 주어진 프롬프트 반환 """
    prompt = """
    당신은 질문에 대한 응답을 하는 AI입니다.
    주어진 context를 사용하여 질문에 답변해주세요.
    만약 답을 모른다면 반드시 답을 모르겠다고 답변해야합니다.
    답변은 항상 한국어로 해야합니다.
    
    Context: {context}
    Question: {question}
    """
    return PromptTemplate.from_template(prompt)

def getLLM():
    """ LLM 반환 """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return llm


class routeQuestionResult(BaseModel):
    """LLM으로 부터 구조화된 답변을 받기 위한 output parser"""
    result: Literal["deny", "answer", "retriever"] = Field(description="사용자 질문에 대해 판단하여 어떤 노드를 호출할지 결정해주세요.")

class GraphState(TypedDict):
    context: Annotated[str, "Context"] # Retrieval 검색 결과
    question: Annotated[str, "Question"] # 사용자 질문
    answer: Annotated[str, "Answer"] # 답변
    history: Annotated[list, add_messages] # 대화 기록

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
    """ 사용자 질문이 캐시플로우 보드 게임과 관련이 없는 경우 대화를 종료하는 노드"""
    return {"answer": "캐시플로우 보드게임과 관련있는 질문만 가능합니다."}
    
def retrieveNode(state: GraphState):
    """사용자 질문을 바탕으로 관련 문서를 찾는 노드"""
    documents = retriever.invoke(state["question"])
    context = parseDocuments(documents)
    return {"context": context}

def answerNode(state: GraphState):
    """ 답변을 생성하는 노드 """
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

if __name__ == "__main__":
    vectorstore = loadVectorstore(VECTORSTORE_PATH)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    
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

    # 그래프 시각화
    # graph_image = app.get_graph().draw_mermaid_png(output_file_path="./mermaid_graph.png")
    
    initial_state = GraphState(question="캐시플로우 직업 소개해줘.", history=[], context="", answer="")
    result = app.invoke(initial_state)
    print(result)