# 캐시플로우 챗봇 (Cashflow Chatbot)

LangGraph 기반으로 구현된 캐시플로우 보드게임 안내 챗봇입니다. RAG(Retrieval-Augmented Generation) 방식을 활용하여 PDF 규칙서를 기반으로 질문에 답변합니다.

CLI 인터페이스와 Streamlit 웹 UI 두 가지 실행 방법을 제공합니다.

## 주요 기능

- **문서 기반 질의응답**: 캐시플로우 보드게임 규칙서 PDF를 임베딩하여 관련 질문에 정확한 답변 제공
- **지능형 라우팅**: 질문의 성격에 따라 자동으로 적절한 처리 경로 선택
  - `deny`: 캐시플로우와 무관한 질문 거부
  - `answer`: 대화 이력 기반 즉시 답변
  - `retriever`: 문서 검색 후 답변
- **벡터 검색**: Chroma DB를 활용한 효율적인 유사도 검색
- **대화 이력 관리**: 이전 대화 맥락을 고려한 답변 생성
- **두 가지 실행 방식**:
  - **CLI 모드**: 빠른 테스트 및 개발용 커맨드라인 인터페이스
  - **웹 UI 모드**: Streamlit 기반 사용자 친화적 채팅 인터페이스
    - 실시간 채팅 인터페이스
    - 대화 이력 자동 관리
    - 반응형 UI 및 스피너 표시
    - 세션 기반 상태 관리

## 기술 스택

- **LangGraph**: 워크플로우 그래프 구조 관리
- **LangChain**: LLM 체인 및 프롬프트 관리
- **Google Gemini**: LLM 모델 (gemini-2.0-flash)
- **Chroma DB**: 벡터 데이터베이스
- **PyMuPDF**: PDF 문서 로딩
- **Google Generative AI**: 임베딩 생성 (gemini-embedding-001)
- **Streamlit**: 웹 UI 프레임워크

## 프로젝트 구조

```
cashflow-chatbot-langgraph/
├── main.py                 # CLI 기반 메인 애플리케이션
├── app.py                  # Streamlit 웹 UI 애플리케이션
├── requirements.txt        # 패키지 의존성
├── .env                    # 환경 변수 (API 키)
├── data/
│   ├── source/            # PDF 원본 파일 저장소
│   └── vectorstore/       # 벡터 데이터베이스 저장소
└── mermaid_graph.png      # 워크플로우 시각화 그래프
```

## 설치 및 실행

### 1. 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일을 생성하고 Google API 키를 설정합니다:

```
GOOGLE_API_KEY=your_api_key_here
```

### 4. PDF 문서 준비

캐시플로우 보드게임 규칙서 PDF 파일을 `data/source/` 디렉토리에 저장합니다.

### 5. 실행

두 가지 실행 방법 중 원하는 방식을 선택하세요:

#### 방법 1: CLI 실행 (개발 및 테스트용)

```bash
python main.py
```

- 커맨드라인에서 빠르게 테스트할 수 있습니다
- 기본 예시 질문("캐시플로우 직업 소개해줘.")이 자동 실행됩니다
- 코드 수정을 통해 다양한 질문을 테스트할 수 있습니다

#### 방법 2: Streamlit 웹 UI 실행 (권장)

```bash
streamlit run app.py
```

- 웹 브라우저에서 자동으로 열립니다 (기본: http://localhost:8501)
- 채팅 인터페이스를 통해 자유롭게 질문할 수 있습니다
- 대화 이력이 자동으로 관리됩니다
- 사용자 친화적인 UI를 제공합니다

**참고**: 첫 실행 시 자동으로 벡터 데이터베이스가 생성됩니다.

## 아키텍처

### LangGraph 워크플로우

```
START → routeQuestion (조건부 라우팅)
         ├─ "deny" → denyNode → END
         ├─ "answer" → answerNode → END
         └─ "retriever" → retrieveNode → answerNode → END
```

### 주요 컴포넌트

1. **routeQuestion**: 사용자 질문을 분석하여 적절한 노드로 라우팅
2. **denyNode**: 주제와 무관한 질문 거부
3. **retrieveNode**: 벡터 DB에서 관련 문서 검색
4. **answerNode**: 컨텍스트와 대화 이력을 기반으로 답변 생성

### State 구조

```python
class GraphState(TypedDict):
    context: str      # 검색된 문서 내용
    question: str     # 사용자 질문
    answer: str       # 생성된 답변
    history: list     # 대화 이력
```

## 설정 사항

### 청크 설정

- `CHUNK_SIZE`: 300 (텍스트 분할 크기)
- `CHUNK_OVERLAP`: 50 (청크 간 중첩 크기)

### 검색 설정

- 검색 타입: `similarity` (유사도 기반)
- 검색 결과 수: `k=3` (상위 3개 문서)

### LLM 모델

- 임베딩: `gemini-embedding-001`
- 생성: `gemini-2.0-flash`

## 사용 예시

### CLI 모드 (main.py)

[main.py](main.py) 파일의 코드를 수정하여 질문을 변경할 수 있습니다:

```python
initial_state = GraphState(
    question="캐시플로우 직업 소개해줘.",
    history=[],
    context="",
    answer=""
)
result = app.invoke(initial_state)
print(result)
```

### Streamlit 웹 UI 모드 (app.py)

1. `streamlit run app.py` 명령어로 실행
2. 웹 브라우저에서 자동으로 열림
3. 채팅 입력창에 질문 입력 (예: "캐시플로우 게임 시작 방법 알려줘")
4. 실시간으로 답변 확인
5. 대화 이력이 화면에 표시되며 자동으로 관리됨

## 워크플로우 시각화

코드 내 주석을 해제하면 워크플로우 그래프를 이미지로 출력할 수 있습니다:

```python
graph_image = app.get_graph().draw_mermaid_png(output_file_path="./mermaid_graph.png")
```

## 라이선스

이 프로젝트는 개인 학습 및 연구 목적으로 제작되었습니다.

## 참고사항

- 첫 실행 시 벡터 데이터베이스 생성에 시간이 소요될 수 있습니다
- 답변의 품질은 제공된 PDF 문서의 내용에 따라 달라집니다
- Google API 키가 필요하며, API 사용량에 따라 요금이 부과될 수 있습니다
