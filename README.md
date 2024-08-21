[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yoHXt_g5)
# MediLaw QA Engine
- RAG 기반 LLM 모델 활용한 의료법 및 판례 QA 엔진
<br>

## 프로젝트 소개
**1. 배경**
  + 의료법과 관련된 정보는 **전문적인 지식이 요구**되고, 일반인에게는 **이해하기 어려운 경우**가 많음
  + 의료법에 대한 지식이 부족한 일반인들이 **법적 권리와 의무를 이해하는데 어려움**을 겪고 있음

**2. 목적**
  + 의료법과 판례에 대한 **정보를 쉽고 빠르게 찾아볼 수 있도록 지원**하기 위함
  + 신뢰할 수 있는 의료 정보를 제공하여 잘못된 정보로 인한 **피해를 예방**하기 위함
  + 전문가와의 상담을 위한 **사전 정보를 제공**하고 **효율적인 소통**을 위함

**3. Architecture**

<img src="https://github.com/user-attachments/assets/9a69738c-0771-40e3-86c2-d327c71b327f" width="500" height="166" />
<br>
<br>

## 1. 개발 환경

- 주 언어 : Python
- 버전 및 이슈관리 : 3.10
- 협업 툴 : notion, git, slack

<br>

## 2. 채택한 개발 기술과 브랜치 전략

### LangChain

- PyPDFLoader
  - **PyPDFLoader**는 **LangChain** 프레임워크에서 제공하는 PyPDFLoader 모듈을 import하는 코드입니다. 이 모듈은 PDF 파일에서 텍스트 데이터를 추출하는데 사용됩니다.
    <br>
    <br>
    **PyPDFLoader의 주요 기능**
      1. PDF 문서 로딩: PyPDFLoader는 PDF 파일을 읽고, 그 안에 포함된 텍스트를 추출하여 사용할 수 있도록 변환합니다.
      2. 텍스트 추출: PDF 문서에서 페이지별로 텍스트를 효과적으로 추출할 수 있습니다. 이를 통해 문서의 내용을 쉽게 분석하거나 처리할 수 있습니다.
      3. 문서 처리 파이프라인 통합: PyPDFLoader는 LangChain의 다른 모듈과 통합하여, 다양한 자연어 처리(NLP) 작업을 수행할 수 있는 파이프라인의 일환으로 사용될 수 있습니다.

- Chroma
  - **Chroma**는 PyPDFLoader와 마찬가지로 **LangChain** 프레임워크에서 제공하는 모듈을 import하는 코드입니다. **Chroma**는 주로 임베딩(embedding)과 검색 관련 작업을 수행하는 벡터 데이터베이스(Vector Database)입니다. LangChain 프레임워크에서 Chroma는 벡터 임베딩을 저장하고, 주어진 쿼리에 대해 가장 유사한 문서나 데이터를 검색하는데 사용됩니다.
    <br>
    <br>
    **Chroma 주요 기능**
      1. 임베딩 저장소: 텍스트 데이터를 벡터로 변환하여 저장할 수 있습니다. 이 벡터는 나중에 유사한 텍스트를 검색하는데 사용됩니다.
      2. 유사도 검색: Chroma는 벡터 간의 유사성을 계산하여 특정 쿼리와 가장 유사한 데이터 포인트를 빠르게 찾아낼 수 있습니다.
      3. 확장성: 대량의 데이터를 처리할 수 있는 기능을 갖추고 있어, 다양한 규모의 프로젝트에서 유용하게 사용될 수 있습니다.
      4. 통합 가능성: LangChain의 다른 모듈과 함께 사용하여 복잡한 자연어 처리 파이프라인을 구성할 수 있습니다.
   
- UpstageEmbeddings
  - **UpstageEmbeddings**는 Upstage에서 제공하는 라이브러리입니다. Upstage는 한국어와 같은 저자원 언어에 특화된 AI 모델을 개발하는 회사로, UpstageEmbeddings는 이 회사의 임베딩 모델을 LangChain에서 사용할 수 있도록 지원합니다.
    <br>
    <br>
    **UpstageEmbeddings 주요 기능**
      1. 임베딩 생성: 텍스트 데이터를 고차원 벡터로 변환하여 기계 학습 모델이나 검색 시스템에서 사용할 수 있도록 합니다.
      2. 유사도 계산: 생성된 임베딩을 사용하여 텍스트 간의 유사성을 평가할 수 있습니다. 이는 추천 시스템이나 검색 엔진에 유용합니다.
      3. 다양한 NLP 작업 지원: UpstageEmbeddings는 문서 분류, 클러스터링, 질문 응답 등의 다양한 자연어 처리 작업에 활용될 수 있습니다.
      4. 모델 통합: UpstageEmbeddings는 LangChain의 다른 모듈과 쉽게 통합되어, 복잡한 NLP 파이프라인을 구축하는데 도움을 줍니다.
   
- ChatUpstage
  - **ChatUpstage**는 Upstage에서 개발한 대화형 인공지능 모델을 LangChain에서 사용할 수 있도록 지원하는 모듈입니다.
    <br>
    <br>
    **ChatUpstage 주요 기능**
      1. 한국어 최적화 대화 모델: ChatUpstage는 한국어에 특화된 대화형 AI 모델을 제공합니다. 한국어로 자연스러운 대화를 생성하며, 한국어를 주요 언어로 사용하는 사용자에게 최적화된 응답을 생성합니다.
      2. LangChain과의 통합: ChatUpstage는 LangChain의 다양한 체인(chain)과 결합하여 더 복잡한 대화 흐름을 구현할 수 있습니다. 예를 들어, 정보 검색, 요약, 문서 생성 등의 작업을 대화형 AI에 통합할 수 있습니다.
      3. 다양한 대화 시나리오 지원: ChatUpstage는 고객 지원, 질의응답, 대화형 정보 제공 등 다양한 비즈니스 및 개인화된 대화 시나리오에 활용될 수 있습니다.
      4. 커스터마이징 가능: Upstage의 대화 모델은 사용자의 필요에 따라 커스터마이징이 가능하여, 특정 도메인이나 컨텍스트에 맞는 대화 모델을 구성할 수 있습니다.
      5. 한국어 자연어 처리 강점: 한국어뿐만 아니라 다중 언어 지원도 가능하지만, 특히 한국어 NLP에 최적화되어 있어 한국어 기반 애플리케이션에서 우수한 성능을 발휘합니다.
   
- HumanMessage와 SystemMessage
  - HumanMessage와 SystemMessage는 대화형 AI 모델에서 메시지를 주고받을 때 사용되는 두 가지 주요 메시지 타입입니다. 이들은 대화 흐름을 정의하고 모델의 응답에 영향을 미치는데 사용됩니다.
    <br>
    <br>
    **HumanMessage 주요 기능**
      1. 정의: HumanMessage 클래스는 사용자가 입력한 메시지를 나타냅니다. 즉, 사용자로부터의 요청이나 질문을 포함합니다.
      2. 주요 기능: 이 클래스는 사용자의 입력을 구조화된 형식으로 저장하며, AI가 이를 이해하고 적절한 응답을 생성하는데 도움을 줍니다.
      3. 사용 예: 예를 들어, "안녕하세요, 오늘 날씨는 어떤가요?"와 같은 메시지가 HumanMessage 객체로 표현될 수 있습니다.
  
    **SystemMessage 주요 기능**
      1. 정의: SystemMessage 클래스는 시스템이나 AI의 메시지를 나타냅니다. 이는 AI가 사용자에게 제공하는 정보나 지침을 포함합니다.
      2. 주요 기능: 이 클래스는 시스템의 응답을 구조화하여, AI가 대화의 맥락을 유지하고 효과적으로 사용자와 상호작용할 수 있도록 돕습니다.
      3. 사용 예: 예를 들어, "저는 인공지능입니다. 무엇을 도와드릴까요?"와 같은 메시지가 SystemMessage 객체로 표현될 수 있습니다.

- create_history_aware_retriever
  - create_history_aware_retriever는 LangChain에서 제공하는 고급 함수로, 대화의 문맥을 유지하면서 검색 작업을 수행할 수 있도록 도와줍니다. 이 함수는 대화 기록을 반영하여 더 정확하고 관련성 높은 검색 결과를 제공하는 것을 목표로 합니다. 특히, 다중턴 대화에서 이전 대화의 내용을 고려해 정보 검색을 수행할 때 유용합니다.
    <br>
    <br>
    **create_history_aware_retriever 주요 기능**
      1. 대화 기록 인식: 기본적인 검색과 달리, 이 함수는 사용자가 이전에 주고받은 대화 기록을 기반으로 검색 작업을 수행합니다. 이를 통해, 현재 질문이나 요청이 이전 대화와 어떻게 연결되는지를 파악하고 보다 맥락에 맞는 응답을 생성할 수 있습니다.
      2. 대화 기반 검색 최적화: 일반적인 정보 검색 기능에 대화의 흐름을 반영하기 때문에, 특히 챗봇이나 대화형 AI 시스템에서 과거 대화의 문맥을 유지하며 더 관련성 높은 검색 결과를 제공하는데 강점이 있습니다.
      3. LangChain과 통합: 이 함수는 LangChain의 기존 체인(chain)이나 검색 기반 워크플로우에 쉽게 통합할 수 있어, 대화형 AI 시스템에서 활용하기에 적합합니다.
      4. 맞춤형 검색: 대화 내에서 사용자가 어떤 의도로 질문을 하고 있는지 파악하여, 단순히 키워드 기반 검색이 아닌 의미 기반 검색을 수행합니다.

- ChatPromptTemplate와 MessagesPlaceholder
  - langchain_core.prompts 모듈의 ChatPromptTemplate와 MessagesPlaceholder는 LangChain에서 대화형 AI 모델을 위한 프롬프트를 유연하게 구성하고, 대화 기록을 효과적으로 관리하는데 사용됩니다.
    <br>
    <br>
    **ChatPromptTemplate 주요 기능**
      1. 대화 내 다양한 메시지를 템플릿으로 설정하여 반복적으로 사용할 수 있습니다.
      2. 메시지 순서, 역할(예: 시스템, 사용자) 등을 정의하여 복잡한 대화 흐름을 구성할 수 있습니다.
      3. 파라미터를 동적으로 채워 넣어 맞춤형 대화를 구성합니다.
   
    **MessagesPlaceholder 주요 기능**
      1. 대화의 특정 지점에 사용자 메시지나 시스템 메시지를 삽입할 수 있도록 하여, 프롬프트의 유연성을 제공합니다.
      2. 대화의 맥락에 따라 적절한 메시지를 자동으로 삽입하여 더 자연스러운 대화를 생성할 수 있습니다.
   
- create_retrieval_chain
  - create_retrieval_chain은 LangChain에서 제공하는 함수로, 정보 검색 기반의 응답 생성 워크플로우를 쉽게 구축할 수 있도록 돕습니다. 이 함수는 검색기(retriever)와 언어 모델을 결합해, 사용자의 질문에 맞는 정보를 검색하고 그 결과를 기반으로 응답을 생성하는 체인(chain)을 만듭니다.
    <br>
    <br>
    **create_retrieval_chain 주요 기능**
      1. 정보 검색: 이 함수는 특정 쿼리에 대한 관련 정보를 검색하는 체인을 생성합니다. 사용자가 입력한 질문이나 요청에 대해 적절한 데이터를 찾아줍니다.
      2. 체인 구성: 다양한 정보 소스나 데이터베이스에서 정보를 검색하고, 이를 처리하여 최종 결과를 생성하는 일련의 프로세스를 구성합니다.
      3. 유연성: 사용자가 원하는 방식에 따라 검색 체인을 커스터마이즈할 수 있어, 다양한 애플리케이션에 적합하게 설계할 수 있습니다.
      4. 통합 가능성: LangChain의 다른 모듈과 쉽게 통합되어, 복잡한 정보 검색 시스템을 구축하는 데 도움을 줍니다.

- create_stuff_documents_chain
  - create_stuff_documents_chain은 LangChain에서 제공하는 함수로, 여러 문서를 하나로 결합한 후 이를 바탕으로 응답을 생성하는 체인을 구축할 때 사용됩니다. 이 체인은 여러 문서를 단순히 연결(“stuff”)한 후, 언어 모델(LLM)을 통해 이를 요약하거나 특정 질문에 대한 답변을 생성하는 방식으로 동작합니다.
    <br>
    <br>
    **create_stuff_documents_chain 주요 기능**
      1. 문서 결합: 여러 개의 문서를 하나의 결과로 결합하여, 더 풍부한 정보를 제공합니다. 이를 통해 문서 간의 관계를 이해하고 통합된 정보를 생성할 수 있습니다.
      2. 유연한 처리: 사용자가 원하는 방식으로 문서를 결합할 수 있는 옵션을 제공하여, 다양한 요구 사항에 맞출 수 있습니다.
      3. 효율적인 정보 검색: 결합된 문서를 기반으로 하여, 관련 정보를 보다 쉽게 검색하고 활용할 수 있게 합니다.
      4. 다양한 애플리케이션: 이 함수는 데이터 분석, 보고서 작성, 정보 요약 등 다양한 애플리케이션에서 유용하게 사용될 수 있습니다.

### tempfile, uuid, base64, time, streamlit, os, dotenv

- tempfile
  - 설명: tempfile 모듈은 임시 파일과 디렉터리를 생성하고 관리하는 기능을 제공합니다. 주로 테스트나 임시 작업에 사용됩니다.
  - 주요 기능: 임시 파일을 안전하게 생성하고, 사용이 끝난 후 자동으로 삭제되도록 설정할 수 있습니다.

- uuid
  - 설명: uuid 모듈은 고유한 식별자를 생성하기 위한 유틸리티를 제공합니다. 주로 객체나 데이터의 고유성을 보장하는 데 사용됩니다.
  - 주요 기능: UUID(Universally Unique Identifier)를 생성하는 다양한 방법을 지원하며, 데이터베이스의 키나 파일 이름에 유용합니다.

- base64
  - 설명: base64 모듈은 바이너리 데이터를 ASCII 문자열 형식으로 인코딩하거나 디코딩하는 기능을 제공합니다. 주로 데이터 전송에 사용됩니다.
  - 주요 기능: 이미지나 파일을 문자열로 변환하여 네트워크를 통해 전송할 수 있도록 도와줍니다.
 
- time
  - 설명: time 모듈은 시간 관련 기능을 제공하며, 시간 측정 및 시간 지연 기능을 포함합니다.
  - 주요 기능: 현재 시간을 가져오거나, 특정 시간 동안 프로그램을 일시 중지하는 기능을 제공합니다.
 
- streamlit
  - 설명: streamlit은 데이터 애플리케이션을 쉽게 만들 수 있도록 도와주는 오픈 소스 프레임워크입니다. 주로 대화형 웹 애플리케이션을 개발하는데 사용됩니다.
  - 주요 기능: 사용자 인터페이스를 간편하게 구축하고, 데이터 시각화 및 머신러닝 모델을 배포할 수 있는 기능을 제공합니다.
 
- os
  - 설명: os 모듈은 운영 체제와 상호작용하기 위한 기능을 제공합니다. 파일 시스템 접근, 환경 변수 관리 등 다양한 기능을 포함합니다.
  - 주요 기능: 파일 및 디렉터리 작업, 환경 변수 설정 및 접근, 시스템 명령 실행 등 다양한 운영 체제 관련 작업을 수행할 수 있습니다.
 
- dotenv
  - 설명: dotenv 모듈은 .env 파일에서 환경 변수를 로드하는 기능을 제공합니다. 주로 애플리케이션의 설정 값을 관리하는데 사용됩니다.
  - 주요 기능: 환경 변수를 쉽게 관리하고, 코드에 하드코딩하지 않고도 비밀 키나 설정을 안전하게 저장할 수 있도록 도와줍니다.

### 브랜치전략 
    
- 브랜치 전략
  - Git-flow 전략을 기반으로 main, develop 브랜치와 feature 보조 브랜치를 운용했습니다.
  - main, develop, Feat 브랜치로 나누어 개발을 하였습니다.
    - **main** 브랜치는 배포 단계에서만 사용하는 브랜치입니다.
    - **develop** 브랜치는 개발 단계에서 git-flow의 master 역할을 하는 브랜치입니다.
    - **Feat** 브랜치는 기능 단위로 독립적인 개발 환경을 위하여 사용하고 merge 후 각 브랜치를 삭제해주었습니다.


<br>

## 3. 프로젝트 구조
```
├── README.md
├── .gitignore
├── .env
├── config.py
├── document_loader.py
├── embedding.py
├── chat_model.py
├── ui_components.py
└── main.py

```

<br>

## 4. 개발 기간 및 작업 관리

### 개발 기간
- 전체 개발 기간 : 2024-08-13 ~ 2024-08-18
- 기능 구현 : 2024-08-15 ~ 2024-08-18
- 그외 기간 작성
  
<br>

### 작업 관리

- chromadb 패키지 오류는 아래와 같이 발생했습니다.

```python
error: subprocess-exited-with-error

  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ?─> [30 lines of output]
      Traceback (most recent call last):
        File "D:\python_test_anaconda\new_firewall_mgmt_qa\venv_312\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 353, in <module>
          main()
        File "D:\python_test_anaconda\new_firewall_mgmt_qa\venv_312\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 335, in main
          json_out['return_val'] = hook(**hook_input['kwargs'])
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "D:\python_test_anaconda\new_firewall_mgmt_qa\venv_312\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 118, in get_requires_for_build_wheel
          return hook(config_settings)
                 ^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-build-env-h91i3ok5\overlay\Lib\site-packages\setuptools\build_meta.py", line 355, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=['wheel'])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-build-env-h91i3ok5\overlay\Lib\site-packages\setuptools\build_meta.py", line 325, in _get_build_requires
          self.run_setup()
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-build-env-h91i3ok5\overlay\Lib\site-packages\setuptools\build_meta.py", line 507, in run_setup
          super(_BuildMetaLegacyBackend, self).run_setup(setup_script=setup_script)
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-build-env-h91i3ok5\overlay\Lib\site-packages\setuptools\build_meta.py", line 341, in run_setup
          exec(code, locals())
        File "<string>", line 90, in <module>
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-install-l7_69h25\zstandard_cd5b5a3f02a54e2ca4473fd6c9cd074f\setup_zstd.py", line 66, in get_c_extension
          compiler.initialize()
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-build-env-h91i3ok5\overlay\Lib\site-packages\setuptools\_distutils\_msvccompiler.py", line 253, in initialize
          vc_env = _get_vc_env(plat_spec)
                   ^^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-build-env-h91i3ok5\overlay\Lib\site-packages\setuptools\msvc.py", line 233, in msvc14_get_vc_env
          return _msvc14_get_vc_env(plat_spec)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\7040_64bit\AppData\Local\Temp\pip-build-env-h91i3ok5\overlay\Lib\site-packages\setuptools\msvc.py", line 190, in _msvc14_get_vc_env
          raise distutils.errors.DistutilsPlatformError("Unable to find vcvarsall.bat")
      distutils.errors.DistutilsPlatformError: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
?─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
...
```

### 설명

- Python 3.12 버전인 경우 위와 같은 에러가 발생된다.


### 해결

- 이를 해결하기 위해서는 2가지 방법이 있다.
  1. Microsoft C++ Build Tools 설치
  2. Python 3.10으로 낮추기


- - -

- Lnagchain 라이브러리에서 **create_history_aware_retrieval**, **create_retrieval_chain** 모듈 사용시 Import Error 발생, 오류는 아래와 같이 발생했습니다.

```python
No Module create_history_aware_retrieval

No Module create_retrieval_chain

```

### 설명

- 버전 업데이트로 모듈 위치가 변경되어 모듈을 불러오지 못했다고 경고창이 뜨고 실행되지 않습니다.


### 해결

- 아래와 같이 import문 변경하면 해결됩니다.
 
```python
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain.chains.retrieval import create_retrieval_chain
```

<br>

## 5. 프로젝트 후기

### 운동하는 피치
협업하는 경험을 통해 한층 더 성장할 수 있는 계기가 되었습니다. 

### 연구하는 네오
문제 해결 과정에서 앞으로 오픈 마인드로 접근해야겠다고 깨달았습니다. 또한 팀 프로젝트를 통해 협업하는 과정에서 많이 배울 수 있었습니다.

### 출근하는 프로도
이번 프로젝트에서 협업/모듈 디자인/아키텍쳐 등 다각도로 공부할 수 있는 기회가 되었고, 여러모로 도움도 많이 받아 뜻깊은 시간이었습니다. 

### 퇴근하는 라이언
첫 프로젝트를 진행하면서 부족한 저를 끝까지 포기하지 않고 함께 완주해주신 팀원분들께 너무나 감사드립니다.
<br>
