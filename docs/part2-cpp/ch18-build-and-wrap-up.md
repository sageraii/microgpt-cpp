# Ch18: 빌드와 마무리

## 학습 목표

- `CMakeLists.txt`의 각 줄이 무엇을 하는지 설명할 수 있다
- 직접 `g++` 명령으로 빌드하는 방법과 CMake를 사용하는 방법 두 가지를 모두 실행한다
- Python과 C++ 구현의 성능 차이와 공통점을 이해한다
- `microgpt.cpp` 515줄의 전체 구조를 한눈에 파악한다
- 다음 학습 방향을 스스로 선택할 수 있다

---

## CMakeLists.txt 완전 해설

### 전체 파일

```cmake
cmake_minimum_required(VERSION 3.16)
project(microgpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

add_executable(microgpt microgpt.cpp)

target_compile_options(microgpt PRIVATE
    -Wall -Wextra -Wpedantic
)

target_link_libraries(microgpt PRIVATE m)
```

### 한 줄씩 설명

**`cmake_minimum_required(VERSION 3.16)`**

이 CMakeLists.txt를 사용하려면 CMake 3.16 이상이 필요하다고 선언합니다. 더 오래된 CMake는 일부 기능이 없어서 오류가 발생할 수 있습니다.

```bash
cmake --version  # 설치된 CMake 버전 확인
```

**`project(microgpt LANGUAGES CXX)`**

프로젝트 이름을 `microgpt`로 정하고, C++ 언어만 사용한다고 선언합니다. `LANGUAGES CXX`를 명시하면 CMake가 C 컴파일러 탐색 등 불필요한 작업을 건너뜁니다.

**`set(CMAKE_CXX_STANDARD 17)`**

C++17 표준을 사용합니다. `std::string_view`, `if constexpr`, structured bindings 등 C++17 기능이 활성화됩니다. 이 프로젝트에서는 `std::unordered_map`, `std::optional` 없이도 되지만, `std::filesystem`을 나중에 쓰거나 `[[nodiscard]]` 등을 위해 17을 지정합니다.

**`set(CMAKE_CXX_STANDARD_REQUIRED ON)`**

C++17 지원이 없는 컴파일러로 빌드하려 할 때 오류를 발생시킵니다. 조용히 낮은 버전으로 fallback되는 것을 방지합니다.

**`set(CMAKE_CXX_EXTENSIONS OFF)`**

GCC/Clang의 비표준 확장을 비활성화합니다(`-std=c++17` 대신 `-std=gnu++17`이 되는 것을 막음). 이식성 높은 코드를 위해 설정합니다.

**`if(NOT CMAKE_BUILD_TYPE) ... endif()`**

빌드 타입이 지정되지 않으면 기본값을 `Release`로 설정합니다.

```bash
# Debug 빌드: cmake -DCMAKE_BUILD_TYPE=Debug ..
# Release 빌드: cmake .. (기본값)
```

**`set(CMAKE_CXX_FLAGS_RELEASE "-O2")`**

Release 빌드 시 `-O2` 최적화 플래그를 사용합니다.

```
-O0: 최적화 없음 (디버그하기 쉬움, 느림)
-O1: 기본 최적화
-O2: 적극적 최적화 (속도 ↑, 컴파일 시간 ↑)
-O3: 최대 최적화 (배열 벡터화 등 공격적 최적화)
```

스칼라 autograd 코드라 `-O3`를 써도 극적인 차이는 없습니다.

**`set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")`**

Debug 빌드 시 `-g`(디버그 심볼 포함)와 `-O0`(최적화 없음)을 사용합니다. GDB나 LLDB로 디버깅할 때 필요합니다.

**`add_executable(microgpt microgpt.cpp)`**

`microgpt.cpp` 하나의 소스 파일로 `microgpt` 실행 파일을 만듭니다. 소스가 여러 개라면:

```cmake
add_executable(microgpt
    microgpt.cpp
    value.cpp
    gpt.cpp
)
```

처럼 나열합니다.

**`target_compile_options(microgpt PRIVATE -Wall -Wextra -Wpedantic)`**

경고 플래그 설정입니다.

```
-Wall      : 일반적인 모든 경고 활성화
-Wextra    : 추가 경고 (미사용 함수 인자 등)
-Wpedantic : 표준을 엄격하게 준수하도록 요구 (GNU 확장 사용 시 경고)

PRIVATE    : 이 플래그가 microgpt 타겟에만 적용됨
             (라이브러리라면 PUBLIC이나 INTERFACE를 쓸 수 있음)
```

교육용 코드에서 경고를 엄격하게 설정하는 이유: 컴파일러가 잠재적 버그를 미리 알려줍니다.

**`target_link_libraries(microgpt PRIVATE m)`**

수학 라이브러리(`libm`)를 링크합니다. Linux에서 `std::sqrt`, `std::log`, `std::exp` 등을 사용할 때 필요할 수 있습니다. macOS와 Windows에서는 보통 자동으로 포함됩니다.

---

## 빌드 방법 2가지

### 방법 1: g++ 직접 사용 (빠르고 단순)

단일 소스 파일이라 g++로 바로 컴파일하는 것이 가장 빠릅니다.

```bash
# 기본 빌드
g++ -std=c++17 -O2 -o microgpt microgpt.cpp

# 엄격한 경고와 함께 빌드
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -o microgpt microgpt.cpp

# Linux에서 수학 라이브러리 명시 링크 필요시
g++ -std=c++17 -O2 -o microgpt microgpt.cpp -lm

# 실행
./microgpt
```

각 플래그:
```
-std=c++17  : C++17 표준 사용
-O2         : 최적화 레벨 2
-o microgpt : 출력 파일 이름
microgpt.cpp: 소스 파일
```

### 방법 2: CMake 사용 (권장, 프로젝트가 커지면 필수)

```bash
# 1) build 디렉토리 생성 (소스 디렉토리를 깔끔하게 유지)
mkdir build
cd build

# 2) CMake 구성 (Makefile 등 빌드 파일 생성)
cmake ..

# 3) 빌드 (실제 컴파일)
cmake --build .

# 4) 실행 (build 디렉토리에서)
./microgpt
```

또는 한 번에:

```bash
mkdir build && cd build && cmake .. && cmake --build . && ./microgpt
```

CMake를 쓰면 좋은 점:
- 빌드 타입 쉽게 전환: `cmake -DCMAKE_BUILD_TYPE=Debug ..`
- IDE(CLion, VS Code) 통합
- 의존성 관리 자동화 (프로젝트가 커질 때)
- 크로스 플랫폼 빌드 (Linux/macOS/Windows)

---

## 성능 비교: Python vs C++

### 실행 시간 측정

```bash
# Python 실행 시간
time python microgpt.py

# C++ 실행 시간
time ./microgpt
```

일반적인 실행 환경에서의 대략적인 비교:

```
Python (microgpt.py):  ~300~600초 (5~10분)
C++    (microgpt.cpp): ~60~120초  (1~2분)

속도 차이: 약 3~5배
```

### 예상보다 차이가 크지 않은 이유

C++이 Python보다 **항상** 100배 빠르다고 알려져 있습니다. 그런데 여기서는 왜 5배 정도밖에 차이가 안 날까요?

```
병목 지점 분석:

Python의 느린 이유:
  - 인터프리터 오버헤드 (타입 검사, GIL 등)
  - 객체 생성/소멸 비용

C++의 한계:
  - 스칼라 단위 계산 (Value 노드 하나씩)
  - 포인터 추적으로 캐시 미스 발생
  - 계산 자체보다 메모리 할당/해제 비용이 큼
```

실제 딥러닝 프레임워크(PyTorch, TensorFlow)가 빠른 이유는 C++이라서가 아니라 **텐서(행렬) 단위 연산**을 하기 때문입니다. GPU에서 수천 개의 숫자를 동시에 처리합니다.

```
microgpt (Python/C++): 숫자 하나씩 처리 (스칼라)
PyTorch:               행렬 전체를 한 번에 처리 (텐서)
GPU + PyTorch:         행렬 전체를 병렬로 처리 (텐서 + 병렬)
```

### 알고리즘은 동일하다

Python과 C++ 구현은 완전히 동일한 알고리즘입니다. 같은 시드, 같은 파라미터 수, 같은 수학 연산. 언어만 다릅니다.

```
Python microgpt.py ←[동일한 알고리즘]→ C++ microgpt.cpp
  32033개 이름 학습                       32033개 이름 학습
  4192개 파라미터                         4192개 파라미터
  loss 3.4 → 2.0                         loss 3.4 → 2.0
  이름 생성                               이름 생성
```

이 동일성이 이 튜토리얼의 핵심입니다: **같은 개념이 두 언어로 어떻게 표현되는지**.

---

## microgpt.cpp 515줄 한눈에 보기

```
microgpt.cpp (515줄)
│
├── [1~12줄]    파일 헤더: 설명, 컴파일 명령
│
├── [14~27줄]   #include: 필요한 표준 라이브러리 헤더 14개
│
├── [29~148줄]  자동 미분 엔진 (Autograd)
│   ├── [41~46줄]   Value 구조체: data, grad, children, local_grads
│   ├── [51~69줄]   Graph 아레나: make(), clear()
│   ├── [71줄]      전역 graph 인스턴스
│   ├── [78~116줄]  미분 가능한 연산: add, mul, scale, power, log, exp, relu
│   ├── [114~120줄] 복합 연산: neg, sub, div, sub_const
│   └── [123~148줄] backward(): 위상정렬 + 역방향 체인룰
│
├── [150~155줄] 타입 별칭: Vec, Mat
│
├── [157~221줄] 신경망 빌딩 블록
│   ├── [163~173줄] linear(): 행렬-벡터 곱
│   ├── [178~200줄] softmax(): 수치 안정 softmax
│   └── [205~221줄] rmsnorm(): RMS 정규화
│
├── [223~231줄] 하이퍼파라미터: constexpr 5개
│
├── [238~317줄] gpt() 함수: 순전파
│   ├── [244~247줄] 임베딩 합산 + RMSNorm
│   ├── [249~313줄] 레이어 루프 (N_LAYER번)
│   │   ├── [253~298줄] 멀티헤드 어텐션 (Q,K,V 프로젝션, KV캐시, 어텐션, 출력)
│   │   └── [305~312줄] MLP 블록 (fc1, relu, fc2, 잔차)
│   └── [316줄]     최종 lm_head 프로젝션
│
└── [323~515줄] main() 함수
    ├── [325~340줄] 데이터셋 다운로드 (curl)
    ├── [342~357줄] 파일 읽기 및 셔플
    ├── [359~374줄] 토크나이저 구성
    ├── [376~408줄] 파라미터 초기화 (make_matrix, state_dict)
    ├── [410~416줄] Adam 버퍼 초기화
    ├── [418~473줄] 학습 루프 (1000 스텝)
    │   ├── 문서 선택 및 토큰화
    │   ├── 순전파 + 손실 계산
    │   ├── backward()
    │   ├── Adam 업데이트
    │   └── graph.clear()
    ├── [475~507줄] 추론 (20개 이름 생성)
    │   ├── temperature 적용
    │   ├── discrete_distribution 샘플링
    │   └── BOS로 종료
    └── [509~514줄] 메모리 정리 + return 0
```

---

## 전체 실행 흐름 정리

프로그램을 `./microgpt`로 실행하면 다음 순서로 진행됩니다.

```
1. input.txt 확인 → 없으면 curl로 다운로드
   ↓
2. 32,033개 이름 읽기 → shuffle
   출력: "num docs: 32033"
   ↓
3. 고유 문자 26개 추출 → vocab_size=27
   출력: "vocab size: 27"
   ↓
4. state_dict 초기화 (4192개 파라미터 랜덤 생성)
   출력: "num params: 4192"
   ↓
5. 학습 루프 (1000 스텝)
   출력: "step 1000 / 1000 | loss 2.0xxx" (실시간 갱신)
   소요 시간: 약 1~2분
   ↓
6. 추론 (20개 이름 생성)
   출력: "--- inference (new, hallucinated names) ---"
          "sample  1: karis"
          ...
   ↓
7. 메모리 해제, 종료
```

---

## 더 배우려면

이 튜토리얼을 완주했다면 GPT의 핵심을 이미 손으로 구현한 것입니다. 다음 단계를 제안합니다.

### Karpathy의 작품들 (강력 추천)

- **micrograd** — 이 튜토리얼의 `Value` 클래스의 Python 원본. 200줄.
  `https://github.com/karpathy/micrograd`

- **makemore** — 이 데이터셋(names.txt)을 사용하는 더 발전된 언어 모델 시리즈.
  `https://github.com/karpathy/makemore`

- **nanoGPT** — GPT-2를 재현하는 최소한의 코드. PyTorch 기반.
  `https://github.com/karpathy/nanoGPT`

- **YouTube 강의** — "Neural Networks: Zero to Hero" (무료)
  개념과 코드를 함께 설명하는 최고의 딥러닝 강의.

### 논문

- **"Attention Is All You Need"** (Vaswani et al., 2017) — Transformer의 원조 논문.
  이 튜토리얼에서 구현한 구조의 수학적 기반.

- **GPT-2 paper** (Radford et al., 2019) — 이 튜토리얼이 따른 아키텍처.

### 실습 아이디어 (난이도 순)

**초급:**
- temperature를 바꿔 가며 생성 결과 관찰 (0.1, 0.5, 1.0, 1.5)
- N_LAYER=2로 바꾸면 파라미터 수가 어떻게 변하는지 계산
- 다른 텍스트 데이터(짧은 시, 단어 목록)로 학습해 보기

**중급:**
- PyTorch로 같은 모델 구현 — `nn.Linear`, `nn.Embedding` 사용
- N_EMBD=64, N_LAYER=2로 키워서 학습 품질 비교
- GeLU 활성화 함수를 추가해 ReLU와 교체

**고급:**
- 텐서 라이브러리 직접 만들기 — Value 대신 `Tensor` 클래스 (행렬 단위 autograd)
- CUDA로 GPU 병렬화 적용
- 자체 BPE 토크나이저 구현 (문자 단위 → 서브워드 단위)

---

## 핵심 정리: 이 튜토리얼에서 배운 것

| 챕터 | 핵심 개념 | C++ 기술 |
|------|----------|---------|
| Ch06 | g++, Hello World | 컴파일러, 빌드 |
| Ch07 | Python→C++ 차이 | 타입, 메모리 모델 |
| Ch08 | Value 구조체 | struct, 필드 |
| Ch09 | Graph 아레나 | new/delete, RAII |
| Ch10 | 미분 가능한 연산 | inline 함수, 람다 |
| Ch11 | 역전파 | 위상정렬, std::function |
| Ch12 | linear, softmax, rmsnorm | Vec, Mat, 알고리즘 |
| Ch13 | 데이터 로딩 | fstream, set, unordered_map |
| Ch14 | 파라미터 초기화 | constexpr, mt19937, normal_distribution |
| Ch15 | GPT 순전파 | KV 캐시, 멀티헤드 어텐션 |
| Ch16 | 학습 루프 | Adam, printf, fflush |
| Ch17 | 추론 | temperature, discrete_distribution |
| Ch18 | 빌드, 마무리 | CMake, 전체 구조 |

---

## 마치며

이 튜토리얼의 시작은 단 하나의 질문이었습니다.

> "ChatGPT는 어떻게 동작하는가?"

그리고 지금 여러분은 그 답을 압니다. 단순히 개념으로 이해한 것이 아니라, C++ 코드 515줄로 직접 구현하면서 체득했습니다.

스칼라 하나에서 시작해서 Value 구조체를 만들고, 계산 그래프를 구축하고, 역전파로 기울기를 계산하고, Adam으로 4,192개의 파라미터를 조정해서, 마침내 모델이 스스로 이름을 지어냅니다.

```
"ChatGPT가 수십억 개의 파라미터로 하는 일"
      =
"microgpt가 4,192개의 파라미터로 하는 일"
      ×
      스케일
```

축하합니다. 당신은 GPT를 처음부터 C++로 구현했습니다.


---
[< 이전: Ch17: 추론](ch17-inference.md) | [목차](../README.md)
