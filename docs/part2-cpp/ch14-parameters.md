# Ch14: 파라미터 초기화

## 학습 목표

- 하이퍼파라미터가 무엇인지, `constexpr`로 왜 선언하는지 이해한다
- 가우시안 분포와 메르센 트위스터 난수 생성기의 역할을 파악한다
- `make_matrix` 람다로 행렬을 생성하고 `params`에 등록하는 흐름을 따라간다
- `state_dict`에 모든 가중치 행렬을 담는 방법을 직접 작성한다
- 총 파라미터 수 4,192개가 어떻게 계산되는지 확인한다

---

## 개념 설명

### 하이퍼파라미터란?

파라미터(parameter)는 학습으로 조정되는 값입니다. 반면 **하이퍼파라미터(hyperparameter)**는 학습 전에 사람이 미리 정하는 설계 값입니다. 모델의 "형태"를 결정합니다.

```
하이퍼파라미터 (사람이 결정)       파라미터 (학습으로 결정)
  N_EMBD = 16                    wte[0][0] = 0.032...
  N_LAYER = 1                    wte[0][1] = -0.071...
  N_HEAD  = 4                    wpe[0][0] = 0.041...
  ...                            ... (4192개)
```

C++에서 하이퍼파라미터는 `constexpr`로 선언합니다.

```
constexpr → "이 값은 컴파일 시점에 확정된다. 절대 바뀌지 않는다."
```

컴파일러가 이 값을 코드에 직접 새겨 넣기 때문에 런타임 오버헤드가 전혀 없습니다.

### 각 하이퍼파라미터의 의미

```
N_LAYER = 1   — 트랜스포머 레이어를 몇 겹 쌓는가
              — 1개면 얕은 네트워크. GPT-3는 96개.
              — 튜토리얼용이라 1로 설정.

N_EMBD = 16   — 각 토큰을 몇 차원의 벡터로 표현하는가
              — 16차원 = "토큰 하나의 의미"를 16개의 숫자로 표현
              — GPT-3는 12288차원.

BLOCK_SIZE = 16 — 한 번에 최대 몇 개의 토큰을 볼 수 있는가
               — 컨텍스트 창(context window)
               — 데이터셋에서 가장 긴 이름이 15자이므로 16으로 충분

N_HEAD = 4    — 어텐션 헤드를 몇 개 사용하는가
              — 각 헤드가 서로 다른 패턴을 학습
              — N_EMBD(16)을 N_HEAD(4)로 나눈 HEAD_DIM = 4

HEAD_DIM = 4  — 각 헤드가 담당하는 차원 수
              — N_EMBD / N_HEAD = 16 / 4 = 4 (파생 값)
```

### 가우시안(정규) 분포로 초기화하는 이유

파라미터를 모두 0으로 초기화하면 어떻게 될까요? 모든 뉴런이 똑같은 출력을 내고, 똑같은 기울기를 받아서, 학습이 전혀 진행되지 않습니다. 이를 **대칭 깨짐 실패(symmetry breaking failure)**라고 합니다.

랜덤 초기화가 필요한 이유입니다. 그런데 왜 **가우시안**(정규분포)인가?

```
가우시안 분포 N(0, 0.08)

확률 밀도
  ▲
  │     ████
  │   ████████
  │  ██████████
  │ ████████████
  │████████████████
  └──────────────────▶  값
    -0.24  0  +0.24
       (평균=0, std=0.08)
```

- 평균 0: 양수/음수가 균형 있게 분포해서 출력값이 폭발하거나 사라지지 않음
- 표준편차 0.08: 너무 크면 활성화가 포화(saturate), 너무 작으면 신호가 소멸

### 메르센 트위스터 — 왜 `std::mt19937`인가?

`std::mt19937`은 메르센 트위스터(Mersenne Twister) 알고리즘으로 난수를 생성합니다.

```
std::mt19937 rng(42);
              ↑
              시드(seed): 42를 넣으면 항상 같은 순서의 난수가 나옴

이유: 재현성(reproducibility)
      → 여러분이 실행해도 이 튜토리얼과 동일한 결과가 나옴
```

Python의 `random.seed(42)` + `random.gauss()`와 정확히 같은 역할입니다.

### state_dict — 이름으로 가중치에 접근하기

신경망의 가중치들을 "이름 → 행렬" 형태로 관리하는 딕셔너리입니다.

```
state_dict["wte"]       : vocab_size × N_EMBD  행렬 (토큰 임베딩)
state_dict["wpe"]       : BLOCK_SIZE × N_EMBD  행렬 (위치 임베딩)
state_dict["lm_head"]   : vocab_size × N_EMBD  행렬 (출력)
state_dict["layer0.attn_wq"] : N_EMBD × N_EMBD 행렬 (Q 프로젝션)
...
```

Python의 dict와 C++의 `std::unordered_map<std::string, Mat>`이 대응됩니다.

### 왜 파라미터는 `new Value{...}`로 힙에 할당하는가?

C++의 메모리는 두 가지 영역으로 나뉩니다.

```
스택(Stack)                    힙(Heap)
┌──────────────────┐           ┌──────────────────────────┐
│ 함수 호출마다 자동  │           │ new로 명시적으로 할당      │
│ 생성/소멸          │           │ delete로 명시적으로 해제   │
│ 빠르지만 크기 제한  │           │ 느리지만 크기 제한 없음    │
│ 함수 끝나면 사라짐  │           │ 명시적으로 해제할 때까지   │
└──────────────────┘           │ 유지됨                   │
                               └──────────────────────────┘
```

파라미터는 1000번의 학습 스텝 내내 살아 있어야 합니다. 따라서 힙에 할당합니다. 반면 순전파 중 생성되는 임시 Value들은 `graph` 아레나가 관리합니다 (Ch09에서 배운 내용).

---

## Python ↔ C++ 비교

### 하이퍼파라미터

```python
# Python (microgpt.py:75-79)
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head  # = 4
```

```cpp
// C++ (microgpt.cpp:227-231)
constexpr int N_LAYER    = 1;
constexpr int N_EMBD     = 16;
constexpr int BLOCK_SIZE = 16;
constexpr int N_HEAD     = 4;
constexpr int HEAD_DIM   = N_EMBD / N_HEAD;  // = 4
```

차이점: Python 변수는 런타임에 변경 가능하지만, `constexpr`은 컴파일 타임에 고정됩니다.

### 행렬 생성

```python
# Python (microgpt.py:80)
matrix = lambda nout, nin, std=0.08: \
    [[Value(random.gauss(0, std)) for _ in range(nin)]
     for _ in range(nout)]
```

```cpp
// C++ (microgpt.cpp:382-391)
std::normal_distribution<double> normal(0.0, 0.08);

auto make_matrix = [&](int nout, int nin) -> Mat {
    Mat m(nout, Vec(nin));
    for (int i = 0; i < nout; i++)
        for (int j = 0; j < nin; j++) {
            auto* v = new Value{normal(rng), 0.0, {}, {}};
            m[i][j] = v;
            params.push_back(v);
        }
    return m;
};
```

| Python | C++ | 의미 |
|--------|-----|------|
| `random.gauss(0, std)` | `normal(rng)` | 정규분포에서 샘플링 |
| `Value(...)` | `new Value{...}` | Value 노드 생성 |
| params에 자동으로 포함 안 됨 | `params.push_back(v)` | params에 명시적 등록 |
| 리스트 컴프리헨션 | 이중 for 루프 | 행렬 채우기 |

### state_dict 구성

```python
# Python (microgpt.py:81-89)
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values()
            for row in mat for p in row]
```

```cpp
// C++ (microgpt.cpp:394-408)
std::unordered_map<std::string, Mat> state_dict;
state_dict["wte"]     = make_matrix(vocab_size, N_EMBD);
state_dict["wpe"]     = make_matrix(BLOCK_SIZE, N_EMBD);
state_dict["lm_head"] = make_matrix(vocab_size, N_EMBD);

for (int i = 0; i < N_LAYER; i++) {
    std::string pfx = "layer" + std::to_string(i);
    state_dict[pfx + ".attn_wq"] = make_matrix(N_EMBD, N_EMBD);
    state_dict[pfx + ".attn_wk"] = make_matrix(N_EMBD, N_EMBD);
    state_dict[pfx + ".attn_wv"] = make_matrix(N_EMBD, N_EMBD);
    state_dict[pfx + ".attn_wo"] = make_matrix(N_EMBD, N_EMBD);
    state_dict[pfx + ".mlp_fc1"] = make_matrix(4 * N_EMBD, N_EMBD);
    state_dict[pfx + ".mlp_fc2"] = make_matrix(N_EMBD, 4 * N_EMBD);
}
// params는 make_matrix 내부에서 자동으로 채워짐
```

---

## 코드 작성 — 한 줄씩

이 챕터의 코드는 `main()` 함수 안에서, 데이터 로딩 이후에 위치합니다. 이전 챕터(Ch13)에서 `vocab_size`를 계산했다고 가정합니다.

### 1단계: 하이퍼파라미터 선언 (전역)

`main()` 함수 위, `gpt()` 함수 선언 앞에 작성합니다.

```cpp
// microgpt.cpp 어딘가 (전역 영역, gpt() 함수 위)

constexpr int N_LAYER    = 1;   // 트랜스포머 깊이
constexpr int N_EMBD     = 16;  // 임베딩 차원
constexpr int BLOCK_SIZE = 16;  // 최대 컨텍스트 길이
constexpr int N_HEAD     = 4;   // 어텐션 헤드 수
constexpr int HEAD_DIM   = N_EMBD / N_HEAD;  // 헤드당 차원 = 4
```

`constexpr int`는 "컴파일 시점에 결정되는 정수 상수"입니다. `#define`과 달리 타입이 있어서 컴파일러가 타입 검사를 해줍니다.

### 2단계: 난수 생성기 설정 (main 안)

```cpp
int main() {
    // ... (Ch13: 데이터 로딩, 토크나이저 코드) ...

    // 난수 생성기 초기화
    std::mt19937 rng(42);  // 시드 42 고정
```

`std::mt19937`은 `<random>` 헤더에 있습니다. 숫자 19937은 메르센 소수(2^19937 - 1)에서 왔습니다.

### 3단계: params 벡터 선언

```cpp
    std::vector<Value*> params;  // 모든 학습 파라미터의 평탄화 리스트
```

이 벡터에 모든 파라미터의 포인터를 모읍니다. Adam 옵티마이저가 이 리스트를 순회하며 업데이트합니다.

### 4단계: 정규분포 객체 생성

```cpp
    std::normal_distribution<double> normal(0.0, 0.08);
```

`std::normal_distribution<double>(평균, 표준편차)` — `<random>` 헤더 제공. `normal(rng)`을 호출할 때마다 정규분포에서 하나의 double을 샘플링합니다.

### 5단계: make_matrix 람다 작성

```cpp
    auto make_matrix = [&](int nout, int nin) -> Mat {
        Mat m(nout, Vec(nin));          // nout × nin 크기의 빈 행렬 생성
        for (int i = 0; i < nout; i++)
            for (int j = 0; j < nin; j++) {
                // 파라미터는 아레나(graph) 밖, 힙에 직접 할당!
                auto* v = new Value{normal(rng), 0.0, {}, {}};
                m[i][j] = v;            // 행렬에 배치
                params.push_back(v);    // 평탄화 리스트에 등록
            }
        return m;
    };
```

핵심 포인트:
- `[&]` — 람다가 주변의 `rng`, `normal`, `params`를 참조로 캡처
- `new Value{data, grad, children, local_grads}` — 힙에 Value 직접 생성
- `params.push_back(v)` — Adam이 나중에 이 포인터로 파라미터를 업데이트

### 6단계: state_dict 구성

```cpp
    std::unordered_map<std::string, Mat> state_dict;

    // 기본 임베딩 테이블
    state_dict["wte"]     = make_matrix(vocab_size, N_EMBD);  // 27 × 16
    state_dict["wpe"]     = make_matrix(BLOCK_SIZE, N_EMBD);  // 16 × 16
    state_dict["lm_head"] = make_matrix(vocab_size, N_EMBD);  // 27 × 16

    // 레이어별 가중치
    for (int i = 0; i < N_LAYER; i++) {
        std::string pfx = "layer" + std::to_string(i);
        state_dict[pfx + ".attn_wq"] = make_matrix(N_EMBD, N_EMBD);     // 16 × 16
        state_dict[pfx + ".attn_wk"] = make_matrix(N_EMBD, N_EMBD);     // 16 × 16
        state_dict[pfx + ".attn_wv"] = make_matrix(N_EMBD, N_EMBD);     // 16 × 16
        state_dict[pfx + ".attn_wo"] = make_matrix(N_EMBD, N_EMBD);     // 16 × 16
        state_dict[pfx + ".mlp_fc1"] = make_matrix(4 * N_EMBD, N_EMBD); // 64 × 16
        state_dict[pfx + ".mlp_fc2"] = make_matrix(N_EMBD, 4 * N_EMBD); // 16 × 64
    }

    std::cout << "num params: " << params.size() << "\n";
```

### 파라미터 수 계산 확인

```
wte:     27 × 16 =   432
wpe:     16 × 16 =   256
lm_head: 27 × 16 =   432

layer0.attn_wq: 16 × 16 = 256
layer0.attn_wk: 16 × 16 = 256
layer0.attn_wv: 16 × 16 = 256
layer0.attn_wo: 16 × 16 = 256
layer0.mlp_fc1: 64 × 16 = 1024
layer0.mlp_fc2: 16 × 64 = 1024
                        ──────
합계:                    4,192
```

Python과 동일하게 **4,192개**입니다.

---

## 컴파일 & 실행

파라미터 초기화만 테스트하는 독립 파일을 만들어 확인해 봅시다.

```cpp
// ch14_test.cpp
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Ch08~Ch10에서 만든 Value와 타입 별칭
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};

using Vec = std::vector<Value*>;
using Mat = std::vector<Vec>;

// 하이퍼파라미터
constexpr int N_LAYER    = 1;
constexpr int N_EMBD     = 16;
constexpr int BLOCK_SIZE = 16;
constexpr int N_HEAD     = 4;
constexpr int HEAD_DIM   = N_EMBD / N_HEAD;

int main() {
    int vocab_size = 27;  // 영어 소문자 26 + BOS 1

    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 0.08);
    std::vector<Value*> params;

    auto make_matrix = [&](int nout, int nin) -> Mat {
        Mat m(nout, Vec(nin));
        for (int i = 0; i < nout; i++)
            for (int j = 0; j < nin; j++) {
                auto* v = new Value{normal(rng), 0.0, {}, {}};
                m[i][j] = v;
                params.push_back(v);
            }
        return m;
    };

    std::unordered_map<std::string, Mat> state_dict;
    state_dict["wte"]     = make_matrix(vocab_size, N_EMBD);
    state_dict["wpe"]     = make_matrix(BLOCK_SIZE, N_EMBD);
    state_dict["lm_head"] = make_matrix(vocab_size, N_EMBD);

    for (int i = 0; i < N_LAYER; i++) {
        std::string pfx = "layer" + std::to_string(i);
        state_dict[pfx + ".attn_wq"] = make_matrix(N_EMBD, N_EMBD);
        state_dict[pfx + ".attn_wk"] = make_matrix(N_EMBD, N_EMBD);
        state_dict[pfx + ".attn_wv"] = make_matrix(N_EMBD, N_EMBD);
        state_dict[pfx + ".attn_wo"] = make_matrix(N_EMBD, N_EMBD);
        state_dict[pfx + ".mlp_fc1"] = make_matrix(4 * N_EMBD, N_EMBD);
        state_dict[pfx + ".mlp_fc2"] = make_matrix(N_EMBD, 4 * N_EMBD);
    }

    std::cout << "num params: " << params.size() << "\n";

    // 첫 번째 토큰 임베딩의 처음 4개 값 출력
    std::cout << "wte[0][0..3]: ";
    for (int j = 0; j < 4; j++)
        std::cout << state_dict["wte"][0][j]->data << " ";
    std::cout << "\n";

    // 정리
    for (auto* p : params) delete p;
    return 0;
}
```

```bash
g++ -std=c++17 -O2 -o ch14_test ch14_test.cpp
./ch14_test
```

**실행 결과:**
```
num params: 4192
wte[0][0..3]: 0.0693... -0.0412... 0.0821... -0.0317...
```

`num params: 4192` — Python과 정확히 일치합니다!

---

## 핵심 정리

| 개념 | Python | C++ | 의미 |
|------|--------|-----|------|
| 하이퍼파라미터 | `n_embd = 16` | `constexpr int N_EMBD = 16` | 컴파일 타임 상수 |
| 정규분포 | `random.gauss(0, 0.08)` | `std::normal_distribution<double>(0.0, 0.08)` | 가우시안 난수 |
| 난수 생성기 | `random.seed(42)` | `std::mt19937 rng(42)` | 재현 가능한 난수 |
| 행렬 생성 | `lambda nout, nin: [[...]]` | `auto make_matrix = [&](int nout, int nin)` | 행렬 생성 람다 |
| 파라미터 힙 할당 | `Value(...)` 자동 관리 | `new Value{...}` | 수동 힙 할당 |
| 파라미터 등록 | list comprehension | `params.push_back(v)` | 평탄화 리스트 |
| state_dict | `dict` | `std::unordered_map<std::string, Mat>` | 이름→행렬 매핑 |
| 총 파라미터 수 | 4,192 | 4,192 | Python과 동일 |

다음 챕터에서는 이 `state_dict`를 실제로 사용하는 `gpt()` 순전파 함수를 구현합니다.


---
[< 이전: Ch13: 데이터 로딩과 토크나이저](ch13-data-loading.md) | [목차](../README.md) | [다음: Ch15: GPT 순전파 >](ch15-gpt-forward.md)
