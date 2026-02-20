# Ch05: Transformer와 학습

## 학습 목표

이 챕터를 마치면 다음을 이해할 수 있습니다.

- `microgpt.py` 전체 구조를 한눈에 파악할 수 있습니다
- 하이퍼파라미터가 무엇이고 왜 그 값을 사용하는지 알 수 있습니다
- 어텐션 메커니즘이 어떻게 "관련 정보"를 찾는지 이해합니다
- 학습 루프가 어떻게 동작하는지, Adam 옵티마이저가 무엇인지 이해합니다
- 추론(inference)으로 이름을 생성하는 과정을 이해합니다

---

## 1. microgpt.py 전체 구조

먼저 파일 전체를 한눈에 봅시다.

```
microgpt.py
│
├── [1] 데이터 준비 (line 14-21)
│     └── input.txt 다운로드 → docs 리스트 로드
│
├── [2] 토크나이저 (line 23-27)
│     └── 문자 목록 → 정수 ID 변환 테이블
│
├── [3] Value 클래스 (line 30-72)
│     └── 자동 미분 엔진 (Ch04에서 배움)
│
├── [4] 파라미터 초기화 (line 74-90)
│     └── 하이퍼파라미터 → 가중치 행렬들
│
├── [5] 모델 정의 (line 92-144)
│     ├── linear, softmax, rmsnorm
│     └── gpt(): Transformer 순전파
│
├── [6] 학습 루프 (line 147-184)
│     └── Adam 옵티마이저로 1000 스텝 학습
│
└── [7] 추론 (line 186-200)
      └── 학습된 모델로 이름 생성
```

이제 각 부분을 상세히 살펴봅니다.

---

## 2. 하이퍼파라미터 — 사람이 정하는 설정값

```python
# microgpt.py:75-79
n_layer = 1     # Transformer 레이어 수 (깊이)
n_embd = 16     # 임베딩 차원 (너비)
block_size = 16 # 최대 컨텍스트 길이
n_head = 4      # 어텐션 헤드 수
head_dim = n_embd // n_head  # = 16 // 4 = 4
```

**하이퍼파라미터(hyperparameter)**는 학습으로 바꿀 수 없고 사람이 미리 정해야 하는 설정값입니다.

| 하이퍼파라미터 | 값 | 의미 | 왜 이 값인가 |
|--------------|---|------|------------|
| `n_layer` | 1 | Transformer 층 수 | 이름 생성처럼 단순한 과제는 1층으로 충분 |
| `n_embd` | 16 | 각 토큰을 표현하는 숫자 개수 | 작은 모델로 빠르게 실험 |
| `block_size` | 16 | 한 번에 볼 수 있는 최대 글자 수 | 이름의 최대 길이(15)보다 살짝 크게 |
| `n_head` | 4 | 어텐션 헤드 수 | n_embd를 나누어 떨어지는 값 |

이 코드의 목표는 "최소한의 코드로 GPT를 구현"하는 것이므로 모든 값이 작습니다. GPT-2(Small)는 n_layer=12, n_embd=768, n_head=12입니다.

---

## 3. 파라미터 초기화 — 모델이 "아는 것"을 만들기

```python
# microgpt.py:80-90
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte':    matrix(vocab_size, n_embd),  # Word Token Embedding: 각 토큰 → 벡터
    'wpe':    matrix(block_size, n_embd),  # Word Position Embedding: 각 위치 → 벡터
    'lm_head': matrix(vocab_size, n_embd), # 출력층: 벡터 → 다음 토큰 확률
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query 프로젝션
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key 프로젝션
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value 프로젝션
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Output 프로젝션
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # MLP 확장층
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # MLP 축소층

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")  # 4,192
```

### 가우시안 초기화

```python
Value(random.gauss(0, std))   # 평균 0, 표준편차 0.08
```

파라미터를 0 근처의 작은 랜덤 값으로 시작합니다. 너무 크면 학습이 불안정해지고, 너무 작으면 신호가 사라집니다. 0.08은 실험적으로 잘 동작하는 값입니다.

### 총 파라미터 수: 4,192개

| 파라미터 행렬 | 크기 | 개수 |
|------------|------|------|
| `wte` | vocab_size(27) × n_embd(16) | 432 |
| `wpe` | block_size(16) × n_embd(16) | 256 |
| `lm_head` | vocab_size(27) × n_embd(16) | 432 |
| `attn_wq/wk/wv/wo` (각 4개) | 16 × 16 | 1,024 |
| `mlp_fc1` | 64 × 16 | 1,024 |
| `mlp_fc2` | 16 × 64 | 1,024 |
| **합계** | | **4,192** |

---

## 4. 어텐션 메커니즘 — 관련 정보 찾기

어텐션(Attention)은 Transformer의 핵심입니다. 비유로 먼저 이해해 봅시다.

### 비유: 도서관에서 답 찾기

당신이 도서관에서 "파이썬 프로그래밍"에 관한 책을 찾는다고 합시다.

1. **질문(Query)**: "파이썬 프로그래밍에 관한 책 있어요?"
2. **색인(Key)**: 각 책의 제목/키워드 목록
3. **내용(Value)**: 실제 책의 내용

질문과 색인을 비교해서 관련도(어텐션 점수)를 계산하고, 관련도가 높은 책의 내용을 가져옵니다.

언어 모델에서:
- **Q (Query)**: "지금 이 토큰이 궁금해하는 것"
- **K (Key)**: "각 이전 토큰이 광고하는 정보"
- **V (Value)**: "각 이전 토큰이 실제로 제공하는 정보"

### 코드로 보기

```python
# microgpt.py:108-133
def gpt(token_id, pos_id, keys, values):
    # 토큰 임베딩 + 위치 임베딩
    tok_emb = state_dict['wte'][token_id]   # 이 토큰의 의미 벡터
    pos_emb = state_dict['wpe'][pos_id]     # 이 위치의 위치 벡터
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 둘을 합침
    x = rmsnorm(x)
```

**토큰 임베딩**: 각 글자를 16차원 벡터로 변환합니다. 'a'라는 글자가 어떤 의미인지를 16개의 숫자로 표현합니다.

**위치 임베딩**: 이 글자가 몇 번째 위치에 있는지를 16차원 벡터로 표현합니다. 같은 글자 'a'라도 이름의 첫 자리와 중간에 있을 때 역할이 다릅니다.

```python
    for li in range(n_layer):
        # 어텐션 블록
        x_residual = x           # 나중에 더할 원본 저장
        x = rmsnorm(x)           # 정규화

        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query 계산
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key 계산
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value 계산

        keys[li].append(k)    # 이 토큰의 Key를 캐시에 저장
        values[li].append(v)  # 이 토큰의 Value를 캐시에 저장
```

`keys`와 `values` 리스트에 이전 토큰들의 K, V를 저장해둡니다. 이것이 **KV 캐시**입니다. 이미 계산한 값을 재사용하므로 효율적입니다.

```python
        x_attn = []
        for h in range(n_head):      # 헤드 수(4)만큼 반복
            hs = h * head_dim        # 이 헤드의 시작 인덱스
            q_h = q[hs:hs+head_dim]  # 이 헤드의 Query (길이 4)
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # 모든 이전 토큰의 Key
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 모든 이전 토큰의 Value

            # 어텐션 점수: Q와 K의 유사도 (내적)
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)  # 합이 1인 확률로 변환

            # 어텐션 가중치로 Value를 가중합산
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
```

**스케일드 닷-프로덕트 어텐션**: Q와 K를 내적(dot product)해서 유사도를 계산합니다. `/ head_dim**0.5`는 차원이 커질수록 내적값이 커지는 것을 막기 위한 스케일링입니다.

**멀티헤드**: 4개의 헤드가 **동시에** 서로 다른 관점으로 어텐션을 계산합니다. 마치 4명이 같은 문장을 읽고 각자 다른 측면에 주목하는 것과 같습니다.

---

## 5. MLP 블록 — 정보 처리

```python
# microgpt.py:135-141
        # Output 프로젝션 + 잔차 연결
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결 1

        # MLP 블록
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 16 → 64 (4배 확장)
        x = [xi.relu() for xi in x]                       # 비선형 활성화
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 64 → 16 (축소)
        x = [a + b for a, b in zip(x, x_residual)]        # 잔차 연결 2
```

MLP(Multi-Layer Perceptron)는 어텐션이 수집한 정보를 처리하는 블록입니다.

**확장 → ReLU → 축소**:
- 16차원 → 64차원: 더 많은 계산 공간 확보 (4배 확장이 GPT 관례)
- ReLU: 음수를 0으로 만들어 비선형성 추가
- 64차원 → 16차원: 다시 원래 크기로

**잔차 연결(Residual Connection)**: `x = x_processed + x_original`

비유: 새로운 정보를 원본에 **더하는** 것입니다. "기존 지식 + 새로 배운 것"처럼 덧붙입니다. 이렇게 하면 기울기가 깊은 네트워크에서도 잘 흐릅니다.

---

## 6. 학습 루프

```python
# microgpt.py:147-149
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # Adam 1차 모멘트 버퍼
v = [0.0] * len(params)  # Adam 2차 모멘트 버퍼

# microgpt.py:152-184
num_steps = 1000
for step in range(num_steps):
```

### 데이터 준비

```python
    # 1. 문서 선택 및 토큰화
    doc = docs[step % len(docs)]                          # 순환하며 이름 선택
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]  # "alan" → [26, 0, 11, 0, 13, 26]
    n = min(block_size, len(tokens) - 1)                  # 처리할 위치 수
```

예시: `doc = "alan"`, `uchars = ['a', 'b', ..., 'z']`, BOS = 26

```
"alan" → [BOS, a, l, a, n, BOS] = [26, 0, 11, 0, 13, 26]
```

### 순전파와 loss 계산

```python
    # 2. 순전파: 각 위치에서 다음 토큰 예측
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id = tokens[pos_id]      # 현재 토큰
        target_id = tokens[pos_id + 1] # 예측해야 할 다음 토큰
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # 크로스 엔트로피
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # 평균 loss
```

이것을 **Teacher Forcing**이라고 합니다. 예측 중에 틀려도 정답 토큰을 다음 입력으로 사용합니다.

예시 (doc = "alan"):

| pos_id | 입력 토큰 | 예측해야 할 토큰 |
|--------|----------|---------------|
| 0      | BOS (26) | 'a' (0)       |
| 1      | 'a' (0)  | 'l' (11)      |
| 2      | 'l' (11) | 'a' (0)       |
| 3      | 'a' (0)  | 'n' (13)      |
| 4      | 'n' (13) | BOS (26)      |

### 역전파

```python
    # 3. 역전파: 기울기 계산
    loss.backward()
```

단 한 줄입니다. Ch04에서 구현한 `backward()`가 계산 그래프 전체를 역방향으로 순회하며 모든 파라미터의 `grad`를 채웁니다.

### Adam 업데이트

```python
    # 4. Adam 파라미터 업데이트
    lr_t = learning_rate * (1 - step / num_steps)  # 선형 학습률 감쇠
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad           # 1차 모멘트 (속도)
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2      # 2차 모멘트 (분산)
        m_hat = m[i] / (1 - beta1 ** (step + 1))             # 편향 보정
        v_hat = v[i] / (1 - beta2 ** (step + 1))             # 편향 보정
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)   # 업데이트
        p.grad = 0                                            # 기울기 초기화
```

**Adam(Adaptive Moment Estimation)**은 경사하강법의 개선판입니다.

기본 경사하강법의 문제: 모든 파라미터에 같은 학습률을 사용합니다.

Adam의 해결책:

| 개념 | 코드 | 의미 |
|------|------|------|
| 1차 모멘트 `m` | `beta1 * m + (1-beta1) * grad` | 기울기의 이동 평균 (방향) |
| 2차 모멘트 `v` | `beta2 * v + (1-beta2) * grad²` | 기울기 제곱의 이동 평균 (크기) |
| 편향 보정 | `m / (1 - beta1^t)` | 초기 스텝에서의 과소추정 보정 |
| 업데이트 | `m_hat / (v_hat^0.5 + eps)` | 크기가 큰 기울기는 덜, 작은 기울기는 더 업데이트 |

비유: 평소에 많이 업데이트되는 파라미터는 조심스럽게, 잘 업데이트 안 되는 파라미터는 적극적으로 건드립니다.

**선형 학습률 감쇠**: `lr_t = learning_rate * (1 - step / num_steps)`

학습이 진행될수록 학습률을 선형으로 줄입니다. 처음엔 크게 탐색하고, 나중엔 섬세하게 조정합니다.

---

## 7. 추론 — 학습된 모델로 이름 만들기

```python
# microgpt.py:187-200
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS   # BOS 토큰에서 시작
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])  # temperature 적용
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:  # BOS를 만나면 이름 완성
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

### Temperature — 창의성 조절 손잡이

**temperature**는 확률 분포의 날카로움을 조절합니다.

- `temperature = 1.0`: 모델의 원래 확률 그대로 사용
- `temperature < 1.0` (예: 0.5): 확률 분포가 더 날카로워짐 → 확신이 높은 토큰을 더 강하게 선호
- `temperature > 1.0`: 분포가 평평해짐 → 더 다양하고 예상하기 어려운 결과

`logits / temperature`를 softmax에 넣으면 이 효과가 납니다. temperature=0.5이면 logits를 0.5로 나누므로 (= 2를 곱하므로) 차이가 더 커집니다.

### 확률적 샘플링

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

가장 확률 높은 토큰을 항상 선택하는 것(greedy)이 아니라, **확률에 비례해서 랜덤하게** 선택합니다. 그래서 실행할 때마다 다른 이름이 생성됩니다.

### BOS 토큰의 역할

- 시작: BOS로 시작하면 "이름의 첫 글자를 예측해"라는 신호
- 종료: BOS가 출력되면 "이름이 끝났다"는 신호

---

## 8. 전체 실행 결과

```
$ python microgpt.py
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.0341

--- inference (new, hallucinated names) ---
sample  1: alan
sample  2: jaynel
sample  3: mara
sample  4: kelan
sample  5: alis
...
```

- 초기 loss ≈ 3.4: 랜덤 추측 수준 (`log(27) ≈ 3.3`)
- 최종 loss ≈ 2.0: 데이터의 패턴을 학습

이름처럼 들리는 단어들을 생성합니다. 실제로 존재하지 않는 이름도 있지만, 영어 이름의 패턴(자음/모음 배치, 흔한 접미사 등)을 학습했기 때문입니다.

---

## 9. 전체 코드 한 눈에 보기

아래는 `microgpt.py` 전체입니다. 이제 모든 줄이 이해됩니다.

```python
"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# [1] 데이터: 이름 목록 로드
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# [2] 토크나이저: 문자 ↔ 정수 변환
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# [3] Value 클래스: 자동 미분 (Ch04 참조)
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# [4] 하이퍼파라미터 및 파라미터 초기화
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# [5] 모델 정의: Transformer 순전파
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# [6] 학습 루프: Adam으로 1000 스텝
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    loss.backward()
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# [7] 추론: 이름 생성
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

---

## 10. 핵심 정리

| 개념 | 한 줄 설명 |
|------|-----------|
| 하이퍼파라미터 | 사람이 미리 정하는 설정값 (n_layer, n_embd, ...) |
| 가우시안 초기화 | 파라미터를 0 근처 작은 랜덤 값으로 시작 |
| 토큰/위치 임베딩 | 글자와 위치를 숫자 벡터로 변환 |
| Q, K, V 어텐션 | "질문으로 관련 정보를 찾아 답을 가져온다" |
| 멀티헤드 | 여러 관점으로 동시에 어텐션 계산 |
| KV 캐시 | 이전 토큰의 K, V를 저장해 재사용 |
| 잔차 연결 | `x = x_processed + x_original` 로 기울기 흐름 보장 |
| Teacher Forcing | 학습 중 정답 토큰을 다음 입력으로 사용 |
| Adam | 파라미터별 적응형 학습률을 쓰는 옵티마이저 |
| Temperature | 0에 가까울수록 결정적, 높을수록 창의적인 출력 (범위: (0, 1]) |

### Part 1 완료

축하합니다. `microgpt.py` 200줄을 전부 이해했습니다.

- Ch01-02: 데이터와 토크나이저
- Ch03: 손실 함수와 경사하강법
- Ch04: 자동 미분 (Value 클래스)
- Ch05: Transformer, 학습, 추론

**Part 2**에서는 이 Python 코드를 C++로 다시 구현합니다. 같은 알고리즘을 훨씬 빠르게 실행하는 방법을 배웁니다.


> **직접 체험하기** — 시각화 도구에서 어텐션 히트맵과 학습률 변화를 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#training)

---
[< 이전: Ch04: Value와 역전파](ch04-value-and-backprop.md) | [목차](../README.md) | [다음: Ch06: C++ 환경 설정 >](../part2-cpp/ch06-cpp-setup.md)
