# Ch15: GPT 순전파

## 학습 목표

- `gpt()` 함수의 시그니처를 읽고 각 인자의 역할을 설명할 수 있다
- 토큰 임베딩과 위치 임베딩을 더해 초기 표현을 만드는 과정을 따라간다
- KV 캐시가 무엇이고 왜 필요한지 이해한다
- 멀티헤드 어텐션의 Q·K^T / √d_k 계산을 한 단계씩 추적한다
- MLP 블록의 확장-활성화-축소 구조를 코드로 작성한다

---

## 개념 설명

### 순전파(Forward Pass)란?

순전파는 입력 토큰이 모델을 통과해서 출력(로짓)이 나오는 과정입니다.

```
입력: token_id=4 (='e'), pos_id=0
          ↓
   [토큰 임베딩 + 위치 임베딩]
          ↓
      [RMSNorm]
          ↓
   [멀티헤드 어텐션]  ← KV 캐시 사용
          ↓
      [잔차 연결]
          ↓
      [RMSNorm]
          ↓
     [MLP 블록]
          ↓
      [잔차 연결]
          ↓
  [선형 프로젝션 (lm_head)]
          ↓
출력: logits (vocab_size=27개의 점수)
```

이 계산 과정이 모두 `Value` 노드들로 연결되어 계산 그래프를 형성합니다. 나중에 `backward(loss)`를 호출하면 이 그래프를 거꾸로 따라가며 기울기를 계산합니다.

### KV 캐시란?

GPT는 자기회귀(autoregressive) 방식으로 동작합니다. "emma"를 처리할 때:

```
위치 0: 'e'를 처리 → keys[0][0], vals[0][0] 저장
위치 1: 'm'을 처리 → keys[0][1], vals[0][1] 저장 + 위치 0의 K,V 재사용
위치 2: 'm'을 처리 → keys[0][2], vals[0][2] 저장 + 위치 0,1의 K,V 재사용
위치 3: 'a'를 처리 → keys[0][3], vals[0][3] 저장 + 위치 0,1,2의 K,V 재사용
```

이전 위치에서 계산한 K(키)와 V(밸류)를 버리지 않고 캐시에 저장해 두는 것이 KV 캐시입니다. 덕분에 각 위치를 처리할 때 이전 모든 위치의 정보를 활용할 수 있습니다.

```cpp
// keys: N_LAYER개의 레이어, 각 레이어는 위치마다 Vec(N_EMBD 크기)
std::vector<std::vector<Vec>> keys(N_LAYER), vals(N_LAYER);

// 위치 pos를 처리하면서 현재 K, V를 추가
keys[li].push_back(k);  // 이번 위치의 Key 저장
vals[li].push_back(v);  // 이번 위치의 Value 저장
// keys[li].size() == 현재까지 처리한 위치 수 T
```

### 멀티헤드 어텐션 직관

어텐션의 핵심 질문: **"현재 위치가 과거의 어떤 위치에 주목해야 하는가?"**

```
Q (Query):  "내가 찾는 것" — 현재 위치의 질문
K (Key):    "각 위치의 라벨" — 각 위치가 가진 정보의 식별자
V (Value):  "각 위치의 내용" — 각 위치가 실제로 담고 있는 정보

어텐션 = softmax(Q · K^T / √d_k) · V

Q · K^T → 현재 위치와 모든 과거 위치의 유사도 점수
/ √d_k  → 스케일링 (차원이 클수록 내적값이 커지는 것 보정)
softmax → 점수를 확률 분포로 변환 (합이 1)
· V     → 확률 가중치로 Value를 합산
```

4개의 헤드가 각자 독립적으로 이 계산을 수행합니다. 각 헤드는 4차원 부분 공간을 담당합니다.

```
N_EMBD=16 차원을 4개의 헤드가 분담:
헤드 0: 차원  0~ 3  (HEAD_DIM=4)
헤드 1: 차원  4~ 7
헤드 2: 차원  8~11
헤드 3: 차원 12~15
```

---

## Python ↔ C++ 전체 비교표

| 단계 | Python | C++ |
|------|--------|-----|
| 함수 시그니처 | `def gpt(token_id, pos_id, keys, values)` | `Vec gpt(int token_id, int pos_id, vector<vector<Vec>>& keys, vector<vector<Vec>>& vals, unordered_map<string, Mat>& sd)` |
| 토큰 임베딩 | `state_dict['wte'][token_id]` | `sd["wte"][token_id]` |
| 임베딩 합산 | `[t + p for t, p in zip(tok_emb, pos_emb)]` | `x[i] = add(sd["wte"][token_id][i], sd["wpe"][pos_id][i])` |
| RMSNorm | `x = rmsnorm(x)` | `x = rmsnorm(x)` |
| 레이어 루프 | `for li in range(n_layer):` | `for (int li = 0; li < N_LAYER; li++)` |
| 레이어 접두사 | `f'layer{li}.attn_wq'` | `"layer" + std::to_string(li) + ".attn_wq"` |
| Q 프로젝션 | `q = linear(x, state_dict[...])` | `Vec q = linear(x, sd[pfx + ".attn_wq"])` |
| KV 캐시 추가 | `keys[li].append(k)` | `keys[li].push_back(k)` |
| 헤드 슬라이스 | `q_h = q[hs:hs+head_dim]` | `Vec q_h(q.begin()+hs, q.begin()+hs+HEAD_DIM)` |
| 어텐션 로짓 내적 | `sum(q_h[j] * k_h[t][j] for j in range(head_dim))` | for loop + `add(dot, mul(q_h[j], keys[li][t][hs+j]))` |
| 스케일링 | `/ head_dim**0.5` | `scale(dot, 1.0 / sqrt(HEAD_DIM))` |
| 어텐션 가중치 | `attn_weights = softmax(attn_logits)` | `Vec attn_weights = softmax(attn_logits)` |
| 가중 합산 | `sum(attn_weights[t] * v_h[t][j] for t in ...)` | for loop + `add(s, mul(attn_weights[t], vals[li][t][hs+j]))` |
| 출력 프로젝션 | `x = linear(x_attn, state_dict[...])` | `x = linear(x_attn, sd[pfx + ".attn_wo"])` |
| 잔차 연결 | `[a + b for a, b in zip(x, x_residual)]` | `x[i] = add(x[i], x_residual[i])` |
| MLP 확장 | `x = linear(x, state_dict[...])` | `x = linear(x, sd[pfx + ".mlp_fc1"])` |
| ReLU | `[xi.relu() for xi in x]` | `for (auto& xi : x) xi = relu(xi)` |
| MLP 축소 | `x = linear(x, state_dict[...])` | `x = linear(x, sd[pfx + ".mlp_fc2"])` |
| 최종 로짓 | `linear(x, state_dict['lm_head'])` | `linear(x, sd["lm_head"])` |

---

## 코드 작성 — 한 줄씩

### gpt() 함수 시그니처

```cpp
Vec gpt(int token_id, int pos_id,
        std::vector<std::vector<Vec>>& keys,
        std::vector<std::vector<Vec>>& vals,
        std::unordered_map<std::string, Mat>& sd) {
```

각 인자의 의미:

```
token_id   — 현재 처리할 토큰의 ID (0~26)
pos_id     — 시퀀스에서의 위치 (0~BLOCK_SIZE-1)
keys       — KV 캐시의 Key 부분: keys[레이어][위치] = Vec(N_EMBD)
vals       — KV 캐시의 Value 부분: vals[레이어][위치] = Vec(N_EMBD)
sd         — state_dict: 이름 → 가중치 행렬
```

`&` 기호는 "참조(reference)"입니다. 복사 없이 원본을 직접 사용합니다. KV 캐시는 함수 호출마다 누적되어야 하므로 반드시 참조로 전달해야 합니다.

### 1단계: 토큰 임베딩 + 위치 임베딩

```cpp
    // 토큰 임베딩 + 위치 임베딩을 element-wise로 더해 초기 표현 생성
    Vec x(N_EMBD);
    for (int i = 0; i < N_EMBD; i++)
        x[i] = add(sd["wte"][token_id][i], sd["wpe"][pos_id][i]);
    x = rmsnorm(x);
```

설명:
- `sd["wte"][token_id]` — wte 행렬의 `token_id`번 행 = 토큰의 의미 벡터
- `sd["wpe"][pos_id]` — wpe 행렬의 `pos_id`번 행 = 위치 정보 벡터
- 두 벡터를 원소별로 더하면 "이 위치의 이 토큰"을 나타내는 벡터가 됩니다
- `rmsnorm(x)`: 값의 크기를 정규화하여 학습 안정성 확보

### 2단계: 레이어 루프 시작

```cpp
    for (int li = 0; li < N_LAYER; li++) {
        std::string pfx = "layer" + std::to_string(li);
        // pfx = "layer0" (N_LAYER=1이므로 li는 0만)
```

Python의 `f'layer{li}'`와 동일합니다. `std::to_string(0)` = `"0"`.

### 3단계: 잔차 연결 저장 + RMSNorm

```cpp
        // 어텐션 블록 시작
        Vec x_residual = x;  // 잔차 연결을 위해 입력 보존
        x = rmsnorm(x);      // 정규화
```

`Vec x_residual = x`는 `Vec`의 복사 생성자를 호출합니다. 포인터들의 배열이 복사되므로, `x_residual[i]`와 `x[i]`는 처음에 같은 `Value*`를 가리킵니다. 이후 `x`를 rmsnorm으로 교체해도 `x_residual`은 원래 값을 유지합니다.

### 4단계: Q, K, V 프로젝션

```cpp
        // 입력 x를 Query, Key, Value 세 공간으로 투영
        Vec q = linear(x, sd[pfx + ".attn_wq"]);
        Vec k = linear(x, sd[pfx + ".attn_wk"]);
        Vec v = linear(x, sd[pfx + ".attn_wv"]);
```

각각 `N_EMBD × N_EMBD` 행렬로 선형 변환합니다. 같은 `x`에서 세 개의 다른 투영을 만들어 각자의 역할(Query/Key/Value)을 부여합니다.

### 5단계: KV 캐시에 추가

```cpp
        // 현재 위치의 K, V를 캐시에 저장
        keys[li].push_back(k);
        vals[li].push_back(v);
        // 이제 keys[li].size() == pos_id + 1 == 지금까지 처리한 위치 수 T
```

### 6단계: 멀티헤드 어텐션

```cpp
        Vec x_attn;
        x_attn.reserve(N_EMBD);

        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;  // 이 헤드의 시작 인덱스

            // 현재 헤드의 Query 슬라이스 (HEAD_DIM=4개 원소)
            Vec q_h(q.begin() + hs, q.begin() + hs + HEAD_DIM);

            int T = static_cast<int>(keys[li].size());  // 캐시된 위치 수

            // Q · K^T / √d_k 계산
            double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
            Vec attn_logits;
            attn_logits.reserve(T);
            for (int t = 0; t < T; t++) {
                // 위치 t의 Key와 현재 Query의 내적
                Value* dot = graph.make(0.0);
                for (int j = 0; j < HEAD_DIM; j++)
                    dot = add(dot, mul(q_h[j], keys[li][t][hs + j]));
                // √d_k로 스케일링
                attn_logits.push_back(scale(dot, inv_sqrt));
            }

            // softmax → 어텐션 가중치 (합이 1인 확률 분포)
            Vec attn_weights = softmax(attn_logits);

            // 어텐션 가중치로 Value를 가중 합산
            for (int j = 0; j < HEAD_DIM; j++) {
                Value* s = graph.make(0.0);
                for (int t = 0; t < T; t++)
                    s = add(s, mul(attn_weights[t], vals[li][t][hs + j]));
                x_attn.push_back(s);
            }
        }
        // x_attn.size() == N_HEAD * HEAD_DIM == N_EMBD == 16
```

단계별 설명:

```
hs = h * HEAD_DIM
     ↑
     헤드 h가 담당하는 차원의 시작점
     헤드 0: hs=0, 헤드 1: hs=4, 헤드 2: hs=8, 헤드 3: hs=12

Vec q_h(q.begin() + hs, q.begin() + hs + HEAD_DIM)
     ↑
     q[hs .. hs+HEAD_DIM) 범위의 원소들로 새 Vec 생성
     Python의 q[hs:hs+head_dim]과 동일

keys[li][t][hs + j]
     ↑   ↑  ↑
     │   │  j번째 원소 (0 ~ HEAD_DIM-1)
     │   위치 t의 Key 벡터
     레이어 li의 KV 캐시
```

### 7단계: 출력 프로젝션 + 잔차 연결

```cpp
        // 4개 헤드의 결과를 합쳐서 다시 N_EMBD 차원으로 프로젝션
        x = linear(x_attn, sd[pfx + ".attn_wo"]);
        // 잔차 연결: 어텐션 결과와 원래 입력을 더함
        for (int i = 0; i < N_EMBD; i++)
            x[i] = add(x[i], x_residual[i]);
```

잔차 연결(residual connection)은 "입력을 그대로 건너뛰어 더해 주는" 연결입니다. 덕분에 기울기가 깊은 네트워크를 통과할 때 사라지지 않습니다(vanishing gradient 방지).

### 8단계: MLP 블록

```cpp
        // MLP 블록 시작
        x_residual = x;  // 이번엔 MLP를 위한 잔차 저장
        x = rmsnorm(x);
        x = linear(x, sd[pfx + ".mlp_fc1"]);  // N_EMBD → 4*N_EMBD (16 → 64)
        for (auto& xi : x) xi = relu(xi);      // 비선형 활성화
        x = linear(x, sd[pfx + ".mlp_fc2"]);  // 4*N_EMBD → N_EMBD (64 → 16)
        for (int i = 0; i < N_EMBD; i++)
            x[i] = add(x[i], x_residual[i]);  // 잔차 연결
    }
    // 레이어 루프 끝
```

MLP(Multi-Layer Perceptron)는 먼저 4배로 확장했다가 다시 원래 크기로 줄입니다. 이 "병목(bottleneck)" 구조에서 ReLU가 비선형성을 주입합니다. GPT-2는 GeLU를 쓰지만 여기서는 ReLU로 단순화했습니다.

### 9단계: 최종 출력

```cpp
    // 최종 선형 프로젝션: 임베딩 차원 → 어휘 크기 로짓
    return linear(x, sd["lm_head"]);
    // 반환값: Vec(vocab_size=27) — 각 토큰이 다음에 올 확률 점수
}
```

---

## 완성된 gpt() 함수 전체

```cpp
Vec gpt(int token_id, int pos_id,
        std::vector<std::vector<Vec>>& keys,
        std::vector<std::vector<Vec>>& vals,
        std::unordered_map<std::string, Mat>& sd) {

    // 1) 임베딩
    Vec x(N_EMBD);
    for (int i = 0; i < N_EMBD; i++)
        x[i] = add(sd["wte"][token_id][i], sd["wpe"][pos_id][i]);
    x = rmsnorm(x);

    for (int li = 0; li < N_LAYER; li++) {
        std::string pfx = "layer" + std::to_string(li);

        // 2) 멀티헤드 어텐션
        Vec x_residual = x;
        x = rmsnorm(x);
        Vec q = linear(x, sd[pfx + ".attn_wq"]);
        Vec k = linear(x, sd[pfx + ".attn_wk"]);
        Vec v = linear(x, sd[pfx + ".attn_wv"]);
        keys[li].push_back(k);
        vals[li].push_back(v);

        Vec x_attn;
        x_attn.reserve(N_EMBD);
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;
            Vec q_h(q.begin() + hs, q.begin() + hs + HEAD_DIM);
            int T = static_cast<int>(keys[li].size());
            double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
            Vec attn_logits;
            attn_logits.reserve(T);
            for (int t = 0; t < T; t++) {
                Value* dot = graph.make(0.0);
                for (int j = 0; j < HEAD_DIM; j++)
                    dot = add(dot, mul(q_h[j], keys[li][t][hs + j]));
                attn_logits.push_back(scale(dot, inv_sqrt));
            }
            Vec attn_weights = softmax(attn_logits);
            for (int j = 0; j < HEAD_DIM; j++) {
                Value* s = graph.make(0.0);
                for (int t = 0; t < T; t++)
                    s = add(s, mul(attn_weights[t], vals[li][t][hs + j]));
                x_attn.push_back(s);
            }
        }
        x = linear(x_attn, sd[pfx + ".attn_wo"]);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = add(x[i], x_residual[i]);

        // 3) MLP
        x_residual = x;
        x = rmsnorm(x);
        x = linear(x, sd[pfx + ".mlp_fc1"]);
        for (auto& xi : x) xi = relu(xi);
        x = linear(x, sd[pfx + ".mlp_fc2"]);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = add(x[i], x_residual[i]);
    }

    return linear(x, sd["lm_head"]);
}
```

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 순전파 | 토큰 → 임베딩 → 어텐션 → MLP → 로짓 |
| KV 캐시 | 이전 위치의 K,V를 저장해 재사용. `push_back`으로 축적 |
| Q·K^T/√d_k | 현재 위치와 모든 과거 위치의 유사도. `mul`+`add` 루프로 구현 |
| 헤드 슬라이스 | `Vec q_h(q.begin()+hs, q.begin()+hs+HEAD_DIM)` = Python의 `q[hs:hs+head_dim]` |
| 잔차 연결 | `x[i] = add(x[i], x_residual[i])` — 기울기 소실 방지 |
| MLP 구조 | N_EMBD → 4*N_EMBD → (ReLU) → N_EMBD |
| 최종 출력 | `linear(x, sd["lm_head"])` — vocab_size개의 로짓 반환 |

다음 챕터에서는 이 `gpt()` 함수를 반복 호출하고, 역전파와 Adam 업데이트로 파라미터를 학습시키는 훈련 루프를 구현합니다.


> **직접 체험하기** — 시각화 도구에서 GPT 순전파의 아키텍처 흐름과 어텐션 히트맵을 실시간으로 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#inference)

---
[< 이전: Ch14: 파라미터 초기화](ch14-parameters.md) | [목차](../README.md) | [다음: Ch16: 학습 루프 >](ch16-training-loop.md)
