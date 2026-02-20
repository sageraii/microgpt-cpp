# Ch16: 학습 루프

## 학습 목표

- Adam 옵티마이저가 왜 단순 경사하강법보다 나은지 이해한다
- 1차 모멘트(m)와 2차 모멘트(v)가 무엇인지, 어떻게 업데이트되는지 설명할 수 있다
- 바이어스 보정(bias correction)이 왜 필요한지 이해한다
- 학습 루프의 7단계를 순서대로 코드로 작성한다
- `graph.clear()`를 어디서 호출해야 하는지, 그리고 왜 파라미터에는 영향이 없는지 설명한다

---

## 개념 설명

### Adam 옵티마이저란?

단순 경사하강법은 모든 파라미터에 같은 학습률을 적용합니다. 문제는 파라미터마다 기울기의 크기가 천차만별이라는 점입니다. 어떤 파라미터는 기울기가 0.001인데, 어떤 파라미터는 10.0일 수 있습니다.

Adam(Adaptive Moment Estimation)은 **각 파라미터마다 적응적인 학습률**을 사용합니다. 기울기가 안정적인 방향으로는 빠르게, 기울기가 요동치는 방향으로는 천천히 업데이트합니다.

```
단순 경사하강법: p -= lr * grad
                 ↑
                 모든 파라미터 동일한 lr

Adam:            p -= lr * m_hat / (√v_hat + ε)
                          ↑               ↑
                     1차 모멘트       2차 모멘트
                  (기울기의 방향)  (기울기의 크기, 적응적 lr)
```

### 1차 모멘트 m — "기울기의 이동 평균"

```
m = β1 * m + (1 - β1) * grad

β1 = 0.85 (이번 튜토리얼의 값, 원래 Adam은 0.9)

의미: 과거 기울기들의 지수 이동 평균 (Exponential Moving Average)
      과거 기울기가 일관되게 같은 방향을 가리키면 m이 커짐
      → "이 방향으로 계속 나아가라"는 모멘텀(momentum) 역할
```

물리적 비유: 공이 비탈을 굴러가는 관성. 방향이 바뀌어도 이전 방향의 속도가 남아 있음.

### 2차 모멘트 v — "기울기 제곱의 이동 평균"

```
v = β2 * v + (1 - β2) * grad²

β2 = 0.99

의미: 과거 기울기 제곱들의 지수 이동 평균
      기울기가 자주 크게 흔들리면 v가 커짐
      → √v로 나누면 학습률이 줄어들어 "흔들리는 파라미터는 천천히"
      기울기가 안정적이면 v가 작음
      → 학습률 거의 그대로 유지 "안정적인 파라미터는 빠르게"
```

### 바이어스 보정(Bias Correction)이 필요한 이유

학습 초기에 m과 v는 모두 0으로 초기화됩니다. 첫 번째 스텝에서:

```
m = 0.85 * 0 + 0.15 * grad = 0.15 * grad

실제 기울기는 grad인데, m은 grad의 15%에 불과
→ 첫 스텝들에서 학습률이 너무 작아지는 현상
```

이를 보정하기 위해 t 번째 스텝에서:

```
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)

t=1:  1 - 0.85^1 = 0.15  → m / 0.15 = m * 6.67  (크게 보정)
t=10: 1 - 0.85^10 = 0.80 → m / 0.80 = m * 1.25  (조금 보정)
t=50: 1 - 0.85^50 ≈ 1.0  → m / 1.0  ≈ m        (보정 거의 없음)

시간이 지날수록 보정이 작아짐 → m과 v가 실제값으로 수렴했으니
```

### 선형 학습률 감쇠(Linear LR Decay)

```
lr_t = lr * (1 - step / num_steps)

step=0:    lr_t = 0.01 * 1.0   = 0.01  (최대)
step=500:  lr_t = 0.01 * 0.5   = 0.005 (절반)
step=999:  lr_t = 0.01 * 0.001 = 0.00001 (거의 0)
```

학습 초기에는 빠르게 탐색하고, 후반에는 세밀하게 조정합니다. 학습이 수렴에 가까워질수록 조금씩 움직여 안정적인 최솟값을 찾습니다.

### 계산 그래프 해제 — graph.clear()의 타이밍

```
학습 스텝 i:
  순전파  → 계산 그래프 생성 (graph에 임시 Value* 누적)
  역전파  → 기울기 계산
  Adam   → 파라미터 업데이트 (파라미터는 graph 밖에 있음!)
  graph.clear() ← 임시 Value*만 삭제. 파라미터는 살아있음

학습 스텝 i+1:
  순전파  → 새 계산 그래프 생성 (깨끗한 상태에서 시작)
  ...
```

`graph.clear()`를 하지 않으면 매 스텝마다 수십만 개의 임시 Value 노드가 누적되어 메모리가 폭발합니다.

---

## Python ↔ C++ 비교

### Adam 버퍼 초기화

```python
# Python (microgpt.py:147-148)
m = [0.0] * len(params)  # 1차 모멘트
v = [0.0] * len(params)  # 2차 모멘트
```

```cpp
// C++ (microgpt.cpp:415-416)
std::vector<double> m_buf(params.size(), 0.0);  // 1차 모멘트
std::vector<double> v_buf(params.size(), 0.0);  // 2차 모멘트
```

C++에서는 Python의 리스트 `[0.0] * n`을 `std::vector<double>(n, 0.0)`으로 작성합니다.

### 학습 루프 전체

```python
# Python (microgpt.py:152-183)
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
```

```cpp
// C++ (microgpt.cpp:419-473)
constexpr int num_steps = 1000;
for (int step = 0; step < num_steps; step++) {

    // 1) 문서 선택 및 토큰화
    const std::string& doc = docs[step % docs.size()];
    std::vector<int> tokens;
    tokens.reserve(doc.size() + 2);
    tokens.push_back(BOS);
    for (char c : doc) tokens.push_back(char_to_id[c]);
    tokens.push_back(BOS);
    int n = std::min(BLOCK_SIZE, static_cast<int>(tokens.size()) - 1);

    // 2) 순전파
    std::vector<std::vector<Vec>> keys(N_LAYER), vals(N_LAYER);
    std::vector<Value*> losses;
    for (int pos = 0; pos < n; pos++) {
        int token_id  = tokens[pos];
        int target_id = tokens[pos + 1];
        Vec logits = gpt(token_id, pos, keys, vals, state_dict);
        Vec probs  = softmax(logits);
        losses.push_back(neg(log(probs[target_id])));
    }

    // 3) 평균 손실 계산
    Value* total_loss = graph.make(0.0);
    for (auto* l : losses) total_loss = add(total_loss, l);
    Value* loss = scale(total_loss, 1.0 / n);

    // 4) 역전파
    backward(loss);

    // 5) Adam 업데이트
    double lr_t = learning_rate * (1.0 - static_cast<double>(step) / num_steps);
    for (size_t i = 0; i < params.size(); i++) {
        Value* p = params[i];
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p->grad;
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p->grad * p->grad;
        double m_hat = m_buf[i] / (1.0 - std::pow(beta1, step + 1));
        double v_hat = v_buf[i] / (1.0 - std::pow(beta2, step + 1));
        p->data -= lr_t * m_hat / (std::sqrt(v_hat) + eps_adam);
        p->grad = 0.0;
    }

    // 6) 진행 상황 출력
    std::printf("\rstep %4d / %4d | loss %.4f", step + 1, num_steps, loss->data);
    std::fflush(stdout);

    // 7) 계산 그래프 해제
    graph.clear();
}
```

---

## 코드 작성 — 한 줄씩

### 1단계: 상수와 버퍼 선언

```cpp
    // Adam 하이퍼파라미터
    constexpr double learning_rate = 0.01;
    constexpr double beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;

    // Adam 상태 버퍼 (파라미터 수만큼, 0으로 초기화)
    std::vector<double> m_buf(params.size(), 0.0);
    std::vector<double> v_buf(params.size(), 0.0);
```

`1e-8`은 `0.00000001`을 과학적 표기법으로 쓴 것입니다. 분모가 0이 되는 것을 방지하는 매우 작은 값(epsilon)입니다.

### 2단계: 루프 시작 및 문서 토큰화

```cpp
    constexpr int num_steps = 1000;
    for (int step = 0; step < num_steps; step++) {

        // 이번 스텝에서 사용할 문서 선택 (순환)
        const std::string& doc = docs[step % docs.size()];
        //                                ↑
        //                        나머지 연산: 32033개의 문서를 순환

        // 토큰 시퀀스 생성: [BOS, 글자1, 글자2, ..., BOS]
        std::vector<int> tokens;
        tokens.reserve(doc.size() + 2);  // 미리 공간 예약 (성능)
        tokens.push_back(BOS);
        for (char c : doc)
            tokens.push_back(char_to_id[c]);
        tokens.push_back(BOS);

        // BLOCK_SIZE를 초과하지 않도록 처리할 위치 수 제한
        int n = std::min(BLOCK_SIZE, static_cast<int>(tokens.size()) - 1);
        //                                                               ↑
        //                           tokens.size()-1: 마지막 위치의 target이 있어야 하므로
```

`step % docs.size()`: step=0이면 0번 문서, step=32033이면 다시 0번 문서. 모든 문서를 반복해서 학습합니다.

### 3단계: 순전파 + 손실 계산

```cpp
        // KV 캐시 초기화 (이 문서를 처음부터 처리)
        std::vector<std::vector<Vec>> keys(N_LAYER), vals(N_LAYER);
        std::vector<Value*> losses;

        for (int pos = 0; pos < n; pos++) {
            int token_id  = tokens[pos];       // 현재 토큰 (입력)
            int target_id = tokens[pos + 1];   // 다음 토큰 (정답)

            // gpt()는 현재 토큰을 입력받아 다음 토큰의 확률 분포를 출력
            Vec logits = gpt(token_id, pos, keys, vals, state_dict);
            Vec probs  = softmax(logits);

            // 크로스 엔트로피 손실: -log(정답 토큰의 확률)
            losses.push_back(neg(log(probs[target_id])));
        }

        // 시퀀스 전체의 평균 손실
        Value* total_loss = graph.make(0.0);
        for (auto* l : losses)
            total_loss = add(total_loss, l);
        Value* loss = scale(total_loss, 1.0 / n);
```

각 위치에서 "다음 글자가 뭔지" 예측합니다. "emma"라면:

```
pos=0: 입력=BOS, target='e'                            → loss_0 = -log(P(e))
pos=1: 입력='e',  target='m'                           → loss_1 = -log(P(m|e))
pos=2: 입력='m',  target='m'                           → loss_2 = -log(P(m|em))
pos=3: 입력='m',  target='a'                           → loss_3 = -log(P(a|emm))
pos=4: 입력='a',  target=BOS                           → loss_4 = -log(P(BOS|emma))
loss = (loss_0 + loss_1 + ... + loss_4) / 5
```

### 4단계: 역전파

```cpp
        backward(loss);
```

한 줄이지만 내부에서 수천 개의 Value 노드를 위상 정렬하고, 역방향으로 기울기를 누적합니다. 이 호출 후 모든 `params[i]->grad`에 기울기가 채워집니다.

### 5단계: Adam 업데이트

```cpp
        // 선형 학습률 감쇠: 처음에는 0.01, 마지막엔 거의 0
        double lr_t = learning_rate * (1.0 - static_cast<double>(step) / num_steps);

        for (size_t i = 0; i < params.size(); i++) {
            Value* p = params[i];

            // 1차 모멘트: 기울기의 지수 이동 평균
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p->grad;

            // 2차 모멘트: 기울기 제곱의 지수 이동 평균
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p->grad * p->grad;

            // 바이어스 보정
            double m_hat = m_buf[i] / (1.0 - std::pow(beta1, step + 1));
            double v_hat = v_buf[i] / (1.0 - std::pow(beta2, step + 1));

            // 파라미터 업데이트
            p->data -= lr_t * m_hat / (std::sqrt(v_hat) + eps_adam);

            // 기울기 리셋 (다음 스텝을 위해)
            p->grad = 0.0;
        }
```

`p->grad = 0.0`을 반드시 해야 합니다. 다음 스텝에서 `backward()`는 기울기를 누적(`+=`)하기 때문에, 이전 스텝의 기울기가 남아 있으면 오염됩니다.

### 6단계: 진행 상황 출력

```cpp
        std::printf("\rstep %4d / %4d | loss %.4f", step + 1, num_steps, loss->data);
        std::fflush(stdout);
```

`\r`은 캐리지 리턴(carriage return)입니다. 커서를 줄 처음으로 되돌려서, 다음 출력이 같은 줄을 덮어씁니다. 덕분에 진행 상황이 한 줄에 계속 갱신됩니다.

`std::fflush(stdout)`: 출력 버퍼를 즉시 비워 화면에 표시합니다. 이것 없이는 `\r`로 쓴 내용이 나타나지 않을 수 있습니다.

Python의 `print(..., end='\r')`에 해당합니다.

### 7단계: 계산 그래프 해제

```cpp
        graph.clear();
    }
    // 학습 루프 끝

    std::cout << "\n";  // 마지막 줄 바꿈
```

`graph.clear()`는 이번 스텝에서 생성된 모든 임시 Value 노드(`add`, `mul`, `softmax` 등의 결과)를 삭제합니다. 파라미터(`params`에 있는 Value*)는 `graph`가 소유하지 않으므로 삭제되지 않습니다.

---

## 기대 실행 결과

1000 스텝 학습 시 loss의 변화:

```
step    1 / 1000 | loss 3.4021
step  100 / 1000 | loss 2.8134
step  500 / 1000 | loss 2.3412
step 1000 / 1000 | loss 2.0187
```

- **step 1: loss ≈ 3.4** — 초기 랜덤 상태. 27개 토큰 중 랜덤 예측 시 기대 loss = ln(27) ≈ 3.30
- **step 1000: loss ≈ 2.0** — 학습 후. 이름의 패턴을 어느 정도 학습한 상태

loss가 0에 가까워지지는 않습니다. 이름 데이터는 본질적으로 불확실성이 있기 때문입니다. "an" 다음에 "n", "d", "e", "g" 등 여러 글자가 올 수 있고, 그것이 모두 맞습니다.

---

## 핵심 정리

| 개념 | Python | C++ | 의미 |
|------|--------|-----|------|
| Adam 버퍼 | `m = [0.0] * n` | `std::vector<double> m_buf(n, 0.0)` | 0으로 초기화된 double 배열 |
| 1차 모멘트 | `m[i] = β1*m[i] + (1-β1)*p.grad` | `m_buf[i] = beta1*m_buf[i] + (1-beta1)*p->grad` | 기울기의 이동 평균 |
| 2차 모멘트 | `v[i] = β2*v[i] + (1-β2)*p.grad**2` | `v_buf[i] = beta2*v_buf[i] + (1-beta2)*p->grad*p->grad` | 기울기 제곱의 이동 평균 |
| 바이어스 보정 | `m[i] / (1 - β1**(step+1))` | `m_buf[i] / (1.0 - pow(beta1, step+1))` | 초기 편향 제거 |
| 파라미터 업데이트 | `p.data -= lr * m_hat / (v_hat**0.5 + eps)` | `p->data -= lr_t * m_hat / (sqrt(v_hat) + eps_adam)` | 적응적 학습률 적용 |
| 기울기 리셋 | `p.grad = 0` | `p->grad = 0.0` | 다음 스텝 오염 방지 |
| 진행 출력 | `print(..., end='\r')` | `printf("\rstep..."); fflush(stdout)` | 같은 줄 덮어쓰기 |
| 그래프 해제 | (Python GC 자동) | `graph.clear()` | 임시 노드 메모리 해제 |
| 기대 초기 loss | ≈ 3.4 | ≈ 3.4 | ln(27) ≈ 3.30에 근접 |
| 기대 최종 loss | ≈ 2.0 | ≈ 2.0 | 이름 패턴 학습 완료 |

다음 챕터에서는 학습된 모델로 새로운 이름을 생성하는 추론 코드를 작성합니다.


> **직접 체험하기** — 시각화 도구에서 Loss 곡선, 학습률 스케줄, 어텐션 스냅샷을 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#training)

---
[< 이전: Ch15: GPT 순전파](ch15-gpt-forward.md) | [목차](../README.md) | [다음: Ch17: 추론 >](ch17-inference.md)
