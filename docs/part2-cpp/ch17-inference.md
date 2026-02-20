# Ch17: 추론

## 학습 목표

- Temperature가 무엇인지, 수치가 낮고 높을 때 어떻게 다른지 설명할 수 있다
- `std::discrete_distribution<int>`로 확률 가중 샘플링을 구현한다
- 자기회귀 생성 루프의 흐름(BOS → 글자 생성 → BOS로 종료)을 코드로 작성한다
- 추론 중 `graph.clear()`를 언제 호출해야 하는지 설명한다
- 메모리 정리(`delete p`)가 왜 필요하고 어디서 해야 하는지 이해한다

---

## 개념 설명

### 추론(Inference)이란?

학습이 끝난 모델에게 새 입력을 주어 출력을 생성하는 과정입니다. 학습 중과 다른 점은 **역전파와 파라미터 업데이트가 없다**는 것입니다. 오직 순전파만 합니다.

```
학습 단계:   순전파 → 손실 계산 → 역전파 → 파라미터 업데이트 (반복)
추론 단계:   순전파 → 샘플링 → 다음 토큰을 입력으로 (반복)
```

### 자기회귀(Autoregressive) 생성

GPT는 한 번에 토큰 하나씩 생성합니다. 이전에 생성한 토큰들이 다음 토큰 예측에 사용됩니다.

```
BOS → gpt() → P(다음 토큰) → 'a' 선택
'a' → gpt() → P(다음 토큰) → 'l' 선택
'l' → gpt() → P(다음 토큰) → 'y' 선택
'y' → gpt() → P(다음 토큰) → 's' 선택
's' → gpt() → P(다음 토큰) → 'a' 선택
'a' → gpt() → P(다음 토큰) → BOS 선택 ← 종료!

생성된 이름: "alysa"
```

### Temperature — 창의성과 안전성의 균형

Temperature는 모델이 얼마나 "과감하게" 샘플링하는지를 조절합니다. 로짓을 temperature로 나눕니다.

```
temperature < 1: 로짓을 크게 만듦 → 확률 분포가 날카로워짐 → 확실한 선택

    낮은 temperature (0.3):
    원래 확률: [a=0.5, b=0.3, c=0.2]
    로짓/0.3 후: [a=0.95, b=0.04, c=0.01]
    → 거의 항상 'a'를 선택. 보수적, 안전

temperature = 1: 변화 없음 → 학습된 확률 그대로 사용

    temperature=1:
    원래 확률 그대로: [a=0.5, b=0.3, c=0.2]

temperature > 1: 로짓을 작게 만듦 → 확률 분포가 평탄해짐 → 다양한 선택

    높은 temperature (2.0):
    원래 확률: [a=0.5, b=0.3, c=0.2]
    로짓/2.0 후: [a=0.38, b=0.33, c=0.29]
    → 'b', 'c'도 자주 선택됨. 창의적, 다양하지만 이상한 결과도 나옴
```

이 튜토리얼에서는 `temperature = 0.5`를 사용합니다. 이름 생성에 적합한 보수적인 값입니다.

### `std::discrete_distribution<int>` — 가중치 기반 샘플링

Python의 `random.choices(range(vocab_size), weights=[p.data for p in probs])`에 해당하는 C++ 도구입니다.

```cpp
std::vector<double> weights = {0.5, 0.3, 0.2};  // 확률 (합이 1일 필요 없음)
std::discrete_distribution<int> dist(weights.begin(), weights.end());
int sampled = dist(rng);
// 50% 확률로 0, 30% 확률로 1, 20% 확률로 2를 반환
```

`dist(rng)`를 호출할 때마다 weights에 비례하는 확률로 인덱스를 하나 선택합니다.

### 추론 중 graph.clear()의 올바른 위치

추론에서 KV 캐시를 사용한다는 점이 중요합니다. 각 위치의 gpt() 호출은 이전 위치에서 만들어진 KV 캐시(keys, vals)를 참조합니다.

```
샘플 생성 중:
  pos=0: gpt() → KV 캐시에 k0,v0 저장 → graph에 노드 생성
  pos=1: gpt() → KV 캐시에 k1,v1 저장 → graph에 노드 생성
         여기서 k0,v0를 참조! → graph를 비우면 안 됨!
  ...
  pos=last: BOS 토큰 생성 → 샘플 완성

graph.clear() ← 샘플 하나가 완전히 끝난 후에만 호출!
```

만약 중간에 `graph.clear()`를 호출하면 KV 캐시의 포인터들이 해제된 메모리를 가리키게 됩니다 (dangling pointer, undefined behavior).

---

## Python ↔ C++ 비교

### Temperature 적용

```python
# Python (microgpt.py:195)
probs = softmax([l / temperature for l in logits])
```

```cpp
// C++ (microgpt.cpp:490-494)
Vec scaled;
scaled.reserve(logits.size());
for (auto* l : logits)
    scaled.push_back(scale(l, 1.0 / temperature));
Vec probs = softmax(scaled);
```

Python의 리스트 컴프리헨션 `[l / temperature for l in logits]`를 C++에서는 for 루프로 작성합니다. `scale(l, 1.0/temperature)`는 `l * (1/temperature)`로 나누기와 동일합니다.

### 확률 가중 샘플링

```python
# Python (microgpt.py:196)
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

```cpp
// C++ (microgpt.cpp:497-501)
std::vector<double> weights;
weights.reserve(probs.size());
for (auto* p : probs) weights.push_back(p->data);
std::discrete_distribution<int> dist(weights.begin(), weights.end());
token_id = dist(rng);
```

| Python | C++ |
|--------|-----|
| `[p.data for p in probs]` | `for (auto* p : probs) weights.push_back(p->data)` |
| `random.choices(..., weights=...)[0]` | `discrete_distribution<int> dist(...); dist(rng)` |
| 한 줄 | 5줄 (더 명시적) |

### 자기회귀 생성 루프 전체

```python
# Python (microgpt.py:189-200)
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

```cpp
// C++ (microgpt.cpp:480-507)
for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
    graph.clear();  // 이전 샘플의 그래프 해제
    std::vector<std::vector<Vec>> keys(N_LAYER), vals(N_LAYER);
    int token_id = BOS;
    std::string sample;

    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
        Vec logits = gpt(token_id, pos, keys, vals, state_dict);

        Vec scaled;
        scaled.reserve(logits.size());
        for (auto* l : logits)
            scaled.push_back(scale(l, 1.0 / temperature));
        Vec probs = softmax(scaled);

        std::vector<double> weights;
        weights.reserve(probs.size());
        for (auto* p : probs) weights.push_back(p->data);
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        token_id = dist(rng);

        if (token_id == BOS) break;
        sample += uchars[token_id];
    }

    std::printf("sample %2d: %s\n", sample_idx + 1, sample.c_str());
}
```

---

## 코드 작성 — 한 줄씩

### 1단계: 상수 선언과 헤더 출력

```cpp
    // 학습 루프 끝난 직후

    constexpr double temperature = 0.5;
    std::cout << "\n--- inference (new, hallucinated names) ---\n";
```

`\n`은 줄 바꿈입니다. 학습 루프의 `\r` 출력 후 새 줄로 이동합니다.

### 2단계: 샘플 루프 시작

```cpp
    for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
```

20개의 이름을 생성합니다.

### 3단계: 각 샘플 초기화

```cpp
        graph.clear();  // 이전 샘플의 계산 그래프 해제
        std::vector<std::vector<Vec>> keys(N_LAYER), vals(N_LAYER);
        int token_id = BOS;    // BOS 토큰에서 생성 시작
        std::string sample;    // 생성된 글자를 모을 문자열
```

`graph.clear()`를 샘플 루프 맨 앞에서 호출합니다. 이 시점은 이전 샘플이 완전히 끝난 직후이므로 안전합니다. 새 `keys`, `vals`로 KV 캐시를 초기화합니다.

### 4단계: 토큰 생성 루프

```cpp
        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            // 현재 token_id로 순전파
            Vec logits = gpt(token_id, pos, keys, vals, state_dict);
```

`pos`가 `BLOCK_SIZE`에 도달하면 강제 종료됩니다. 이름이 너무 길어지는 것을 방지합니다. 보통은 BOS 토큰이 먼저 샘플링되어 훨씬 일찍 종료됩니다.

### 5단계: Temperature 적용

```cpp
            // 로짓에 temperature 스케일링 적용
            Vec scaled;
            scaled.reserve(logits.size());
            for (auto* l : logits)
                scaled.push_back(scale(l, 1.0 / temperature));
            // temperature=0.5 → 로짓을 2배로 → 확률 분포가 더 날카로워짐
```

`1.0 / temperature = 1.0 / 0.5 = 2.0`이므로, 모든 로짓을 2배로 키웁니다. softmax 후 확률 차이가 더 극명해집니다.

### 6단계: Softmax → 확률

```cpp
            Vec probs = softmax(scaled);
```

scaled 로짓을 확률 분포로 변환합니다. 이제 `probs[i]->data`가 i번 토큰이 선택될 확률입니다.

### 7단계: 확률 가중 샘플링

```cpp
            // Value* 포인터에서 double 값만 추출
            std::vector<double> weights;
            weights.reserve(probs.size());
            for (auto* p : probs)
                weights.push_back(p->data);

            // 가중 확률로 토큰 하나 샘플링
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            token_id = dist(rng);
```

`std::discrete_distribution`은 `<random>` 헤더에 있습니다. 생성자에 가중치 범위를 전달하면, `dist(rng)`를 호출할 때마다 가중치에 비례하는 확률로 인덱스를 반환합니다.

### 8단계: 종료 조건 확인 및 글자 추가

```cpp
            if (token_id == BOS) break;  // BOS 토큰 → 이름 끝
            sample += uchars[token_id];  // 글자를 문자열에 추가
        }
```

Python의 `sample.append(uchars[token_id])`와 달리, C++에서는 `std::string`에 `+=`으로 문자를 추가합니다. `uchars[token_id]`는 `char` 타입입니다.

### 9단계: 결과 출력

```cpp
        std::printf("sample %2d: %s\n", sample_idx + 1, sample.c_str());
    }
    // 샘플 루프 끝
```

`sample.c_str()`은 `std::string`을 C 스타일 null 종료 문자열(`const char*`)로 변환합니다. `printf`의 `%s` 포맷에 필요합니다.

### 10단계: 메모리 정리

```cpp
    // 모든 파라미터 노드 해제 (힙에 할당된 Value* 해제)
    graph.clear();
    for (auto* p : params)
        delete p;

    return 0;
}
```

프로그램이 끝날 때 운영체제가 메모리를 회수하므로 엄밀히는 필수가 아닙니다. 그러나 명시적으로 해제하는 것이 좋은 C++ 습관이고, 메모리 누수 탐지 도구(Valgrind 등)를 사용할 때 필요합니다.

---

## 기대 실행 결과

1000 스텝 학습 후 temperature=0.5에서의 샘플 출력 예시:

```
--- inference (new, hallucinated names) ---
sample  1: karis
sample  2: arden
sample  3: myla
sample  4: jorin
sample  5: naelan
sample  6: tayli
sample  7: brenna
sample  8: caelan
sample  9: emari
sample 10: ryla
sample 11: aeron
sample 12: maren
sample 13: karis
sample 14: jaylen
sample 15: nyla
sample 16: brayden
sample 17: alysa
sample 18: caelyn
sample 19: emilia
sample 20: rylan
```

이름처럼 들리는 글자 조합이 생성됩니다. 실제로 존재하는 이름도 있고, 이름처럼 들리지만 없는 이름도 있습니다. 이것이 "hallucinated names"입니다.

### 결과가 다를 수 있는 이유

`std::mt19937 rng(42)`로 시드를 고정했지만, 학습 데이터와 샘플링 모두 같은 rng 인스턴스를 공유합니다. 학습 루프에서 rng가 사용된 횟수에 따라 추론 시의 초기 상태가 달라질 수 있습니다. 또한 컴파일러나 OS에 따라 부동소수점 연산 결과가 미묘하게 다를 수 있습니다.

---

## 핵심 정리

| 개념 | Python | C++ | 의미 |
|------|--------|-----|------|
| Temperature 적용 | `l / temperature` | `scale(l, 1.0/temperature)` | 로짓 스케일링으로 분포 날카롭게/평탄하게 |
| 낮은 temperature | 보수적 (항상 높은 확률 선택) | 동일 | temperature=0.5 → 안정적 이름 |
| 높은 temperature | 창의적 (다양한 선택) | 동일 | temperature=1.5 → 독특하지만 이상한 이름 |
| 가중 샘플링 | `random.choices(..., weights=...)` | `discrete_distribution<int>` | 확률에 비례해서 토큰 선택 |
| weights 추출 | `[p.data for p in probs]` | `for (auto* p : probs) weights.push_back(p->data)` | Value* → double 변환 |
| 자기회귀 루프 | `for pos_id in range(block_size)` | `for (int pos = 0; pos < BLOCK_SIZE; pos++)` | 한 번에 토큰 하나씩 생성 |
| 종료 조건 | `if token_id == BOS: break` | `if (token_id == BOS) break` | BOS 토큰이 나오면 이름 완성 |
| 글자 추가 | `sample.append(uchars[token_id])` | `sample += uchars[token_id]` | char를 string에 추가 |
| graph.clear() 위치 | (자동 GC) | 각 샘플 시작 직전 | KV 캐시 보호 후 이전 그래프 해제 |
| 메모리 정리 | (자동 GC) | `for (auto* p : params) delete p` | 힙 할당 파라미터 해제 |

다음 챕터에서는 전체 코드를 CMake로 빌드하는 방법과, 이 프로젝트를 통해 배운 것들을 정리합니다.


> **직접 체험하기** — 시각화 도구에서 Temperature 조절과 실시간 GPT 추론을 직접 체험할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#inference)

---
[< 이전: Ch16: 학습 루프](ch16-training-loop.md) | [목차](../README.md) | [다음: Ch18: 빌드와 마무리 >](ch18-build-and-wrap-up.md)
