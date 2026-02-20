# Ch12: 신경망 빌딩 블록

## 학습 목표

- `Vec`과 `Mat` 타입 별칭이 코드를 어떻게 단순화하는지 이해한다
- Python의 리스트 컴프리헨션을 C++ for 루프로 변환하는 패턴을 익힌다
- `linear()`, `softmax()`, `rmsnorm()` 세 함수를 한 줄씩 직접 작성한다
- 각 함수의 수치적 의미와 역할을 파악한다
- 컴파일 가능한 테스트로 각 함수의 출력을 직접 확인한다

---

## Vec과 Mat — 타입 별칭

C++은 `using`으로 기존 타입에 새로운 이름을 붙일 수 있습니다.

```cpp
// microgpt.cpp 154~155줄
using Vec = std::vector<Value*>;  // 1차원: 임베딩 벡터 하나
using Mat = std::vector<Vec>;     // 2차원: 가중치 행렬
```

왜 쓰는가:

```cpp
// 별칭 없이 — 길고 읽기 어려움
std::vector<Value*> linear(const std::vector<Value*>& x,
                            const std::vector<std::vector<Value*>>& w);

// 별칭 사용 — 간결하고 의미가 명확
Vec linear(const Vec& x, const Mat& w);
```

`using`은 새로운 타입을 만드는 것이 아닙니다. `Vec`과 `std::vector<Value*>`는 완전히 동일한 타입입니다. 코드를 읽기 쉽게 만드는 이름표입니다.

```
Vec의 실제 구조:
┌─────────────────────────────────┐
│ Vec (= std::vector<Value*>)     │
│  [Value*, Value*, Value*, ...]  │
│     ↓       ↓       ↓           │
│  {2.0}   {-1.3}  {0.5}  ...    │
└─────────────────────────────────┘

Mat의 실제 구조:
┌─────────────────────────────────┐
│ Mat (= std::vector<Vec>)        │
│  [Vec,    Vec,    Vec,   ...]   │
│   ↓        ↓       ↓            │
│  [행0] [행1]  [행2]  ...        │
└─────────────────────────────────┘
```

---

## `linear()` — 선형 변환

### 수학적 의미

선형 변환은 행렬-벡터 곱입니다.

```
출력 y = W @ x

y[0] = W[0][0]*x[0] + W[0][1]*x[1] + W[0][2]*x[2]  (W의 0번 행과 x의 내적)
y[1] = W[1][0]*x[0] + W[1][1]*x[1] + W[1][2]*x[2]  (W의 1번 행과 x의 내적)
...

W: (출력 차원) × (입력 차원) 행렬
x: (입력 차원) 벡터
y: (출력 차원) 벡터
```

### Python vs C++

```python
# microgpt.py 94~95줄
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    # 해석: w의 각 행(wo)에 대해, wo와 x의 내적을 계산
```

Python의 리스트 컴프리헨션과 `sum()`을 C++ for 루프로 변환합니다.

```cpp
// microgpt.cpp 163~173줄
Vec linear(const Vec& x, const Mat& w) {
    Vec out;
    out.reserve(w.size());              // ①: 메모리 미리 할당
    for (const auto& row : w) {         // ②: w의 각 행을 순회
        Value* s = graph.make(0.0);     // ③: 내적의 합산 시작값 0
        for (size_t i = 0; i < x.size(); i++)  // ④: 내적 계산
            s = add(s, mul(row[i], x[i]));
        out.push_back(s);               // ⑤: 결과를 출력 벡터에 추가
    }
    return out;
}
```

각 줄 설명:

**① `out.reserve(w.size())`**

`reserve`는 벡터의 내부 메모리를 미리 확보합니다. `push_back`을 반복할 때 메모리 재할당이 일어나는 것을 방지합니다. 속도 최적화입니다. 기능상 없어도 되지만, 좋은 습관입니다.

**② `for (const auto& row : w)`**

`w`의 각 행을 `row`로 순회합니다. `const auto&`는 "복사하지 말고 참조로 읽기만 하겠다"는 뜻입니다. `Vec`(= `std::vector<Value*>`)는 복사 비용이 있으므로 참조(`&`)를 씁니다.

```
Python:   for wo in w:
C++:      for (const auto& row : w) {
```

**③ `Value* s = graph.make(0.0)`**

내적(dot product)의 누적 합계를 저장할 `Value` 노드입니다. 0으로 시작합니다.

```
Python:   sum(wi * xi for wi, xi in zip(wo, x))
           ↑ sum()은 내부적으로 0에서 시작해 누적

C++:      s = 0.0 → add(s, mul(row[0], x[0])) → add(s, mul(row[1], x[1])) → ...
```

**④ `s = add(s, mul(row[i], x[i]))`**

`row[i] * x[i]`를 계산해서 누적 합에 더합니다. 이 연산들이 모두 계산 그래프에 기록되므로 역전파가 자동으로 됩니다.

```
Python:   sum(wi * xi for wi, xi in zip(wo, x))
C++:      for (...) s = add(s, mul(row[i], x[i]))
```

### 직관

```
W = [[w00, w01, w02],   x = [x0, x1, x2]
     [w10, w11, w12]]

linear(x, W):
  출력[0] = w00*x0 + w01*x1 + w02*x2   ← W의 0번 행과 x의 내적
  출력[1] = w10*x0 + w11*x1 + w12*x2   ← W의 1번 행과 x의 내적
```

---

## `softmax()` — 확률 분포 변환

### 수학적 의미

softmax는 임의의 숫자 목록(로짓)을 합이 1인 확률 분포로 변환합니다.

```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**왜 `max`를 빼는가?** 수치 안정성 때문입니다.

```
x = [1000, 999, 998]
exp(1000) = 무한대!  → 오버플로 오류

x - max(x) = [0, -1, -2]
exp(0) = 1
exp(-1) = 0.368
exp(-2) = 0.135  → 안전한 범위
```

`max`를 빼도 softmax의 결과는 수학적으로 동일합니다(분자 분모 모두 같은 값으로 나뉘므로).

### Python vs C++

```python
# microgpt.py 97~101줄
def softmax(logits):
    max_val = max(val.data for val in logits)          # ①: 최댓값 (data 비교)
    exps = [(val - max_val).exp() for val in logits]   # ②: exp(val - max)
    total = sum(exps)                                   # ③: 합산
    return [e / total for e in exps]                   # ④: 정규화
```

```cpp
// microgpt.cpp 178~200줄
Vec softmax(const Vec& logits) {
    // ①: 수치 안정성을 위한 최댓값 계산 (data 값을 비교)
    double max_val = -1e30;
    for (auto* v : logits)
        max_val = std::max(max_val, v->data);

    // ②: exp(logit - max) 계산 및 합산
    Vec exps;
    exps.reserve(logits.size());
    Value* total = graph.make(0.0);
    for (auto* v : logits) {
        Value* e = exp(sub_const(v, max_val));   // exp(v - max_val)
        exps.push_back(e);
        total = add(total, e);
    }

    // ③: 각 exp를 합계로 나누어 확률 산출
    Vec probs;
    probs.reserve(logits.size());
    for (auto* e : exps)
        probs.push_back(div(e, total));
    return probs;
}
```

각 줄 설명:

**① 최댓값 계산**

```cpp
double max_val = -1e30;    // 매우 작은 초기값 (어떤 값이든 이보다 크므로)
for (auto* v : logits)
    max_val = std::max(max_val, v->data);
```

`v->data`를 비교합니다. `Value*` 포인터가 아니라 그 안의 실제 숫자를 비교합니다. `max_val`은 `double`이므로 계산 그래프에 기록되지 않습니다 — 이것은 의도적입니다. 최댓값을 빼는 것은 수치 안정화 트릭이므로 역전파에 참여시킬 필요가 없습니다.

```
Python:   max_val = max(val.data for val in logits)
C++:      for (auto* v : logits) max_val = std::max(max_val, v->data);
```

**② `sub_const(v, max_val)`**

Ch10에서 배운 `sub_const`를 씁니다. 상수 `max_val`을 빼는 전용 함수입니다. `sub(v, graph.make(max_val))`와 달리, `max_val`을 위한 `Value*` 노드를 만들지 않으므로 효율적입니다.

```
Python:   (val - max_val).exp()    → val.__sub__(max_val) → Value(max_val).__exp__()
C++:      exp(sub_const(v, max_val))  → 상수를 위한 노드 없음
```

**③ `total = add(total, e)`**

누적 합계도 `Value*`로 관리합니다. `softmax`가 역전파에 참여하기 때문에 `total`도 계산 그래프의 일부여야 합니다.

```
Python:   total = sum(exps)   → Python sum()은 0에서 시작해 __add__ 반복
C++:      total = graph.make(0.0)
          total = add(total, e0)
          total = add(total, e1)
          ...
```

**④ `div(e, total)`**

Ch10의 `div` 함수로 각 `exp` 값을 `total`로 나눕니다.

```
Python:   [e / total for e in exps]  → Value.__truediv__
C++:      probs.push_back(div(e, total))
```

---

## `rmsnorm()` — RMS 정규화

### 수학적 의미

RMSNorm은 벡터를 그 원소들의 제곱 평균 제곱근(RMS)으로 나눕니다.

```
rms(x) = sqrt(mean(x²) + ε)   (ε은 분모가 0이 되는 것을 막는 작은 상수)

rmsnorm(x)_i = x_i / rms(x)
```

LayerNorm과 비교:
- LayerNorm: 평균을 빼고 표준편차로 나눔
- RMSNorm: 평균을 빼지 않고 RMS로만 나눔 (더 단순, GPT-2 이후 많이 사용)

### Python vs C++

```python
# microgpt.py 103~106줄
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)    # ①: 제곱의 평균
    scale = (ms + 1e-5) ** -0.5               # ②: 역제곱근 스케일 팩터
    return [xi * scale for xi in x]            # ③: 각 원소에 스케일 적용
```

```cpp
// microgpt.cpp 205~221줄
Vec rmsnorm(const Vec& x) {
    // ①: 제곱의 평균 (mean of squares)
    Value* ms = graph.make(0.0);
    for (auto* xi : x)
        ms = add(ms, mul(xi, xi));
    ms = scale(ms, 1.0 / static_cast<double>(x.size()));

    // ②: 스케일 팩터: 1 / sqrt(ms + eps) = (ms + eps)^(-0.5)
    Value* s = power(add(ms, graph.make(1e-5)), -0.5);

    // ③: 각 원소에 스케일 적용
    Vec out;
    out.reserve(x.size());
    for (auto* xi : x)
        out.push_back(mul(xi, s));
    return out;
}
```

각 줄 설명:

**① 제곱의 평균**

```cpp
Value* ms = graph.make(0.0);
for (auto* xi : x)
    ms = add(ms, mul(xi, xi));           // ms += xi * xi
ms = scale(ms, 1.0 / static_cast<double>(x.size()));
```

`mul(xi, xi)` — `xi`의 제곱입니다.
`scale(ms, 1.0 / ...)` — 합계를 개수로 나누어 평균을 구합니다.

```
Python:   sum(xi * xi for xi in x) / len(x)
          → xi * xi 는 Value.__mul__, / len(x)는 Value.__truediv__

C++:      for (...) ms = add(ms, mul(xi, xi));
          ms = scale(ms, 1.0 / static_cast<double>(x.size()));
```

**`static_cast<double>(x.size())`**

`x.size()`는 `size_t`(부호 없는 정수) 타입입니다. `1.0 / size_t`는 정수 나눗셈이 되어 잘못된 결과를 낼 수 있으므로, `static_cast<double>`로 명시적으로 실수로 변환합니다.

```cpp
// 잘못된 예 (정수 나눗셈)
ms = scale(ms, 1.0 / x.size());   // x.size()가 size_t라 문제 없긴 하지만 의도 불명확

// 명시적 변환 (권장)
ms = scale(ms, 1.0 / static_cast<double>(x.size()));
```

Python에서는 `/`가 항상 실수 나눗셈이므로 신경 쓸 필요가 없습니다.

**② 스케일 팩터**

```cpp
Value* s = power(add(ms, graph.make(1e-5)), -0.5);
```

단계별로 분해하면:

```
graph.make(1e-5)           → 1e-5 상수 노드
add(ms, 1e-5 노드)         → ms + ε
power(..., -0.5)           → (ms + ε)^(-0.5) = 1/sqrt(ms + ε)
```

```
Python:   (ms + 1e-5) ** -0.5
          → ms + 1e-5:  Value.__add__(1e-5) → Value(ms.data + 1e-5, ...)
          → ** -0.5:    Value.__pow__(-0.5)

C++:      power(add(ms, graph.make(1e-5)), -0.5)
```

**③ 각 원소에 스케일 적용**

```cpp
for (auto* xi : x)
    out.push_back(mul(xi, s));
```

같은 `s` 노드를 모든 원소에 공유합니다. 역전파 시 `s`의 기울기는 모든 경로에서 누적됩니다.

```
Python:   [xi * scale for xi in x]
C++:      for (auto* xi : x) out.push_back(mul(xi, s));
```

---

## 세 함수 한눈에 비교

```
linear(x, W):
  입력: 벡터 x (크기 n), 행렬 W (크기 m×n)
  출력: 벡터 y (크기 m)
  계산: y[i] = 내적(W의 i번째 행, x)

softmax(logits):
  입력: 벡터 logits (크기 n, 임의의 실수)
  출력: 벡터 probs  (크기 n, 합이 1인 확률)
  계산: probs[i] = exp(logits[i] - max) / sum(exp(logits[j] - max))

rmsnorm(x):
  입력: 벡터 x (크기 n)
  출력: 벡터 y (크기 n, 정규화됨)
  계산: y[i] = x[i] / sqrt(mean(x^2) + ε)
```

---

## 컴파일 가능한 전체 테스트 프로그램

```cpp
// ch12_test.cpp — 신경망 빌딩 블록 테스트
//
// 컴파일:
//   g++ -std=c++17 -O2 -o ch12_test ch12_test.cpp
// 실행:
//   ./ch12_test

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <vector>

// ---- Value, Graph ----
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};
struct Graph {
    std::vector<Value*> nodes;
    Value* make(double data, std::vector<Value*> ch = {},
                std::vector<double> lg = {}) {
        auto* v = new Value{data, 0.0, std::move(ch), std::move(lg)};
        nodes.push_back(v);
        return v;
    }
    void clear() { for (auto* v : nodes) delete v; nodes.clear(); }
    ~Graph() { clear(); }
};
static Graph graph;

// ---- 연산 ----
inline Value* add(Value* a, Value* b) {
    return graph.make(a->data + b->data, {a, b}, {1.0, 1.0});
}
inline Value* mul(Value* a, Value* b) {
    return graph.make(a->data * b->data, {a, b}, {b->data, a->data});
}
inline Value* scale(Value* a, double s) {
    return graph.make(a->data * s, {a}, {s});
}
inline Value* power(Value* a, double n) {
    return graph.make(std::pow(a->data, n), {a}, {n * std::pow(a->data, n-1)});
}
inline Value* exp(Value* a) {
    double e = std::exp(a->data);
    return graph.make(e, {a}, {e});
}
inline Value* div(Value* a, Value* b) { return mul(a, power(b, -1.0)); }
inline Value* sub_const(Value* a, double c) {
    return graph.make(a->data - c, {a}, {1.0});
}

// ---- 타입 별칭 ----
using Vec = std::vector<Value*>;
using Mat = std::vector<Vec>;

// ---- 신경망 빌딩 블록 ----
Vec linear(const Vec& x, const Mat& w) {
    Vec out;
    out.reserve(w.size());
    for (const auto& row : w) {
        Value* s = graph.make(0.0);
        for (size_t i = 0; i < x.size(); i++)
            s = add(s, mul(row[i], x[i]));
        out.push_back(s);
    }
    return out;
}

Vec softmax(const Vec& logits) {
    double max_val = -1e30;
    for (auto* v : logits)
        max_val = std::max(max_val, v->data);
    Vec exps;
    exps.reserve(logits.size());
    Value* total = graph.make(0.0);
    for (auto* v : logits) {
        Value* e = exp(sub_const(v, max_val));
        exps.push_back(e);
        total = add(total, e);
    }
    Vec probs;
    probs.reserve(logits.size());
    for (auto* e : exps)
        probs.push_back(div(e, total));
    return probs;
}

Vec rmsnorm(const Vec& x) {
    Value* ms = graph.make(0.0);
    for (auto* xi : x)
        ms = add(ms, mul(xi, xi));
    ms = scale(ms, 1.0 / static_cast<double>(x.size()));
    Value* s = power(add(ms, graph.make(1e-5)), -0.5);
    Vec out;
    out.reserve(x.size());
    for (auto* xi : x)
        out.push_back(mul(xi, s));
    return out;
}

bool approx(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// ---- 테스트 ----

void test_linear() {
    std::cout << "--- linear() 테스트 ---\n";
    // 가중치 W (2x3):
    //   행0: [1, 0, 0]  → 입력[0]만 통과
    //   행1: [0, 1, 0]  → 입력[1]만 통과
    // 입력 x: [5, 7, 3]
    // 기대 출력: [1*5+0*7+0*3, 0*5+1*7+0*3] = [5, 7]
    Mat w = {
        {graph.make(1.0), graph.make(0.0), graph.make(0.0)},
        {graph.make(0.0), graph.make(1.0), graph.make(0.0)}
    };
    Vec x = {graph.make(5.0), graph.make(7.0), graph.make(3.0)};

    Vec y = linear(x, w);

    std::cout << "  출력[0] = " << y[0]->data << "  (기대: 5.0)\n";
    std::cout << "  출력[1] = " << y[1]->data << "  (기대: 7.0)\n";
    assert(approx(y[0]->data, 5.0));
    assert(approx(y[1]->data, 7.0));
    graph.clear();

    // 일반 행렬 곱 테스트: W(3x2) @ x(2) = y(3)
    // W = [[1,2],[3,4],[5,6]], x = [1,1]
    // y = [1+2, 3+4, 5+6] = [3, 7, 11]
    Mat w2 = {
        {graph.make(1.0), graph.make(2.0)},
        {graph.make(3.0), graph.make(4.0)},
        {graph.make(5.0), graph.make(6.0)}
    };
    Vec x2 = {graph.make(1.0), graph.make(1.0)};
    Vec y2 = linear(x2, w2);

    std::cout << "  출력[0] = " << y2[0]->data << "  (기대: 3.0)\n";
    std::cout << "  출력[1] = " << y2[1]->data << "  (기대: 7.0)\n";
    std::cout << "  출력[2] = " << y2[2]->data << "  (기대: 11.0)\n";
    assert(approx(y2[0]->data,  3.0));
    assert(approx(y2[1]->data,  7.0));
    assert(approx(y2[2]->data, 11.0));
    graph.clear();

    std::cout << "  통과!\n\n";
}

void test_softmax() {
    std::cout << "--- softmax() 테스트 ---\n";

    // 입력: [2.0, 1.0, 0.5]
    // 출력: 합이 1이어야 함
    Vec logits = {graph.make(2.0), graph.make(1.0), graph.make(0.5)};
    Vec probs = softmax(logits);

    double sum_prob = 0.0;
    for (auto* p : probs) {
        std::cout << "  prob = " << p->data << "\n";
        assert(p->data >= 0.0 && p->data <= 1.0);  // 0 이상 1 이하
        sum_prob += p->data;
    }
    std::cout << "  합계 = " << sum_prob << "  (기대: 1.0)\n";
    assert(approx(sum_prob, 1.0));

    // 가장 큰 로짓(2.0)이 가장 높은 확률을 가져야 함
    assert(probs[0]->data > probs[1]->data);
    assert(probs[1]->data > probs[2]->data);
    graph.clear();

    // 수치 안정성 테스트: 큰 값에서도 오버플로 없이 동작
    Vec large_logits = {graph.make(1000.0), graph.make(999.0), graph.make(998.0)};
    Vec large_probs = softmax(large_logits);
    double large_sum = 0.0;
    for (auto* p : large_probs) large_sum += p->data;
    std::cout << "  큰 로짓 합계 = " << large_sum << "  (기대: 1.0)\n";
    assert(approx(large_sum, 1.0));
    graph.clear();

    std::cout << "  통과!\n\n";
}

void test_rmsnorm() {
    std::cout << "--- rmsnorm() 테스트 ---\n";

    // 입력: [3.0, 4.0]
    // mean(x^2) = (9 + 16) / 2 = 12.5
    // rms = sqrt(12.5 + 1e-5) ≈ 3.5355
    // 출력: [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
    Vec x = {graph.make(3.0), graph.make(4.0)};
    Vec y = rmsnorm(x);

    double ms = (9.0 + 16.0) / 2.0;
    double rms = std::sqrt(ms + 1e-5);
    double expected0 = 3.0 / rms;
    double expected1 = 4.0 / rms;

    std::cout << "  y[0] = " << y[0]->data << "  (기대: " << expected0 << ")\n";
    std::cout << "  y[1] = " << y[1]->data << "  (기대: " << expected1 << ")\n";
    assert(approx(y[0]->data, expected0, 1e-5));
    assert(approx(y[1]->data, expected1, 1e-5));
    graph.clear();

    // 이미 정규화된 벡터를 넣으면 크게 변하지 않아야 함
    // x = [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    // mean(x^2) = 1/3, rms = sqrt(1/3 + eps) ≈ 1/sqrt(3)
    // 출력 ≈ [1, 1, 1] * 1/sqrt(3) / (1/sqrt(3)) ≈ [1, 1, 1]
    double val = 1.0 / std::sqrt(3.0);
    Vec x2 = {graph.make(val), graph.make(val), graph.make(val)};
    Vec y2 = rmsnorm(x2);
    std::cout << "  균일 벡터 정규화: y2[0] = " << y2[0]->data
              << "  (기대: ≈1.0)\n";
    // 약간의 eps 오차가 있으므로 느슨한 허용치 사용
    assert(approx(y2[0]->data, 1.0, 1e-4));
    graph.clear();

    std::cout << "  통과!\n\n";
}

int main() {
    std::cout << "=== Ch12 신경망 빌딩 블록 테스트 ===\n\n";
    test_linear();
    test_softmax();
    test_rmsnorm();
    std::cout << "모든 테스트 통과!\n";
    return 0;
}
```

### 컴파일과 실행

```bash
g++ -std=c++17 -O2 -o ch12_test ch12_test.cpp
./ch12_test
```

예상 출력:

```
=== Ch12 신경망 빌딩 블록 테스트 ===

--- linear() 테스트 ---
  출력[0] = 5  (기대: 5.0)
  출력[1] = 7  (기대: 7.0)
  출력[0] = 3  (기대: 3.0)
  출력[1] = 7  (기대: 7.0)
  출력[2] = 11  (기대: 11.0)
  통과!

--- softmax() 테스트 ---
  prob = 0.576117
  prob = 0.211942
  prob = 0.211942   (참고: 실제 값은 달라질 수 있음)
  합계 = 1  (기대: 1.0)
  큰 로짓 합계 = 1  (기대: 1.0)
  통과!

--- rmsnorm() 테스트 ---
  y[0] = 0.848528  (기대: 0.848528)
  y[1] = 1.13137   (기대: 1.13137)
  균일 벡터 정규화: y2[0] = 1  (기대: ≈1.0)
  통과!

모든 테스트 통과!
```

---

## 핵심 정리

| 개념 | Python | C++ |
|------|--------|-----|
| 타입 별칭 | (없음) | `using Vec = std::vector<Value*>` |
| 리스트 컴프리헨션 | `[f(x) for x in xs]` | `for (auto* x : xs) out.push_back(f(x))` |
| 메모리 예약 | (자동) | `out.reserve(n)` |
| 정수→실수 변환 | (자동) | `static_cast<double>(n)` |
| 참조로 읽기 | (자동) | `const auto&` |

세 함수의 구조 패턴:

```cpp
Vec 함수이름(const Vec& 입력, ...) {
    Vec out;
    out.reserve(결과_크기);   // 메모리 예약

    for (/* 각 원소 순회 */) {
        // 중간 계산 (Value* 노드들)
        out.push_back(최종_노드);
    }

    return out;
}
```

다음 챕터에서는 이 신경망을 학습시킬 **데이터를 불러오고 토크나이저를 만드는** 방법을 구현합니다.


> **직접 체험하기** — 시각화 도구에서 linear, softmax, rmsnorm이 추론에서 어떻게 동작하는지 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#inference)

---
[< 이전: Ch11: 역전파 구현](ch11-backward.md) | [목차](../README.md) | [다음: Ch13: 데이터 로딩과 토크나이저 >](ch13-data-loading.md)
