# Ch11: 역전파 구현

## 학습 목표

- Python의 `backward()` 메서드가 C++에서 어떻게 구현되는지 이해한다
- 위상 정렬(topological sort)의 의미와 DFS 기반 구현을 직접 작성한다
- `std::unordered_set`, `std::function`, 재귀 람다를 처음으로 사용해 본다
- 체인룰이 역방향 순회에서 어떻게 적용되는지 코드 레벨로 확인한다
- 수치 그래디언트(finite differences)로 역전파 결과를 검증한다

---

## Python 코드 복습 (microgpt.py 59~72줄)

```python
def backward(self):
    topo = []          # 위상 정렬 결과를 담을 리스트
    visited = set()    # 방문한 노드를 기록하는 집합

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)   # 자식들을 먼저 방문 (DFS)
            topo.append(v)          # 자식 다 방문 후 자신을 추가

    build_topo(self)

    self.grad = 1          # 손실의 자기 자신에 대한 미분 = 1
    for v in reversed(topo):               # 역방향 순회
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad   # 체인룰
```

두 개의 핵심 단계로 나눌 수 있습니다.

1. **위상 정렬:** 계산 그래프를 DFS로 순회하여 "자식이 부모보다 앞에 오는" 순서로 노드를 정렬한다.
2. **역방향 누적:** 위상 정렬의 역순으로 순회하면서 체인룰로 기울기를 전파한다.

---

## 위상 정렬이란?

계산 그래프를 생각해 보세요.

```
z = x * y + x

그래프:
  x ──┬──► mul ──► add ──► z
  y ──┘          ↑
  x ─────────────┘
```

역전파할 때 `add`의 기울기가 `mul`보다 먼저 계산되어야 합니다. `mul`의 기울기를 계산할 때 `add`의 기울기(`add.grad`)가 필요하기 때문입니다.

**위상 정렬**은 이 의존성 순서를 보장합니다. DFS로 그래프를 순회할 때, 어떤 노드의 **모든 자식을 방문한 후에** 그 노드를 리스트에 추가합니다. 그러면 리스트의 순서가 "잎 노드(leaf) → 루트(root)"가 됩니다.

```
DFS 순회 결과 (topo):   [x, y, mul, x, add]  (자식이 부모보다 앞)
역방향 순회 (rbegin):   [add, mul, x, y, x]  (부모에서 자식으로)
```

역방향으로 순회할 때, 항상 "부모의 기울기가 계산된 후"에 자식의 기울기를 계산하므로 체인룰이 올바르게 적용됩니다.

---

## C++ 구현 — 한 줄씩

### 1단계: 데이터 구조 선언

```cpp
// microgpt.cpp 130~131줄
std::vector<Value*> topo;               // 위상 정렬 결과
std::unordered_set<Value*> visited;     // 방문한 포인터를 저장하는 집합
```

**`std::vector<Value*> topo`**

Python의 `topo = []`에 해당합니다. `Value*` 포인터들의 동적 배열입니다.

**`std::unordered_set<Value*> visited`**

Python의 `visited = set()`에 해당합니다.

```
Python set:              C++ unordered_set:
visited = set()          std::unordered_set<Value*> visited;
visited.add(v)           visited.insert(v);
v in visited             visited.count(v)  → 0이면 없음, 1이면 있음
```

`unordered_set`을 쓰는 이유: 원소가 있는지 확인하는 연산(`.count()`)이 평균 O(1)로 매우 빠릅니다. 내부적으로 해시 테이블을 사용합니다.

**포인터를 키로 쓰는 이유:** Python의 `set`은 객체의 동일성(`is`)을 판별할 수 있습니다. C++에서도 같은 효과를 내려면 **포인터(메모리 주소)** 를 키로 씁니다. 같은 `Value` 객체는 항상 같은 주소를 가지므로, 포인터 비교로 동일성을 확인합니다.

---

### 2단계: 재귀 람다로 DFS 구현

```cpp
// microgpt.cpp 132~138줄
std::function<void(Value*)> build_topo = [&](Value* v) {
    if (visited.count(v)) return;   // 이미 방문한 노드면 무시
    visited.insert(v);              // 방문 표시
    for (auto* child : v->children)
        build_topo(child);          // 자식들을 먼저 재귀 방문
    topo.push_back(v);              // 자식 다 방문 후 자신을 추가
};
build_topo(root);
```

Python에서는 함수 안에 함수를 정의할 수 있습니다 (`def build_topo(v):`). C++에서도 비슷한 일을 **람다(lambda)** 로 할 수 있습니다.

**람다 문법 분해:**

```cpp
std::function<void(Value*)> build_topo = [&](Value* v) {
//  ↑ 람다를 담는 변수 타입    ↑ 캡처  ↑ 매개변수
    // ... 함수 본체
};
```

- `std::function<void(Value*)>` — "Value* 하나를 받고 아무것도 반환하지 않는 함수"를 담는 타입
- `[&]` — 캡처(capture): 주변 변수들을 참조(`&`)로 캡처합니다. `topo`, `visited`, `build_topo` 자신까지 모두 접근 가능해집니다.
- `(Value* v)` — 람다의 매개변수

**왜 `std::function`이 필요한가?**

람다가 자기 자신(`build_topo`)을 재귀 호출하려면 먼저 변수에 이름이 있어야 합니다. `std::function`으로 선언하면 `build_topo`라는 이름으로 람다를 캡처해서 재귀 호출이 가능합니다.

```cpp
// [&]로 캡처했기 때문에 build_topo 변수를 람다 안에서 참조 가능
for (auto* child : v->children)
    build_topo(child);   // 재귀 호출
```

**`auto*`의 의미:**

```cpp
for (auto* child : v->children)
```

`auto*`는 컴파일러가 타입을 자동으로 추론하게 합니다. `v->children`은 `std::vector<Value*>`이므로 각 원소는 `Value*`입니다. 즉 `auto*`는 `Value*`로 추론됩니다.

---

### 3단계: 역방향 누적 (체인룰)

```cpp
// microgpt.cpp 142~147줄
root->grad = 1.0;
for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    Value* v = *it;
    for (size_t i = 0; i < v->children.size(); i++)
        v->children[i]->grad += v->local_grads[i] * v->grad;
}
```

**`root->grad = 1.0`**

손실 함수(Loss) L 의 자기 자신에 대한 미분은 1입니다. `dL/dL = 1`. 이것이 역전파의 시작점입니다.

```
Python:              C++:
self.grad = 1        root->grad = 1.0;
```

**`topo.rbegin()` / `topo.rend()`**

`rbegin()`은 역방향 반복자(reverse iterator)의 시작, `rend()`는 끝입니다. `topo`를 거꾸로 순회합니다.

```
topo:            [x, y, mul, add]   (자식 → 루트 순서)
역방향 순회:     [add, mul, y, x]   (루트 → 자식 순서)

Python:
for v in reversed(topo):

C++:
for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    Value* v = *it;   // *it로 실제 포인터를 꺼냄
```

**체인룰 적용:**

```cpp
v->children[i]->grad += v->local_grads[i] * v->grad;
```

```
Python:
child.grad += local_grad * v.grad

C++ 대응:
v->children[i]->grad  ← child.grad
v->local_grads[i]     ← local_grad
v->grad               ← v.grad
```

체인룰: `d(Loss)/d(child) += d(v)/d(child) * d(Loss)/d(v)`

`+=`을 쓰는 이유: 하나의 노드가 계산 그래프에서 여러 곳에 쓰일 수 있습니다. 예를 들어 `z = x*x`에서 `x`는 두 곳에 사용됩니다. 각 경로에서 오는 기울기를 모두 더해야 합니다.

---

### 전체 `backward` 함수

```cpp
// microgpt.cpp 128~148줄 전체
void backward(Value* root) {
    // 1단계: 위상 정렬
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;
    std::function<void(Value*)> build_topo = [&](Value* v) {
        if (visited.count(v)) return;
        visited.insert(v);
        for (auto* child : v->children)
            build_topo(child);
        topo.push_back(v);
    };
    build_topo(root);

    // 2단계: 역방향 누적
    root->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Value* v = *it;
        for (size_t i = 0; i < v->children.size(); i++)
            v->children[i]->grad += v->local_grads[i] * v->grad;
    }
}
```

---

## 수치 그래디언트로 검증하기

역전파가 올바른지 확인하는 황금 기준이 있습니다: **유한 차분법(finite differences)** 입니다.

미분의 정의에서 출발합니다:

```
f'(x) = lim(h→0) [f(x+h) - f(x-h)] / (2h)
```

h를 아주 작은 값(예: 1e-5)으로 고정하면, 이 공식으로 수치적으로 기울기를 근사할 수 있습니다.

### 예시: f(x) = x² 에서 x = 3.0

**해석적 기울기 (역전파):**
- `f(x) = x²`
- `f'(x) = 2x`
- `f'(3) = 6.0`

**수치적 기울기 (유한 차분):**

```
h = 1e-5
f(3 + h) = (3.000010)² = 9.000060000100
f(3 - h) = (2.999990)² = 8.999940000100
[f(3+h) - f(3-h)] / (2h) = (9.000060... - 8.999940...) / 0.000020
                          ≈ 6.000000...
```

두 값이 거의 일치하면 역전파가 올바르게 구현된 것입니다.

---

## 컴파일 가능한 전체 테스트 프로그램

```cpp
// ch11_test.cpp — 역전파 + 수치 검증
//
// 컴파일:
//   g++ -std=c++17 -O2 -o ch11_test ch11_test.cpp
// 실행:
//   ./ch11_test

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <vector>

// ---- Value, Graph (Ch10과 동일) ----
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
inline Value* log(Value* a) {
    return graph.make(std::log(a->data), {a}, {1.0 / a->data});
}
inline Value* exp(Value* a) {
    double e = std::exp(a->data);
    return graph.make(e, {a}, {e});
}
inline Value* relu(Value* a) {
    return graph.make(std::max(0.0, a->data), {a}, {a->data > 0 ? 1.0 : 0.0});
}
inline Value* neg(Value* a)           { return scale(a, -1.0); }
inline Value* sub(Value* a, Value* b) { return add(a, neg(b)); }
inline Value* div(Value* a, Value* b) { return mul(a, power(b, -1.0)); }
inline Value* sub_const(Value* a, double c) {
    return graph.make(a->data - c, {a}, {1.0});
}

// ---- 역전파 ----
void backward(Value* root) {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;
    std::function<void(Value*)> build_topo = [&](Value* v) {
        if (visited.count(v)) return;
        visited.insert(v);
        for (auto* child : v->children)
            build_topo(child);
        topo.push_back(v);
    };
    build_topo(root);
    root->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Value* v = *it;
        for (size_t i = 0; i < v->children.size(); i++)
            v->children[i]->grad += v->local_grads[i] * v->grad;
    }
}

bool approx(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

// ---- 테스트 1: f(x) = x^2, x=3 ----
// 해석적: f'(3) = 2*3 = 6
// 수치적: (f(3+h) - f(3-h)) / (2h) ≈ 6
void test_square() {
    std::cout << "--- 테스트 1: f(x) = x^2, x=3 ---\n";

    // 해석적 역전파
    Value* x = graph.make(3.0);
    Value* z = mul(x, x);   // z = x * x
    backward(z);
    double analytical = x->grad;  // 기대값: 6.0
    graph.clear();

    // 수치적 기울기 (중앙 차분)
    double h = 1e-5;
    double x0 = 3.0;
    double numerical = ((x0+h)*(x0+h) - (x0-h)*(x0-h)) / (2*h);

    std::cout << "  해석적 기울기: " << analytical << "  (기대: 6.0)\n";
    std::cout << "  수치적 기울기: " << numerical  << "  (기대: 6.0)\n";
    std::cout << "  차이: " << std::abs(analytical - numerical) << "\n";

    assert(approx(analytical, 6.0));
    assert(approx(numerical,  6.0));
    assert(approx(analytical, numerical));
    std::cout << "  통과!\n\n";
}

// ---- 테스트 2: f(x) = x^3 + 2x, x=2 ----
// 해석적: f'(x) = 3x^2 + 2 → f'(2) = 12 + 2 = 14
// 수치적으로 검증
void test_poly() {
    std::cout << "--- 테스트 2: f(x) = x^3 + 2x, x=2 ---\n";

    // 해석적 역전파
    Value* x = graph.make(2.0);
    Value* x3 = mul(mul(x, x), x);          // x^3
    Value* two_x = scale(x, 2.0);           // 2x
    Value* z = add(x3, two_x);              // x^3 + 2x
    backward(z);
    double analytical = x->grad;  // 기대값: 14.0
    graph.clear();

    // 수치적 검증
    auto f = [](double xv) { return xv*xv*xv + 2*xv; };
    double h = 1e-5;
    double numerical = (f(2.0+h) - f(2.0-h)) / (2*h);

    std::cout << "  해석적 기울기: " << analytical << "  (기대: 14.0)\n";
    std::cout << "  수치적 기울기: " << numerical  << "  (기대: 14.0)\n";
    std::cout << "  차이: " << std::abs(analytical - numerical) << "\n";

    assert(approx(analytical, 14.0));
    assert(approx(analytical, numerical));
    std::cout << "  통과!\n\n";
}

// ---- 테스트 3: f(x, y) = x*y + log(x), x=2, y=3 ----
// df/dx = y + 1/x = 3 + 0.5 = 3.5
// df/dy = x = 2.0
void test_multi_var() {
    std::cout << "--- 테스트 3: f(x,y) = x*y + log(x), x=2, y=3 ---\n";

    Value* x = graph.make(2.0);
    Value* y = graph.make(3.0);
    Value* xy = mul(x, y);
    Value* lx = log(x);
    Value* z = add(xy, lx);
    backward(z);

    std::cout << "  df/dx = " << x->grad << "  (기대: 3.5)\n";
    std::cout << "  df/dy = " << y->grad << "  (기대: 2.0)\n";

    assert(approx(x->grad, 3.5));
    assert(approx(y->grad, 2.0));
    graph.clear();
    std::cout << "  통과!\n\n";
}

// ---- 테스트 4: f(x) = relu(x), x=-1과 x=2 ----
// x=-1: f(x)=0, f'(x)=0
// x= 2: f(x)=2, f'(x)=1
void test_relu_grad() {
    std::cout << "--- 테스트 4: ReLU 기울기 ---\n";

    Value* a = graph.make(-1.0);
    Value* ra = relu(a);
    backward(ra);
    std::cout << "  relu(-1): 값=" << ra->data << " 기울기=" << a->grad << "  (기대: 0, 0)\n";
    assert(approx(ra->data, 0.0));
    assert(approx(a->grad,  0.0));
    graph.clear();

    Value* b = graph.make(2.0);
    Value* rb = relu(b);
    backward(rb);
    std::cout << "  relu( 2): 값=" << rb->data << " 기울기=" << b->grad << "  (기대: 2, 1)\n";
    assert(approx(rb->data, 2.0));
    assert(approx(b->grad,  1.0));
    graph.clear();

    std::cout << "  통과!\n\n";
}

// ---- 테스트 5: 동일 노드가 여러 곳에 쓰일 때 기울기 누적 ----
// f(x) = x * x + x, x=3
// df/dx = 2x + 1 = 7
void test_reuse() {
    std::cout << "--- 테스트 5: 노드 재사용 (기울기 누적) ---\n";

    Value* x = graph.make(3.0);
    Value* xx = mul(x, x);   // x가 두 번 사용
    Value* z = add(xx, x);   // x가 세 번째 사용
    backward(z);

    std::cout << "  f(x) = x*x + x, x=3\n";
    std::cout << "  df/dx = " << x->grad << "  (기대: 7.0)\n";
    assert(approx(x->grad, 7.0));
    graph.clear();
    std::cout << "  통과!\n\n";
}

int main() {
    std::cout << "=== Ch11 역전파 테스트 ===\n\n";
    test_square();
    test_poly();
    test_multi_var();
    test_relu_grad();
    test_reuse();
    std::cout << "모든 역전파 테스트 통과!\n";
    return 0;
}
```

### 컴파일과 실행

```bash
g++ -std=c++17 -O2 -o ch11_test ch11_test.cpp
./ch11_test
```

예상 출력:

```
=== Ch11 역전파 테스트 ===

--- 테스트 1: f(x) = x^2, x=3 ---
  해석적 기울기: 6  (기대: 6.0)
  수치적 기울기: 6  (기대: 6.0)
  차이: 2.98023e-11
  통과!

--- 테스트 2: f(x) = x^3 + 2x, x=2 ---
  해석적 기울기: 14  (기대: 14.0)
  수치적 기울기: 14  (기대: 14.0)
  차이: 4.73695e-10
  통과!

--- 테스트 3: f(x,y) = x*y + log(x), x=2, y=3 ---
  df/dx = 3.5  (기대: 3.5)
  df/dy = 2  (기대: 2.0)
  통과!

--- 테스트 4: ReLU 기울기 ---
  relu(-1): 값=0 기울기=0  (기대: 0, 0)
  relu( 2): 값=2 기울기=1  (기대: 2, 1)
  통과!

--- 테스트 5: 노드 재사용 (기울기 누적) ---
  f(x) = x*x + x, x=3
  df/dx = 7  (기대: 7.0)
  통과!

모든 역전파 테스트 통과!
```

---

## 핵심 정리

| 개념 | Python | C++ |
|------|--------|-----|
| 위상 정렬 결과 저장 | `topo = []` | `std::vector<Value*> topo` |
| 방문 체크 | `visited = set()` | `std::unordered_set<Value*> visited` |
| 집합에 추가 | `visited.add(v)` | `visited.insert(v)` |
| 집합에 있는지 확인 | `v in visited` | `visited.count(v)` |
| 중첩 함수 | `def build_topo(v):` | `std::function<void(Value*)> build_topo = [&](Value* v) {...};` |
| 역방향 순회 | `reversed(topo)` | `topo.rbegin()` ~ `topo.rend()` |
| 역방향 반복자 역참조 | (자동) | `*it` |

역전파의 두 단계:

```
1. 위상 정렬 (DFS):
   build_topo(root)
   → topo = [leaf_nodes, ..., root]  (자식이 부모보다 앞)

2. 역방향 누적 (체인룰):
   root->grad = 1.0
   for v in reversed(topo):
       for each child:
           child.grad += local_grad * v.grad
```

다음 챕터에서는 이 자동 미분 엔진 위에 `linear`, `softmax`, `rmsnorm` 같은 **신경망 빌딩 블록**을 구현합니다.


> **직접 체험하기** — 시각화 도구에서 역전파의 그래디언트 흐름을 단계별로 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#backprop)

---
[< 이전: Ch10: 미분 가능한 연산](ch10-differentiable-ops.md) | [목차](../README.md) | [다음: Ch12: 신경망 빌딩 블록 >](ch12-nn-building-blocks.md)
