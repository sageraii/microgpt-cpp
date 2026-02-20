# Ch10: 미분 가능한 연산

## 학습 목표

- Python의 연산자 오버로딩(`__add__`, `__mul__` 등)이 C++에서 왜 다른 방식으로 구현되는지 이해한다
- C++ 자유 함수(free function)로 미분 가능한 연산을 하나씩 직접 작성한다
- 각 연산의 수학적 의미(순전파 값, 역전파 기울기)를 코드와 연결해서 이해한다
- `inline` 키워드의 역할을 파악한다
- 각 연산에 대한 단위 테스트를 작성하고 컴파일해서 검증한다

---

## Python 코드 복습

Part 1에서 배운 `Value` 클래스는 Python의 **연산자 오버로딩**을 이용했습니다. 파이썬 객체에 `__add__` 같은 특수 메서드를 정의하면 `a + b`라는 자연스러운 문법이 자동으로 그 메서드를 호출합니다.

```python
# microgpt.py 39~56줄 — Python의 연산자 오버로딩
class Value:
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
        #                                     자식들          국소 기울기

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # 복합 연산 — 위 기본 연산들의 조합
    def __neg__(self):   return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1
```

Python에서는 `a + b`를 쓰면 자동으로 `a.__add__(b)`가 호출됩니다. 매우 편리합니다.

---

## C++에서 왜 자유 함수를 쓰는가?

C++로 옮길 때 가장 먼저 부딪히는 질문이 있습니다: **"C++에도 연산자 오버로딩이 있는데, 그냥 `Value`에 `operator+`를 정의하면 되지 않나?"**

`microgpt.cpp`의 `Value`는 **포인터**로 다룹니다:

```cpp
Value* a = graph.make(2.0);
Value* b = graph.make(3.0);
```

`Value*`는 C++이 제공하는 내장 포인터 타입입니다. C++ 규칙상 **내장 타입에는 연산자 오버로딩을 추가할 수 없습니다.** `Value` 클래스 안에 `operator+`를 넣어도 `Value* + Value*`라는 문법을 만들 수 없습니다.

```
Python:  a + b   →  a.__add__(b)  (a는 Value 객체)
C++:     add(a, b)               (a, b는 Value* 포인터)
```

그래서 `add(a, b)`, `mul(a, b)` 같은 **자유 함수(free function)** 를 사용합니다. 클래스 밖에 독립적으로 정의된 일반 함수입니다. 문법이 약간 다르지만, 하는 일은 Python과 완전히 같습니다.

---

## `inline` 키워드란?

`microgpt.cpp`의 모든 연산 함수 앞에 `inline`이 붙어 있습니다.

```cpp
inline Value* add(Value* a, Value* b) { ... }
```

`inline`은 컴파일러에게 "이 함수를 호출하는 자리마다 함수 본체를 직접 복사해서 넣어 줘" 라고 요청하는 힌트입니다.

```
// inline 없이 (함수 호출 발생)
call add        ; 함수 주소로 점프
...             ; add 내부 실행
ret             ; 호출한 곳으로 복귀

// inline 적용 시 (복사해서 삽입)
; add 내부 코드가 이 자리에 직접 들어옴
```

함수 호출 자체에도 약간의 비용이 있는데, `add`처럼 본체가 한 줄인 함수는 호출 비용이 본체 실행 비용보다 클 수 있습니다. `inline`으로 이 비용을 없앱니다. 최신 컴파일러는 대부분 `inline` 없이도 알아서 판단하지만, 명시적으로 써 두면 의도가 명확해집니다.

---

## 연산 구현 — 한 줄씩

이제 각 연산을 하나씩 살펴보겠습니다. 모든 연산의 구조는 동일합니다.

```
graph.make(순전파_값,  {자식1, 자식2, ...},  {국소기울기1, 국소기울기2, ...})
```

### 덧셈 `add(a, b)` — a + b

**수학:** z = a + b

**기울기:**
- `dz/da = 1` — a가 1 늘면 z도 1 늠
- `dz/db = 1` — b가 1 늘면 z도 1 늠

```
Python:                          C++:
a + b                            add(a, b)
↓                                ↓
Value(a.data + b.data,           graph.make(a->data + b->data,
      (a, b),                              {a, b},
      (1, 1))                              {1.0, 1.0});
```

```cpp
// C++ 구현 (microgpt.cpp 78~80줄)
inline Value* add(Value* a, Value* b) {
    return graph.make(a->data + b->data, {a, b}, {1.0, 1.0});
}
```

줄별 설명:
- `a->data + b->data` — 순전파: 두 값의 합을 계산
- `{a, b}` — 자식 노드: a와 b를 역전파 때 기울기를 받을 노드로 등록
- `{1.0, 1.0}` — 국소 기울기: a에 대한 기울기 1.0, b에 대한 기울기 1.0

---

### 곱셈 `mul(a, b)` — a × b

**수학:** z = a × b

**기울기:**
- `dz/da = b` — a가 1 늘면 z는 b만큼 늠
- `dz/db = a` — b가 1 늘면 z는 a만큼 늠

```
Python:                          C++:
a * b                            mul(a, b)
↓                                ↓
Value(a.data * b.data,           graph.make(a->data * b->data,
      (a, b),                              {a, b},
      (b.data, a.data))                    {b->data, a->data});
```

```cpp
// microgpt.cpp 83~85줄
inline Value* mul(Value* a, Value* b) {
    return graph.make(a->data * b->data, {a, b}, {b->data, a->data});
}
```

**주목:** `a->data`에서 `->` 는 포인터로 멤버에 접근하는 연산자입니다. Python의 `a.data`에 해당합니다. `a`가 포인터이기 때문에 `.` 대신 `->`를 씁니다.

---

### 스칼라 곱 `scale(a, s)` — a × s (s는 상수)

**수학:** z = a × s (s는 학습되지 않는 상수)

**기울기:**
- `dz/da = s` — 상수 s는 기울기에 영향을 줌
- s 자체는 `Value*`가 아니므로 역전파를 받지 않음

```
Python:                          C++:
self * -1  (예: neg)             scale(a, -1.0)
↓                                ↓
Value(self.data * (-1),          graph.make(a->data * s,
      (self,),                             {a},
      (-1,))                               {s});
```

```cpp
// microgpt.cpp 88~90줄
inline Value* scale(Value* a, double s) {
    return graph.make(a->data * s, {a}, {s});
}
```

`mul`과 달리 자식이 `a` 하나뿐입니다. `s`는 상수이므로 계산 그래프에 등록되지 않습니다.

---

### 거듭제곱 `power(a, n)` — a^n

**수학:** z = aⁿ

**기울기:**
- `dz/da = n × a^(n-1)` — 미적분의 멱함수 규칙

```
Python:                          C++:
a ** n                           power(a, n)
↓                                ↓
Value(a.data**n,                 graph.make(std::pow(a->data, n),
      (a,),                                {a},
      (n * a.data**(n-1),))                {n * std::pow(a->data, n-1)});
```

```cpp
// microgpt.cpp 93~95줄
inline Value* power(Value* a, double n) {
    return graph.make(std::pow(a->data, n), {a}, {n * std::pow(a->data, n - 1)});
}
```

`std::pow(x, y)`는 `<cmath>` 헤더에서 제공하는 거듭제곱 함수입니다. Python의 `x**y`에 해당합니다.

---

### 자연로그 `log(a)` — ln(a)

**수학:** z = ln(a)

**기울기:**
- `dz/da = 1/a`

```
Python:                          C++:
self.log()                       log(a)
↓                                ↓
Value(math.log(self.data),       graph.make(std::log(a->data),
      (self,),                             {a},
      (1/self.data,))                      {1.0 / a->data});
```

```cpp
// microgpt.cpp 98~100줄
inline Value* log(Value* a) {
    return graph.make(std::log(a->data), {a}, {1.0 / a->data});
}
```

**주의:** C++ 표준 라이브러리에도 `std::log`가 있습니다. 우리가 정의한 `log` 함수와 이름이 겹치지 않도록 `microgpt.cpp`는 `using namespace std;`를 사용하지 않습니다. 연산 함수 안에서 `std::log`를 명시적으로 씁니다.

---

### 지수함수 `exp(a)` — e^a

**수학:** z = eᵃ

**기울기:**
- `dz/da = eᵃ` — 지수함수의 미분은 자기 자신

```
Python:                          C++:
self.exp()                       exp(a)
↓                                ↓
Value(math.exp(self.data),       double e = std::exp(a->data);
      (self,),                   graph.make(e, {a}, {e});
      (math.exp(self.data),))
```

```cpp
// microgpt.cpp 103~106줄
inline Value* exp(Value* a) {
    double e = std::exp(a->data);   // 순전파 값을 미리 계산
    return graph.make(e, {a}, {e}); // 기울기도 같은 값 e
}
```

`e`를 변수에 저장하는 이유: `std::exp(a->data)`를 두 번 계산하는 낭비를 피하기 위해서입니다.

---

### ReLU 활성화 `relu(a)` — max(0, a)

**수학:** z = max(0, a)

**기울기:**
- `dz/da = 1` (a > 0일 때)
- `dz/da = 0` (a ≤ 0일 때)

```
[ReLU의 모양]

z
│      /
│     /
│    /
│   /
0───────── a
  (음수 구간은 0, 양수 구간은 그대로)
```

```
Python:                          C++:
self.relu()                      relu(a)
↓                                ↓
Value(max(0, self.data),         graph.make(std::max(0.0, a->data),
      (self,),                             {a},
      (float(self.data > 0),))             {a->data > 0 ? 1.0 : 0.0});
```

```cpp
// microgpt.cpp 109~111줄
inline Value* relu(Value* a) {
    return graph.make(std::max(0.0, a->data), {a}, {a->data > 0 ? 1.0 : 0.0});
}
```

`a->data > 0 ? 1.0 : 0.0`은 **삼항 연산자**입니다. `if a->data > 0 then 1.0 else 0.0`과 동일합니다. Python의 `float(self.data > 0)`에 해당합니다.

---

### 복합 연산 — 기본 연산의 조합

나머지 연산들은 위에서 정의한 기본 연산들을 조합합니다. 새로운 역전파 규칙을 추가할 필요 없이, 체인룰이 자동으로 처리합니다.

```
Python:                          C++:
def __neg__(self):               inline Value* neg(Value* a) {
    return self * -1                 return scale(a, -1.0);
                                 }

def __sub__(self, other):        inline Value* sub(Value* a, Value* b) {
    return self + (-other)           return add(a, neg(b));
                                 }

def __truediv__(self, other):    inline Value* div(Value* a, Value* b) {
    return self * other**-1          return mul(a, power(b, -1.0));
                                 }
```

```cpp
// microgpt.cpp 114~121줄
inline Value* neg(Value* a)              { return scale(a, -1.0); }
inline Value* sub(Value* a, Value* b)    { return add(a, neg(b)); }
inline Value* div(Value* a, Value* b)    { return mul(a, power(b, -1.0)); }

// 상수 빼기: 상수를 위한 노드를 만들지 않아 더 효율적
inline Value* sub_const(Value* a, double c) {
    return graph.make(a->data - c, {a}, {1.0});
}
```

`sub_const`는 Python에는 없는 C++ 전용 최적화입니다. `val - max_val`처럼 상수를 뺄 때, 상수에 대한 `Value*` 노드를 만들지 않으므로 메모리와 연산이 절약됩니다.

---

## 전체 연산 요약 표

| 함수 | 수식 | 순전파 | 기울기 (da) |
|------|------|--------|------------|
| `add(a,b)` | a+b | `a->data + b->data` | 1.0, 1.0 |
| `mul(a,b)` | a×b | `a->data * b->data` | `b->data`, `a->data` |
| `scale(a,s)` | a×s | `a->data * s` | `s` |
| `power(a,n)` | aⁿ | `pow(a->data, n)` | `n * pow(a->data, n-1)` |
| `log(a)` | ln(a) | `log(a->data)` | `1.0 / a->data` |
| `exp(a)` | eᵃ | `exp(a->data)` | `exp(a->data)` |
| `relu(a)` | max(0,a) | `max(0, a->data)` | `a->data > 0 ? 1 : 0` |
| `neg(a)` | -a | `scale(a,-1)` | -1.0 |
| `sub(a,b)` | a-b | `add(a, neg(b))` | 1.0, -1.0 |
| `div(a,b)` | a/b | `mul(a, power(b,-1))` | 1/b, -a/b² |
| `sub_const(a,c)` | a-c | `a->data - c` | 1.0 |

---

## 단위 테스트 코드

각 연산이 올바르게 구현되었는지 확인하는 테스트 프로그램입니다. 컴파일해서 직접 실행해 보세요.

```cpp
// ch10_test.cpp — 미분 가능한 연산 단위 테스트
//
// 컴파일:
//   g++ -std=c++17 -O2 -o ch10_test ch10_test.cpp
// 실행:
//   ./ch10_test

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <vector>

// ---- Value 구조체와 Graph (microgpt.cpp에서 복사) ----
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};

struct Graph {
    std::vector<Value*> nodes;

    Value* make(double data, std::vector<Value*> children = {},
                std::vector<double> local_grads = {}) {
        auto* v = new Value{data, 0.0, std::move(children), std::move(local_grads)};
        nodes.push_back(v);
        return v;
    }

    void clear() {
        for (auto* v : nodes) delete v;
        nodes.clear();
    }
    ~Graph() { clear(); }
};

static Graph graph;

// ---- 연산 함수들 ----
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
    return graph.make(std::pow(a->data, n), {a}, {n * std::pow(a->data, n - 1)});
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

// ---- 역전파 (microgpt.cpp에서 복사) ----
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

// ---- 헬퍼: 부동소수점 근사 비교 ----
bool approx(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// ---- 테스트 함수들 ----

void test_add() {
    // c = a + b, a=2, b=3 → c=5
    // dc/da = 1, dc/db = 1
    Value* a = graph.make(2.0);
    Value* b = graph.make(3.0);
    Value* c = add(a, b);

    assert(approx(c->data, 5.0));     // 순전파 확인

    backward(c);
    assert(approx(a->grad, 1.0));     // dc/da = 1
    assert(approx(b->grad, 1.0));     // dc/db = 1

    graph.clear();
    std::cout << "add:      OK\n";
}

void test_mul() {
    // c = a * b, a=3, b=4 → c=12
    // dc/da = b = 4, dc/db = a = 3
    Value* a = graph.make(3.0);
    Value* b = graph.make(4.0);
    Value* c = mul(a, b);

    assert(approx(c->data, 12.0));

    backward(c);
    assert(approx(a->grad, 4.0));     // dc/da = b = 4
    assert(approx(b->grad, 3.0));     // dc/db = a = 3

    graph.clear();
    std::cout << "mul:      OK\n";
}

void test_scale() {
    // c = a * 5.0, a=3 → c=15
    // dc/da = 5.0
    Value* a = graph.make(3.0);
    Value* c = scale(a, 5.0);

    assert(approx(c->data, 15.0));

    backward(c);
    assert(approx(a->grad, 5.0));

    graph.clear();
    std::cout << "scale:    OK\n";
}

void test_power() {
    // c = a^3, a=2 → c=8
    // dc/da = 3 * 2^2 = 12
    Value* a = graph.make(2.0);
    Value* c = power(a, 3.0);

    assert(approx(c->data, 8.0));

    backward(c);
    assert(approx(a->grad, 12.0));    // 3 * 2^2 = 12

    graph.clear();
    std::cout << "power:    OK\n";
}

void test_log() {
    // c = ln(a), a=1 → c=0
    // dc/da = 1/a = 1/1 = 1
    Value* a = graph.make(1.0);
    Value* c = log(a);

    assert(approx(c->data, 0.0));

    backward(c);
    assert(approx(a->grad, 1.0));

    graph.clear();

    // 추가 확인: a=e → ln(e)=1, dc/da=1/e
    Value* a2 = graph.make(std::exp(1.0));
    Value* c2 = log(a2);
    assert(approx(c2->data, 1.0));
    backward(c2);
    assert(approx(a2->grad, 1.0 / std::exp(1.0)));

    graph.clear();
    std::cout << "log:      OK\n";
}

void test_exp() {
    // c = e^a, a=0 → c=1
    // dc/da = e^0 = 1
    Value* a = graph.make(0.0);
    Value* c = exp(a);

    assert(approx(c->data, 1.0));

    backward(c);
    assert(approx(a->grad, 1.0));

    graph.clear();

    // a=1 → c=e, dc/da=e
    Value* a2 = graph.make(1.0);
    Value* c2 = exp(a2);
    assert(approx(c2->data, std::exp(1.0)));
    backward(c2);
    assert(approx(a2->grad, std::exp(1.0)));

    graph.clear();
    std::cout << "exp:      OK\n";
}

void test_relu() {
    // 양수 입력: c = relu(3) = 3, dc/da = 1
    Value* a1 = graph.make(3.0);
    Value* c1 = relu(a1);
    assert(approx(c1->data, 3.0));
    backward(c1);
    assert(approx(a1->grad, 1.0));
    graph.clear();

    // 음수 입력: c = relu(-2) = 0, dc/da = 0
    Value* a2 = graph.make(-2.0);
    Value* c2 = relu(a2);
    assert(approx(c2->data, 0.0));
    backward(c2);
    assert(approx(a2->grad, 0.0));
    graph.clear();

    std::cout << "relu:     OK\n";
}

void test_neg() {
    // c = -a, a=5 → c=-5, dc/da=-1
    Value* a = graph.make(5.0);
    Value* c = neg(a);
    assert(approx(c->data, -5.0));
    backward(c);
    assert(approx(a->grad, -1.0));
    graph.clear();
    std::cout << "neg:      OK\n";
}

void test_sub() {
    // c = a - b, a=7, b=3 → c=4
    // dc/da=1, dc/db=-1
    Value* a = graph.make(7.0);
    Value* b = graph.make(3.0);
    Value* c = sub(a, b);
    assert(approx(c->data, 4.0));
    backward(c);
    assert(approx(a->grad,  1.0));
    assert(approx(b->grad, -1.0));
    graph.clear();
    std::cout << "sub:      OK\n";
}

void test_div() {
    // c = a / b, a=6, b=3 → c=2
    // dc/da = 1/b = 1/3
    // dc/db = -a/b^2 = -6/9 = -2/3
    Value* a = graph.make(6.0);
    Value* b = graph.make(3.0);
    Value* c = div(a, b);
    assert(approx(c->data, 2.0));
    backward(c);
    assert(approx(a->grad,  1.0/3.0));
    assert(approx(b->grad, -2.0/3.0));
    graph.clear();
    std::cout << "div:      OK\n";
}

void test_sub_const() {
    // c = a - 10.0, a=15 → c=5, dc/da=1
    Value* a = graph.make(15.0);
    Value* c = sub_const(a, 10.0);
    assert(approx(c->data, 5.0));
    backward(c);
    assert(approx(a->grad, 1.0));
    graph.clear();
    std::cout << "sub_const:OK\n";
}

int main() {
    std::cout << "=== Ch10 연산 단위 테스트 ===\n";
    test_add();
    test_mul();
    test_scale();
    test_power();
    test_log();
    test_exp();
    test_relu();
    test_neg();
    test_sub();
    test_div();
    test_sub_const();
    std::cout << "\n모든 테스트 통과!\n";
    return 0;
}
```

### 컴파일과 실행

```bash
g++ -std=c++17 -O2 -o ch10_test ch10_test.cpp
./ch10_test
```

예상 출력:

```
=== Ch10 연산 단위 테스트 ===
add:      OK
mul:      OK
scale:    OK
power:    OK
log:      OK
exp:      OK
relu:     OK
neg:      OK
sub:      OK
div:      OK
sub_const:OK

모든 테스트 통과!
```

---

## 핵심 정리

| 개념 | Python | C++ |
|------|--------|-----|
| 연산 방식 | `a + b` (연산자 오버로딩) | `add(a, b)` (자유 함수) |
| 포인터 멤버 접근 | `a.data` | `a->data` |
| 거듭제곱 | `a ** n` | `std::pow(a, n)` |
| 로그 | `math.log(x)` | `std::log(x)` |
| 삼항 연산자 | `1.0 if x > 0 else 0.0` | `x > 0 ? 1.0 : 0.0` |
| `inline` | 없음 | 함수 호출 비용 제거 힌트 |

핵심 패턴:

```cpp
// 모든 연산의 구조
inline Value* 연산이름(Value* a, ...) {
    return graph.make(
        순전파_값,        // 실제 계산 결과
        {자식1, 자식2},  // 역전파를 받을 노드들
        {기울기1, 기울기2} // 각 자식에 대한 국소 미분
    );
}
```

다음 챕터에서는 이 연산들로 구성된 계산 그래프를 역방향으로 순회하는 **역전파(backward)** 알고리즘을 구현합니다.


> **직접 체험하기** — 시각화 도구에서 미분 가능한 연산의 순전파/역전파를 애니메이션으로 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#backprop)

---
[< 이전: Ch09: Graph 아레나](ch09-graph-arena.md) | [목차](../README.md) | [다음: Ch11: 역전파 구현 >](ch11-backward.md)
