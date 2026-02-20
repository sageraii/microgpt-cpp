# Ch07: Python→C++ 브릿지

## 학습 목표

- Python과 C++의 핵심 차이점을 비교표로 파악한다
- Python의 자동 메모리 관리 vs C++의 수동 메모리 관리를 이해한다
- 왜 `Value*`(포인터)를 사용하고 `a + b` 대신 `add(a, b)`를 쓰는지 납득한다
- 같은 수식을 Python과 C++ 양쪽으로 써서 차이를 몸으로 익힌다
- `using Vec`과 `using Mat` 타입 별칭을 이해한다

---

## 개념 설명

### 두 언어의 철학

Python은 "쉽게 쓰되 컴퓨터가 대신 해준다"는 철학입니다. C++는 "컴퓨터가 무슨 일을 하는지 프로그래머가 직접 결정한다"는 철학입니다.

이 차이는 GPT 구현 전반에 영향을 줍니다. 이 챕터에서 그 차이를 하나씩 짚어봅니다.

---

## Python↔C++ 핵심 차이점 비교

### 비교표

| Python | C++ | 설명 |
|--------|-----|------|
| `class Value` | `struct Value` | 데이터 묶음 정의 방식 |
| 가비지 컬렉션 | `new` / `delete` | 메모리 수명 관리 |
| `dict` | `std::unordered_map` | 키-값 저장소 |
| `list` | `std::vector` | 동적 배열 |
| 리스트 컴프리헨션 | `for` 루프 + `push_back` | 리스트 생성 |
| `__add__` 연산자 | `add(a, b)` 자유 함수 | 연산자 정의 방식 |
| `random.gauss` | `std::normal_distribution` | 가우시안 난수 |
| `True` / `False` | `true` / `false` | 불리언 값 |
| `None` | `nullptr` | 없음을 나타내는 값 |

---

### 차이 1: `class` vs `struct`

Python에서 `Value`는 클래스로 정의됩니다.

```python
# Python: class로 정의
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

C++에서는 `struct`로 정의합니다.

```cpp
// C++: struct로 정의
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};
```

C++에서 `struct`와 `class`의 실질적 차이는 기본 접근 제어뿐입니다(`struct`는 `public`이 기본). 이 튜토리얼에서는 데이터 묶음을 나타낼 때 `struct`를 씁니다. 간결하고 의도가 명확하기 때문입니다.

---

### 차이 2: 메모리 관리

이것이 두 언어의 가장 큰 차이입니다.

**Python: 자동 (가비지 컬렉션)**

```python
# Python
x = Value(2.0)   # Value 객체가 힙에 생성됨
y = Value(3.0)   # 또 생성
z = x + y       # 또 생성

# x, y, z가 더 이상 필요 없을 때
# → Python이 알아서 메모리를 해제함
# → 프로그래머는 신경 안 써도 됨
```

Python은 각 객체가 몇 번 참조되고 있는지 세고 있습니다(참조 카운팅). 참조 수가 0이 되면 자동으로 메모리를 해제합니다.

**C++: 수동 (new / delete)**

```cpp
// C++
Value* x = new Value{2.0, 0.0, {}, {}};  // 힙에 생성 (new)
Value* y = new Value{3.0, 0.0, {}, {}};  // 힙에 생성 (new)

// ... 사용 ...

delete x;   // 반드시 직접 해제해야 함
delete y;   // 안 하면 메모리 누수!
```

C++에서 `new`로 만든 객체는 반드시 `delete`해야 합니다. 안 하면 프로그램이 끝날 때까지 메모리가 사라지지 않습니다. 이를 **메모리 누수(memory leak)**라고 합니다.

**스택 vs 힙:**

```
스택(Stack)                    힙(Heap)
───────────────────────        ──────────────────────────────
함수 안에서 선언한 변수         new로 생성한 객체
함수가 끝나면 자동 소멸         delete를 불러야 소멸
크기 제한 있음 (수 MB)         크기 제한 거의 없음
빠르다                         상대적으로 느리다

Value v{2.0};  ← 스택        new Value{2.0}  ← 힙
(함수 끝나면 사라짐)           (delete 전까지 살아 있음)
```

---

### 차이 3: 왜 `Value*` 포인터를 사용하나?

이것이 이 튜토리얼에서 가장 중요한 개념입니다.

**계산 그래프는 같은 노드를 여러 곳에서 참조해야 합니다.**

예를 들어 `z = x * y + x`를 생각해 봅시다. `x`는 `mul`의 왼쪽 입력이기도 하고, `add`의 오른쪽 입력이기도 합니다. 계산 그래프에서 `x`는 **하나의 노드**이고, 두 곳에서 그 노드를 **가리켜야** 합니다.

```
    x ──┬──→ [mul] ──→ [add] ──→ z
        │                 ↑
    y ──┘                 │
                          └── x (같은 x 노드!)
```

포인터를 쓰면 여러 곳에서 같은 `Value` 객체를 가리킬 수 있습니다.

```cpp
Value* x = /* ... */;  // x는 Value 객체를 가리키는 포인터

// mul과 add 모두 x 포인터(같은 주소)를 저장한다
// → 역전파 때 x->grad가 두 경로에서 제대로 누적됨
```

포인터를 안 쓰고 값 복사를 하면:

```cpp
Value x_copy1 = *x;  // x 복사본 1
Value x_copy2 = *x;  // x 복사본 2 (완전히 별개의 객체!)

// 역전파 때 x_copy1.grad와 x_copy2.grad가 각자 업데이트됨
// → 원본 x에는 기울기가 전혀 쌓이지 않음!
// → 학습 불가
```

**결론: 계산 그래프의 모든 노드는 포인터(`Value*`)로 다뤄야 합니다.**

---

### 차이 4: 연산자 오버로딩 vs 자유 함수

Python에서는 `__add__`, `__mul__` 같은 특수 메서드로 `+`, `*` 연산자를 정의합니다.

```python
# Python: 연산자 오버로딩 자연스럽게 작동
class Value:
    def __add__(self, other):
        return Value(self.data + other.data, (self, other), (1, 1))

x = Value(2.0)
y = Value(3.0)
z = x + y  # __add__ 호출, 자연스럽게 읽힘
```

C++에서도 연산자 오버로딩이 가능합니다. 그러나 `Value*` 포인터를 쓰면 문제가 생깁니다.

```cpp
// 만약 이렇게 했다면 (실제로는 사용하지 않는 방식)
Value* x = /* ... */;
Value* y = /* ... */;
Value* z = x + y;  // 이것은 포인터 산술!
                   // z = x의 주소 + y의 주소 → 쓰레기 값
```

C++에서 `Value*`끼리 `+`를 하면 두 포인터의 주소값이 더해집니다. `Value`의 `data`가 더해지는 게 아닙니다. 포인터 산술입니다.

해결책은 **자유 함수(free function)**입니다.

```cpp
// C++: 자유 함수 사용
Value* add(Value* a, Value* b) {
    return graph.make(a->data + b->data, {a, b}, {1.0, 1.0});
}

Value* x = /* ... */;
Value* y = /* ... */;
Value* z = add(x, y);  // 명확하게 함수 호출
```

포인터를 쓰기 때문에 `a + b` 대신 `add(a, b)`를 쓰는 것이 핵심 설계 결정입니다.

---

## 같은 수식을 양쪽으로 비교

### 간단한 수식: `z = x * y + x`

**Python:**
```python
# Python (microgpt.py 스타일)
x = Value(2.0)
y = Value(3.0)
z = x * y + x        # 연산자 오버로딩으로 자연스럽게
# z.data = 2*3 + 2 = 8.0
```

**C++:**
```cpp
// C++ (microgpt.cpp 스타일)
Value* x = new Value{2.0, 0.0, {}, {}};
Value* y = new Value{3.0, 0.0, {}, {}};
Value* z = add(mul(x, y), x);    // 자유 함수 체이닝
// z->data = 2*3 + 2 = 8.0
```

### 리스트 생성: 행렬의 한 행

**Python:**
```python
# Python: 리스트 컴프리헨션
import random
row = [Value(random.gauss(0, 0.08)) for _ in range(16)]
```

**C++:**
```cpp
// C++: for 루프 + reserve + push_back
#include <random>
#include <vector>

std::mt19937 rng(42);
std::normal_distribution<double> normal(0.0, 0.08);

std::vector<Value*> row;
row.reserve(16);                          // 미리 공간 확보 (선택사항, 효율적)
for (int i = 0; i < 16; i++) {
    row.push_back(new Value{normal(rng), 0.0, {}, {}});
}
```

`reserve(n)`은 벡터가 내부적으로 메모리를 미리 할당하게 합니다. `push_back`을 반복할 때 재할당이 일어나지 않아 효율적입니다.

### 딕셔너리(해시맵): `state_dict`

**Python:**
```python
# Python
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
}
# 접근
embedding = state_dict['wte'][0]
```

**C++:**
```cpp
// C++
#include <unordered_map>
#include <string>

std::unordered_map<std::string, Mat> state_dict;
state_dict["wte"] = make_matrix(vocab_size, N_EMBD);
state_dict["wpe"] = make_matrix(BLOCK_SIZE, N_EMBD);

// 접근 (Python과 거의 같은 문법)
Vec embedding = state_dict["wte"][0];
```

---

## 코드 작성: 비교 실습

다음 코드를 `bridge_demo.cpp`로 저장하고 컴파일해 봅시다.

```cpp
// bridge_demo.cpp
// Python↔C++ 주요 차이점 실습

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>

// ─── Value 구조체 (Ch08에서 자세히 다룸) ───────────────────
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};

// ─── 타입 별칭 ──────────────────────────────────────────────
// Python: list[Value]   → C++: Vec
// Python: list[list[Value]] → C++: Mat
using Vec = std::vector<Value*>;
using Mat = std::vector<Vec>;

// ─── 자유 함수: 덧셈 (간소화 버전, 아레나 없이) ─────────────
// 실제 microgpt.cpp는 graph.make()를 사용한다 (Ch09에서 설명)
Value* add_simple(Value* a, Value* b) {
    return new Value{a->data + b->data, 0.0, {a, b}, {1.0, 1.0}};
}

Value* mul_simple(Value* a, Value* b) {
    return new Value{a->data * b->data, 0.0, {a, b}, {b->data, a->data}};
}

int main() {
    // ─── 1. 포인터로 Value 생성 ───────────────────────────
    // Python: x = Value(2.0)
    Value* x = new Value{2.0, 0.0, {}, {}};
    Value* y = new Value{3.0, 0.0, {}, {}};

    std::cout << "x->data = " << x->data << "\n";  // 2
    std::cout << "y->data = " << y->data << "\n";  // 3

    // ─── 2. 자유 함수로 수식 계산 ─────────────────────────
    // Python: z = x * y + x  (연산자 오버로딩)
    // C++:    z = add(mul(x, y), x)  (자유 함수 체이닝)
    Value* z = add_simple(mul_simple(x, y), x);
    std::cout << "z->data = " << z->data << "\n";  // 2*3 + 2 = 8

    // ─── 3. vector: Python의 list ─────────────────────────
    // Python: row = [1.0, 2.0, 3.0]
    std::vector<double> row = {1.0, 2.0, 3.0};
    row.push_back(4.0);                              // append
    std::cout << "row size = " << row.size() << "\n"; // 4

    // ─── 4. unordered_map: Python의 dict ─────────────────
    // Python: scores = {"alice": 95, "bob": 87}
    std::unordered_map<std::string, int> scores;
    scores["alice"] = 95;
    scores["bob"]   = 87;
    std::cout << "alice = " << scores["alice"] << "\n"; // 95

    // ─── 5. 가우시안 난수: Python의 random.gauss ─────────
    // Python: v = random.gauss(0, 0.08)
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 0.08);
    double v = normal(rng);
    std::cout << "gauss(0, 0.08) = " << v << "\n";

    // ─── 정리: new로 만든 것은 delete로 해제 ─────────────
    delete z;
    delete x;
    delete y;

    return 0;
}
```

**컴파일 및 실행:**

```bash
g++ -std=c++17 -O2 -o bridge_demo bridge_demo.cpp && ./bridge_demo
```

**실행 결과:**
```
x->data = 2
y->data = 3
z->data = 8
row size = 4
alice = 95
gauss(0, 0.08) = -0.0583...
```

---

## 타입 별칭: `Vec`과 `Mat`

`microgpt.cpp`에서 자주 등장하는 두 가지 별칭입니다.

```cpp
// microgpt.cpp:154-155
using Vec = std::vector<Value*>;  // 1D: 임베딩 하나 (길이 N_EMBD)
using Mat = std::vector<Vec>;     // 2D: 가중치 행렬 (nout × nin)
```

`using 별칭 = 타입;`은 긴 타입 이름에 짧은 이름을 붙이는 것입니다. Python의 타입 별칭(`type Vec = list[Value]`)과 동일한 역할입니다.

**사용 예:**

```cpp
Vec embedding(16);        // std::vector<Value*> embedding(16); 과 동일
Mat weights(27, Vec(16)); // 27×16 행렬 초기화
```

`Mat`의 각 원소는 `Vec`(포인터 배열)이고, `Vec`의 각 원소는 `Value*`(포인터)입니다.

```
Mat weights
 ┌──────────────────────────┐
 │ Vec row0: [V* V* V* ...]  │  ← weights[0]: 첫 번째 행
 │ Vec row1: [V* V* V* ...]  │  ← weights[1]: 두 번째 행
 │ ...                       │
 └──────────────────────────┘
```

---

## 핵심 정리

| Python | C++ | 이유 |
|--------|-----|------|
| `class Value` | `struct Value` | 데이터 묶음에는 struct가 간결 |
| 자동 GC | `new` / `delete` | C++는 수동 메모리 관리 |
| `x + y` (Value) | `add(x, y)` | `Value*`끼리 `+`는 포인터 산술 |
| `list` | `std::vector` | 동적 배열 |
| `dict` | `std::unordered_map` | 해시맵 |
| `random.gauss` | `std::normal_distribution` | 가우시안 난수 |
| `[... for ...]` | `for` + `push_back` | 리스트 생성 |

**포인터(`Value*`)를 쓰는 핵심 이유:**
- 계산 그래프에서 같은 노드를 여러 곳에서 참조해야 한다
- 값을 복사하면 역전파 때 기울기가 원본에 쌓이지 않는다
- 포인터끼리 `+` 연산은 포인터 산술이므로 자유 함수 `add(a, b)`를 쓴다

다음 챕터에서는 `Value` 구조체의 각 필드를 하나씩 뜯어보며 설계 이유를 깊게 이해합니다.


---
[< 이전: Ch06: C++ 환경 설정](ch06-cpp-setup.md) | [목차](../README.md) | [다음: Ch08: Value 구조체 >](ch08-value-struct.md)
