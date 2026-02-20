# Ch08: Value 구조체

## 학습 목표

- Python `Value` 클래스의 역할을 복습하고 C++ `struct Value`와 1:1 대응시킨다
- 각 필드(`data`, `grad`, `children`, `local_grads`)의 목적을 이해한다
- `std::vector<Value*>`가 Python 튜플 `(self, other)`에 대응함을 파악한다
- `new Value{...}`로 힙에 생성해야 하는 이유를 납득한다
- 직접 `Value`를 생성하고 필드에 접근하는 코드를 작성한다

---

## 개념 설명

### Python `Value` 클래스 복습

Part 1에서 배운 `Value` 클래스를 다시 봅시다. 이것이 자동 미분 엔진의 핵심입니다.

```python
# microgpt.py:30-37
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data            # 이 노드의 스칼라 값
        self.grad = 0               # 손실에 대한 이 노드의 기울기
        self._children = children   # 계산 그래프에서 이 노드의 입력들
        self._local_grads = local_grads  # 각 입력에 대한 국소 미분
```

`Value` 하나는 계산 그래프의 **노드 하나**입니다. 예를 들어 `z = x * y`라면:

```
    x(data=2) ──→ [mul] ──→ z(data=6)
    y(data=3) ──→

    z._children    = (x, y)
    z._local_grads = (y.data, x.data) = (3, 2)  # dz/dx=y, dz/dy=x
```

### C++ `struct Value` 설계

이제 Python `Value`를 C++로 옮겨 봅시다. 필드는 동일합니다.

```cpp
// microgpt.cpp:41-46
struct Value {
    double data;                       // 스칼라 값 (순전파에서 계산)
    double grad = 0.0;                 // dLoss/d(this) (역전파에서 계산)
    std::vector<Value*> children;      // 계산 그래프에서의 자식 노드들
    std::vector<double> local_grads;   // 각 자식에 대한 국소 미분값
};
```

---

## 각 필드 상세 설명

### `double data` — 노드의 값

```
Python: self.data = data    →    C++: double data;
```

`double`은 64비트 부동소수점 실수입니다. Python의 `float`가 실제로 이것입니다.

순전파(forward pass) 때 이 값이 채워집니다.

```
z = x * y
→ z.data = x.data * y.data = 2.0 * 3.0 = 6.0
```

### `double grad = 0.0` — 기울기

```
Python: self.grad = 0    →    C++: double grad = 0.0;
```

`= 0.0`은 **멤버 기본값 초기화**입니다. C++11부터 지원합니다. `Value` 객체가 생성될 때 `grad`가 자동으로 `0.0`으로 초기화됩니다. Python의 `self.grad = 0`과 동일합니다.

역전파(backward pass) 때 이 값이 채워집니다.

```
dLoss/d(z)가 계산되면 z.grad에 저장된다
→ 이것이 연쇄적으로 x.grad, y.grad에 전파된다
```

### `std::vector<Value*> children` — 자식 노드들

```
Python: self._children = children  (tuple)
C++:    std::vector<Value*> children;
```

Python에서 `_children`은 `(x, y)` 같은 튜플입니다. C++에서는 `std::vector<Value*>`, 즉 **포인터의 동적 배열**입니다.

왜 튜플 대신 `vector`인가? C++의 `tuple`은 컴파일 타임에 크기가 고정되어야 합니다. 우리는 연산에 따라 자식이 1개(`log`, `relu`, `exp`)일 수도 있고 2개(`add`, `mul`)일 수도 있습니다. `vector`는 크기가 동적으로 바뀝니다.

왜 `Value`가 아니라 `Value*`인가? Ch07에서 배운 이유 그대로입니다. 같은 노드를 여러 곳에서 참조하려면 포인터가 필요합니다.

```
z = x * y + x

z.children = [mul_result_ptr, x_ptr]  (포인터들)
              ↑ mul 결과 노드          ↑ 원본 x 노드 (복사 아님!)
```

### `std::vector<double> local_grads` — 국소 미분값

```
Python: self._local_grads = local_grads  (tuple)
C++:    std::vector<double> local_grads;
```

역전파 때 체인룰 적용에 쓰이는 값입니다. `children`과 1:1 대응합니다.

```
z = x * y일 때:
  children    = [x, y]
  local_grads = [y.data, x.data]  // dz/dx = y.data, dz/dy = x.data

역전파:
  x.grad += local_grads[0] * z.grad  // x.grad += y.data * z.grad
  y.grad += local_grads[1] * z.grad  // y.grad += x.data * z.grad
```

---

## Python `__slots__` vs C++ struct

Python에서 `__slots__`를 쓰면 `__dict__`를 만들지 않아 메모리를 아낍니다.

```python
# __slots__ 없이: 각 인스턴스가 __dict__ 딕셔너리를 갖는다 (메모리 낭비)
# __slots__ 있이: 딱 선언한 필드만, 고정 레이아웃으로 저장 (효율적)
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
```

C++ `struct`는 항상 `__slots__`를 쓴 것과 같습니다. 선언한 필드만 메모리에 순서대로 놓입니다. 별도 딕셔너리가 없습니다.

```
메모리 레이아웃
──────────────────────────────────────
Python Value (without __slots__):
  [__dict__ 포인터][data][grad][...]   ← 딕셔너리 오버헤드 있음

Python Value (with __slots__):
  [data][grad][children_ptr][local_grads_ptr]  ← 효율적

C++ struct Value:
  [data(8B)][grad(8B)][children(24B)][local_grads(24B)]  ← 항상 효율적
──────────────────────────────────────
```

---

## 코드 작성: Value 생성 실습

### 집합체 초기화(Aggregate Initialization)

C++ `struct`는 중괄호 `{}`로 필드를 순서대로 초기화할 수 있습니다.

```cpp
// 형식: Value{ data, grad, children, local_grads }
Value v{3.14, 0.0, {}, {}};
//      ^^^^  ^^^  ^^  ^^
//      data  grad  children(빈 vector) local_grads(빈 vector)
```

### 필드 접근

포인터로 만든 `Value`의 필드에 접근할 때는 `->`를 씁니다. 일반 변수일 때는 `.`을 씁니다.

```cpp
Value  v{3.14, 0.0, {}, {}};
Value* p = &v;

v.data;    // 일반 변수 → 점(.)
p->data;   // 포인터    → 화살표(->)
(*p).data; // 역참조 후 점(.) → 위와 동일, 번거로움
```

### 힙에 생성해야 하는 이유

```cpp
// 스택에 만들면?
Value* bad_ptr;
{
    Value v{3.14, 0.0, {}, {}};  // 스택에 생성
    bad_ptr = &v;                 // v의 주소 저장
}
// 여기서 v는 소멸됨! bad_ptr이 가리키는 메모리는 쓰레기
std::cout << bad_ptr->data; // 미정의 동작 (Undefined Behavior)!
```

순전파가 끝난 뒤에도 역전파를 위해 모든 `Value` 노드가 살아 있어야 합니다. 스택에 만들면 함수가 끝나는 순간 사라집니다. 힙(`new`)에 만들어야 역전파 때까지 유지됩니다.

```cpp
// 힙에 만들면?
Value* good_ptr = new Value{3.14, 0.0, {}, {}};  // 힙에 생성
// → delete를 부르기 전까지 살아 있음
// → 함수 경계를 넘어도 안전
```

---

## 완전한 실습 코드

다음 코드를 `value_demo.cpp`로 저장하고 컴파일해 봅시다.

```cpp
// value_demo.cpp
// Value 구조체 생성, 필드 접근, 계산 그래프 수동 구성 실습

#include <iostream>
#include <vector>

// ─── Value 구조체 ────────────────────────────────────────────
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};

int main() {
    // ─── 1. 스택에 생성 (함수 내에서만 유효) ─────────────────
    Value stack_val{2.0, 0.0, {}, {}};
    std::cout << "스택 Value: data=" << stack_val.data
              << ", grad=" << stack_val.grad << "\n";

    // ─── 2. 힙에 생성 (delete 전까지 유효) ───────────────────
    Value* x = new Value{2.0, 0.0, {}, {}};
    Value* y = new Value{3.0, 0.0, {}, {}};

    std::cout << "x->data = " << x->data << "\n";  // 2
    std::cout << "y->data = " << y->data << "\n";  // 3

    // ─── 3. 수동으로 계산 그래프 구성: z = x * y ─────────────
    // 순전파: z.data = x.data * y.data
    // local_grads: dz/dx = y.data = 3.0, dz/dy = x.data = 2.0
    Value* z = new Value{
        x->data * y->data,   // data = 6.0
        0.0,                  // grad (역전파 전에는 0)
        {x, y},               // children: x와 y를 가리키는 포인터
        {y->data, x->data}    // local_grads: dz/dx=3.0, dz/dy=2.0
    };

    std::cout << "z->data = " << z->data << "\n";  // 6

    // ─── 4. 역전파 흉내 내기 (loss=z라고 가정) ───────────────
    z->grad = 1.0;  // 손실에 대한 z의 기울기 = 1 (chain rule 시작점)

    // z.children = [x, y], z.local_grads = [3.0, 2.0]
    for (size_t i = 0; i < z->children.size(); i++) {
        z->children[i]->grad += z->local_grads[i] * z->grad;
    }

    std::cout << "x->grad = " << x->grad << "\n";  // 3.0 (dz/dx = y.data)
    std::cout << "y->grad = " << y->grad << "\n";  // 2.0 (dz/dy = x.data)

    // ─── 5. children이 같은 노드를 가리키는 경우: w = x + x ──
    // x에서 두 경로로 기울기가 쌓여야 한다
    x->grad = 0.0;  // 리셋
    Value* w = new Value{
        x->data + x->data,   // data = 4.0
        0.0,
        {x, x},               // 같은 x 노드를 두 번 참조
        {1.0, 1.0}            // dw/dx1 = 1, dw/dx2 = 1
    };

    w->grad = 1.0;
    for (size_t i = 0; i < w->children.size(); i++) {
        w->children[i]->grad += w->local_grads[i] * w->grad;
    }

    // x->grad = 1.0 + 1.0 = 2.0 (두 경로에서 각각 1씩 누적)
    std::cout << "w = x+x, x->grad = " << x->grad << "\n";  // 2.0

    // ─── 정리 ─────────────────────────────────────────────────
    delete w;
    delete z;
    delete x;
    delete y;

    return 0;
}
```

**컴파일 및 실행:**

```bash
g++ -std=c++17 -O2 -o value_demo value_demo.cpp && ./value_demo
```

**실행 결과:**
```
스택 Value: data=2, grad=0
x->data = 2
y->data = 3
z->data = 6
x->grad = 3
y->grad = 2
w = x+x, x->grad = 2
```

`x->grad = 2`가 나오는 것에 주목하세요. `x` 포인터가 `w->children`에 두 번 들어 있기 때문에, 역전파 때 기울기가 1.0씩 두 번 누적되어 2.0이 됩니다. 포인터를 쓰기 때문에 이것이 가능합니다. 값 복사를 썼다면 각각 별도의 `x`가 되어 원본에는 기울기가 쌓이지 않습니다.

---

## 실제 microgpt.cpp와 비교

실제 코드에서는 `Value`를 직접 `new`하지 않고 `Graph::make()`를 씁니다(Ch09에서 설명). 하지만 `struct Value` 자체는 완전히 동일합니다.

```cpp
// microgpt.cpp:41-46 (실제 코드)
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};
```

Python과 필드 이름이 어떻게 대응하는지 정리합니다:

| Python (`Value`) | C++ (`struct Value`) | 타입 |
|------------------|----------------------|------|
| `self.data` | `v->data` | `double` |
| `self.grad` | `v->grad` | `double` (기본값 `0.0`) |
| `self._children` | `v->children` | `std::vector<Value*>` |
| `self._local_grads` | `v->local_grads` | `std::vector<double>` |

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `struct Value` | 계산 그래프의 노드 하나를 나타내는 구조체 |
| `double data` | 순전파에서 계산되는 스칼라 값 |
| `double grad = 0.0` | 역전파에서 누적되는 기울기, 기본값 0.0 |
| `std::vector<Value*> children` | 이 노드의 입력 노드들 (포인터 배열) |
| `std::vector<double> local_grads` | 각 입력에 대한 국소 미분 |
| `v.field` | 일반 변수의 필드 접근 |
| `p->field` | 포인터의 필드 접근 (`(*p).field`와 동일) |
| `new Value{...}` | 힙에 Value 생성 (함수 끝나도 살아 있음) |
| `delete p` | 힙 메모리 해제 |

다음 챕터에서는 수천 개의 임시 `Value` 노드를 효율적으로 관리하는 **Graph 아레나** 패턴을 배웁니다.


---
[< 이전: Ch07: Python→C++ 브릿지](ch07-python-to-cpp-bridge.md) | [목차](../README.md) | [다음: Ch09: Graph 아레나 >](ch09-graph-arena.md)
