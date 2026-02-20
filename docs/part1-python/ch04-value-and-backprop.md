# Ch04: Value와 역전파

## 학습 목표

이 챕터를 마치면 다음을 이해할 수 있습니다.

- 자동 미분이 왜 필요한지
- `Value` 클래스가 어떻게 연산을 기록하는지
- 계산 그래프가 무엇인지
- 체인룰과 역전파(backpropagation)가 어떻게 동작하는지
- 수치 미분으로 역전파 결과를 검증하는 방법

---

## 1. 왜 자동 미분이 필요한가?

Ch03에서 `y = ax + b`의 기울기를 손으로 계산했습니다.

```python
grad_a = (2/n) * sum((a*x + b - y) * x  for x, y in zip(xs, ys))
grad_b = (2/n) * sum((a*x + b - y)      for x, y in zip(xs, ys))
```

이 계산이 가능했던 이유는 모델이 단순한 직선이었기 때문입니다.

`microgpt.py`의 Transformer는 수천 개의 파라미터가 덧셈, 곱셈, 로그, 지수, ReLU 등을 통해 복잡하게 얽혀 있습니다. 이 모든 파라미터에 대해 손으로 미분 공식을 유도하는 것은 사실상 불가능합니다.

해결책: **연산을 할 때마다 "나중에 미분을 어떻게 계산할지"를 함께 기록해두는 것**입니다. 이것이 자동 미분(Automatic Differentiation)이고, `microgpt.py`에서 `Value` 클래스가 그 역할을 합니다.

---

## 2. Value 클래스 — 연산을 기록하는 숫자

`Value`는 그냥 숫자가 아닙니다. 숫자 값뿐만 아니라, 그 값이 **어떻게 계산됐는지**도 기억합니다.

```python
# microgpt.py:30-37
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # 이 노드의 숫자 값 (순전파에서 계산됨)
        self.grad = 0             # 이 노드의 기울기 (역전파에서 채워짐)
        self._children = children # 이 값을 만드는 데 사용된 입력들
        self._local_grads = local_grads  # 각 입력에 대한 국소 미분
```

네 개의 속성이 핵심입니다.

| 속성 | 역할 | 비유 |
|------|------|------|
| `data` | 이 노드의 숫자 값 | 지금 내 위치 (산의 높이) |
| `grad` | loss에 대한 이 노드의 기울기 | 내 발바닥의 경사 |
| `_children` | 이 값을 만든 입력들 | 이 위치에 도달하는 데 밟은 발판들 |
| `_local_grads` | 각 입력에 대한 국소 미분 | 각 발판이 현재 위치에 미친 영향 |

### `__slots__`에 대해

`__slots__`은 Python의 메모리 최적화 기법입니다. 클래스 속성을 미리 선언해두면, Python이 더 효율적인 방식으로 메모리를 사용합니다. `microgpt.py`에서는 `Value` 객체가 수만 개 생성되므로 이 최적화가 의미 있습니다. 지금은 "성능을 위한 장치"라고만 이해하면 됩니다.

---

## 3. 덧셈: `__add__`

```python
# microgpt.py:39-41
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))
```

한 줄씩 읽겠습니다.

- `other = other if isinstance(other, Value) else Value(other)`: `other`가 일반 숫자(`int` 또는 `float`)이면 `Value`로 감쌉니다. `a + 3`처럼 쓸 수 있게 해줍니다.
- `self.data + other.data`: 실제 덧셈 결과
- `(self, other)`: 자식(children) — 이 덧셈에 사용된 두 값
- `(1, 1)`: 국소 미분 — `d(a+b)/da = 1`, `d(a+b)/db = 1`

왜 덧셈의 미분이 1인가? `f(a, b) = a + b`에서:
- a를 1 늘리면 결과가 1 늘어남 → 기울기 = 1
- b를 1 늘리면 결과가 1 늘어남 → 기울기 = 1

그림으로 표현하면:

```
a --[기울기: 1]--> (a + b)
b --[기울기: 1]--> (a + b)
```

---

## 4. 곱셈: `__mul__`

```python
# microgpt.py:43-45
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

- `self.data * other.data`: 실제 곱셈 결과
- `(self, other)`: 자식
- `(other.data, self.data)`: 국소 미분 — `d(a*b)/da = b`, `d(a*b)/db = a`

왜 곱셈의 미분이 그런가? `f(a, b) = a × b`에서:
- a를 1 늘리면 결과가 b만큼 늘어남 → a의 기울기 = b
- b를 1 늘리면 결과가 a만큼 늘어남 → b의 기울기 = a

그림으로 표현하면:

```
a --[기울기: b]--> (a × b)
b --[기울기: a]--> (a × b)
```

---

## 5. 나머지 연산들

```python
# microgpt.py:47-50
def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
def log(self):            return Value(math.log(self.data), (self,), (1/self.data,))
def exp(self):            return Value(math.exp(self.data), (self,), (math.exp(self.data),))
def relu(self):           return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

| 연산 | 결과 | 국소 미분 | 이유 |
|------|------|----------|------|
| `x ** n` | `x^n` | `n * x^(n-1)` | 거듭제곱 미분 공식 |
| `log(x)` | `ln(x)` | `1/x` | 자연로그 미분 |
| `exp(x)` | `e^x` | `e^x` | 지수함수는 미분해도 자기 자신 |
| `relu(x)` | `max(0, x)` | `x > 0이면 1, 아니면 0` | 양수면 통과, 음수면 차단 |

---

## 6. 계산 그래프 — 연산의 지도

`Value` 객체들이 연결되면 **계산 그래프(computation graph)**가 만들어집니다.

예를 들어 `z = x * y + w`를 계산하면:

```python
x = Value(2.0)
y = Value(3.0)
w = Value(1.0)

mul = x * y        # mul.data = 6.0,  children=(x, y), local_grads=(3.0, 2.0)
z   = mul + w      # z.data   = 7.0,  children=(mul, w), local_grads=(1, 1)
```

이를 그래프로 그리면:

```
x(2.0) --[기울기: y=3.0]--> [*](x*y = 6.0) --[기울기: 1]--> [+](z = 7.0)
y(3.0) --[기울기: x=2.0]--> [*](x*y = 6.0)                      ↑
w(1.0) -------------------------------------------[기울기: 1]---→ [+](z = 7.0)
```

순전파(forward pass): 왼쪽에서 오른쪽으로 계산
역전파(backward pass): 오른쪽에서 왼쪽으로 기울기 전달

---

## 7. backward() — 기울기를 자동으로 계산하기

```python
# microgpt.py:59-72
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
```

두 단계로 동작합니다.

### 1단계: 위상 정렬 (Topological Sort)

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children:
            build_topo(child)  # 먼저 자식들을 방문
        topo.append(v)         # 자식들을 모두 방문한 후에 자신을 추가
```

"자식을 부모보다 먼저" 처리하도록 순서를 정합니다. `z = x*y + w` 예시에서:

```
topo 순서: [x, y, (x*y), w, z]
```

역방향으로 처리하면: `z → w → (x*y) → y → x`

### 2단계: 체인룰로 기울기 전파

```python
self.grad = 1  # 출력 노드의 기울기는 1 (loss에 대한 loss의 미분 = 1)
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad
```

핵심 공식:

```
child.grad += local_grad * parent.grad
```

이것이 **체인룰(chain rule)**입니다. 국소 미분과 상위에서 전달된 기울기를 곱해서 하위로 전달합니다. `+=`인 이유는 하나의 노드가 여러 경로로 사용될 수 있기 때문입니다(기울기 누적).

### z = x*y + w 예제를 손으로 계산

x=2, y=3, w=1일 때 z.backward()가 어떻게 동작하는지 추적해 봅시다.

```
초기 상태: x.grad=0, y.grad=0, w.grad=0, mul.grad=0, z.grad=0

step 1: z.grad = 1  (출력 노드는 기울기 1로 시작)

step 2: z 노드 처리 (z = mul + w)
  local_grads = (1, 1)
  mul.grad += 1 * z.grad = 1 * 1 = 1
  w.grad   += 1 * z.grad = 1 * 1 = 1

step 3: w 노드 처리 (w는 leaf, children 없음)
  → 아무것도 안 함

step 4: mul 노드 처리 (mul = x * y)
  local_grads = (y.data, x.data) = (3.0, 2.0)
  x.grad += 3.0 * mul.grad = 3.0 * 1 = 3.0
  y.grad += 2.0 * mul.grad = 2.0 * 1 = 2.0

최종 결과:
  x.grad = 3.0  (dz/dx = y = 3)
  y.grad = 2.0  (dz/dy = x = 2)
  w.grad = 1.0  (dz/dw = 1)
```

수학적 정답과 일치합니다.

---

## 8. 직접 실행해 보기

아래 코드를 실행하면 위 계산을 확인할 수 있습니다.

```python
import math

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

# z = x * y + w
x = Value(2.0)
y = Value(3.0)
w = Value(1.0)
z = x * y + w

print(f"z.data = {z.data}")  # 7.0

z.backward()

print(f"x.grad = {x.grad}")  # 3.0  (dz/dx = y = 3)
print(f"y.grad = {y.grad}")  # 2.0  (dz/dy = x = 2)
print(f"w.grad = {w.grad}")  # 1.0  (dz/dw = 1)
```

### 실행 결과

```
z.data = 7.0
x.grad = 3.0
y.grad = 2.0
w.grad = 1.0
```

---

## 9. 수치 미분으로 검증하기

역전파 결과가 맞는지 확인하는 방법이 있습니다. **수치 미분(finite differences)**입니다.

```
수치 미분 ≈ (f(x + h) - f(x - h)) / (2h)
```

x를 아주 조금(h) 더하고 뺐을 때 결과가 얼마나 변하는지를 측정합니다.

```python
def numerical_grad(f, x, h=1e-5):
    """x에 대한 f의 수치 미분"""
    return (f(x + h) - f(x - h)) / (2 * h)

# z = x*y + w 에서 dz/dx를 수치적으로 계산
# x=2, y=3, w=1 고정
f = lambda x_val: x_val * 3.0 + 1.0  # z = x*y + w = x*3 + 1

numerical = numerical_grad(f, 2.0)
print(f"수치 미분: dz/dx = {numerical:.6f}")  # 3.000000
print(f"역전파:   dz/dx = {x.grad:.6f}")      # 3.000000
print(f"오차: {abs(numerical - x.grad):.2e}") # 매우 작음
```

### 실행 결과

```
수치 미분: dz/dx = 3.000000
역전파:   dz/dx = 3.000000
오차: 0.00e+00
```

두 방법의 결과가 일치합니다. 역전파 구현이 올바르다는 것을 확인했습니다.

---

## 10. 핵심 정리

| 개념 | 한 줄 설명 |
|------|-----------|
| 자동 미분 | 연산할 때마다 "어떻게 미분할지"도 함께 기록해두는 방법 |
| `Value.data` | 순전파에서 계산된 숫자 값 |
| `Value.grad` | 역전파에서 채워지는 기울기 (loss에 대한 민감도) |
| `_local_grads` | 이 연산의 국소 미분 (chain rule의 재료) |
| 계산 그래프 | Value 객체들이 연결된 연산의 지도 |
| 위상 정렬 | "자식을 부모보다 먼저 처리"하는 순서 |
| 체인룰 | `child.grad += local_grad * parent.grad` |
| 수치 미분 | `(f(x+h) - f(x-h)) / (2h)` 로 역전파를 검증하는 방법 |

### 다음 챕터 예고

이제 자동 미분의 엔진을 이해했습니다. 다음 챕터에서는 이 엔진 위에 실제 Transformer 모델이 어떻게 구축되고, 학습되고, 이름을 생성하는지 `microgpt.py` 전체를 따라가며 살펴봅니다.


> **직접 체험하기** — 시각화 도구에서 계산 그래프와 역전파 애니메이션을 직접 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#backprop)

---
[< 이전: Ch03: 왜 학습이 되는가?](ch03-why-learning-works.md) | [목차](../README.md) | [다음: Ch05: Transformer와 학습 >](ch05-transformer-and-training.md)
