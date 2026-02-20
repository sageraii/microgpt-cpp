# Ch09: Graph 아레나

## 학습 목표

- 순전파 중 수천 개의 임시 `Value` 노드가 생기는 문제를 파악한다
- Python의 가비지 컬렉터와 달리 C++에서 직접 해제해야 함을 이해한다
- **아레나 패턴(Arena Pattern)**이 이 문제를 어떻게 해결하는지 납득한다
- `Graph` 구조체의 `make()`, `clear()`, `~Graph()` 각각의 역할을 안다
- 파라미터 노드(수명: 프로그램 전체)와 계산 그래프 노드(수명: 한 스텝)의 차이를 구분한다
- 직접 아레나를 만들어 10개의 `Value`를 생성하고 `clear()`로 해제한다

---

## 개념 설명

### 문제: 누가 임시 노드를 해제하나?

GPT의 순전파(forward pass) 한 번에 얼마나 많은 `Value` 노드가 만들어질까요?

토큰 하나를 처리하는 동안 `linear`, `rmsnorm`, `softmax` 등의 연산이 수백 번 일어납니다. 레이어 수, 임베딩 차원, 시퀀스 길이를 곱하면 **수천에서 수만 개의 임시 노드**가 생깁니다.

Python에서는 걱정이 없습니다.

```python
# Python: 스텝이 끝나면 가비지 컬렉터가 알아서 해제
for step in range(1000):
    loss = forward(...)   # 수천 개의 Value 노드 생성
    loss.backward()
    update_params()
    # 스텝이 끝나면? → Python GC가 알아서 청소
```

C++에서는 직접 해제해야 합니다.

```cpp
// C++ (순진한 방식, 사용하지 않음): 어떻게 해제하지?
for (int step = 0; step < 1000; step++) {
    Value* loss = forward(...);  // 수천 개의 new Value{...} 발생
    backward(loss);
    update_params();
    // 여기서 수천 개를 어떻게 찾아서 delete 할까?
    // → 불가능! 임시 노드들의 주소를 따로 저장해 두지 않으면
}
```

임시 노드들은 `add()`, `mul()` 등의 내부에서 `new`로 만들어집니다. 반환된 포인터는 체이닝 과정에서 바로 다음 함수로 넘어갑니다. 최종 `loss` 포인터만 남고, 중간 노드들의 주소는 어디에도 기록되어 있지 않습니다.

```
add(mul(x, y), x)
    └→ new Value{6.0} ← 이 포인터가 어딘가 저장되지 않으면
                          나중에 delete할 방법이 없다!
```

---

### 해결책: 아레나 패턴

**아레나(Arena)**는 "모든 임시 노드를 한 곳에 등록해 두고, 나중에 한꺼번에 해제"하는 패턴입니다.

비유: 요리사의 작업대

```
요리를 시작한다 (순전파 시작)
  → 재료, 중간 결과물을 작업대 위에 올린다 (노드 생성 + 아레나 등록)
  → 완성된 요리가 나온다 (loss 계산)

요리를 평가한다 (역전파)
  → 작업대 위의 물건들을 모두 치운다 (arena.clear())

다음 요리를 시작한다 (다음 스텝)
  → 작업대가 비어 있으니 새 재료를 올린다
```

아레나 패턴의 핵심: **`new`로 만드는 순간 아레나에 등록**, 끝나면 **아레나째 한꺼번에 해제**.

---

## `Graph` 구조체 구현

### 전체 코드 (microgpt.cpp:51-71)

```cpp
// 그래프 아레나: 순전파 중 생성되는 모든 임시 Value 노드를 소유한다.
// 학습 스텝 사이에 clear()로 계산 그래프를 해제한다.
// 모델 파라미터는 별도로 할당되어 스텝 간 유지된다.
struct Graph {
    std::vector<Value*> nodes;   // 등록된 모든 임시 노드의 포인터

    // 새 Value 노드를 생성하고 아레나에 등록
    Value* make(double data, std::vector<Value*> children = {},
                std::vector<double> local_grads = {}) {
        auto* v = new Value{data, 0.0, std::move(children), std::move(local_grads)};
        nodes.push_back(v);   // 아레나에 등록!
        return v;
    }

    // 아레나의 모든 임시 노드를 해제 (파라미터는 영향 없음)
    void clear() {
        for (auto* v : nodes) delete v;
        nodes.clear();
    }

    ~Graph() { clear(); }   // 소멸자: Graph 객체 소멸 시 자동으로 clean
};

static Graph graph; // 현재 스텝의 계산 그래프를 위한 전역 아레나
```

### `make()` — 생성과 등록을 한 번에

```cpp
Value* make(double data, std::vector<Value*> children = {},
            std::vector<double> local_grads = {}) {
    auto* v = new Value{data, 0.0, std::move(children), std::move(local_grads)};
    nodes.push_back(v);
    return v;
}
```

- `new Value{...}` — 힙에 `Value` 생성
- `nodes.push_back(v)` — 아레나 목록에 등록 (이게 핵심!)
- `return v` — 포인터 반환 (연산 체이닝에 사용)

`std::move(...)`는 벡터를 복사하지 않고 **이동**시킵니다. 벡터를 복사하면 원소를 하나씩 새로 만드는데, 이동하면 내부 포인터만 옮겨집니다. 훨씬 빠릅니다.

`make()`를 쓰면 `add()` 구현이 이렇게 됩니다:

```cpp
// microgpt.cpp:78-80
inline Value* add(Value* a, Value* b) {
    return graph.make(a->data + b->data, {a, b}, {1.0, 1.0});
    //               ────────────────── ─────── ───────────
    //               계산된 값          자식들   국소 미분값
}
```

`new Value{...}`가 아니라 `graph.make(...)`를 씁니다. 생성과 동시에 아레나에 등록됩니다.

### `clear()` — 한꺼번에 해제

```cpp
void clear() {
    for (auto* v : nodes) delete v;  // 등록된 모든 노드 해제
    nodes.clear();                    // 벡터 자체를 비움
}
```

- `for (auto* v : nodes)` — 범위 기반 for 루프. `nodes`의 모든 포인터를 순회합니다.
- `delete v` — 각 노드의 힙 메모리 해제
- `nodes.clear()` — 벡터를 비워서 다음 스텝을 위한 새 공간 준비

학습 루프에서는 이렇게 씁니다:

```cpp
// microgpt.cpp:472
graph.clear();  // 스텝 끝: 임시 노드 전부 해제
```

### `~Graph()` — RAII 소멸자

```cpp
~Graph() { clear(); }
```

`~Graph()`는 **소멸자(destructor)**입니다. `Graph` 객체가 소멸될 때 자동으로 호출됩니다. `clear()`를 직접 호출하지 않더라도, 프로그램이 끝나면 전역 `graph` 객체가 소멸되면서 자동으로 메모리가 해제됩니다.

이 패턴을 **RAII (Resource Acquisition Is Initialization)**라고 합니다. "자원(메모리)의 수명을 객체의 수명에 묶는다"는 C++의 핵심 이디엄입니다.

```
RAII 흐름:
  Graph graph 선언 → 생성자 호출 (nodes 벡터 초기화)
  graph.make(...)  → 노드 등록
  graph.clear()    → 스텝 끝마다 청소
  프로그램 종료    → ~Graph() 자동 호출 → clear() → 메모리 해제
```

---

## 두 종류의 메모리 수명

`microgpt.cpp`에는 서로 다른 수명을 가진 두 종류의 `Value` 노드가 있습니다.

```
메모리 수명 비교
──────────────────────────────────────────────────────────
종류          | 수명              | 관리 방법
──────────────┼───────────────────┼──────────────────────
파라미터 노드  | 프로그램 전체     | new로 직접 생성,
(가중치)      | (1000 스텝 내내)  | 끝에서 delete
──────────────┼───────────────────┼──────────────────────
계산 그래프   | 한 학습 스텝       | graph.make()로 생성,
임시 노드     | (순전파+역전파)    | 스텝마다 graph.clear()
──────────────────────────────────────────────────────────
```

**파라미터 노드 (아레나 밖):**

```cpp
// microgpt.cpp:386-390
auto make_matrix = [&](int nout, int nin) -> Mat {
    Mat m(nout, Vec(nin));
    for (int i = 0; i < nout; i++)
        for (int j = 0; j < nin; j++) {
            auto* v = new Value{normal(rng), 0.0, {}, {}};  // 직접 new
            m[i][j] = v;
            params.push_back(v);   // params 목록에 등록 (아레나가 아님)
        }
    return m;
};
```

파라미터는 `graph.make()`를 쓰지 않습니다. 직접 `new`로 만들고 `params` 벡터에 등록합니다. `graph.clear()`를 불러도 파라미터는 영향을 받지 않습니다.

**계산 그래프 노드 (아레나 안):**

```cpp
// microgpt.cpp:78-80 (연산 함수들)
inline Value* add(Value* a, Value* b) {
    return graph.make(a->data + b->data, {a, b}, {1.0, 1.0}); // 아레나에 등록
}
```

순전파 중에 생기는 모든 임시 노드는 `graph.make()`를 통해 아레나에 등록됩니다.

**시각화:**

```
[학습 루프 한 스텝]

파라미터 메모리 (독립, 유지됨)
┌─────────────────────────────────────────┐
│  wte[0][0]  wte[0][1]  ...  lm_head[...]│  ← 1000 스텝 내내 살아 있음
└─────────────────────────────────────────┘

아레나 (스텝마다 생성 → 해제)
┌─────────────────────────────────────────┐
│ 순전파 시작 → node1, node2, ..., nodeN  │
│              (수천 개의 임시 노드)       │
│ 역전파 완료 → graph.clear() → 전부 해제 │
└─────────────────────────────────────────┘
```

---

## `static Graph graph` — 전역 아레나

```cpp
// microgpt.cpp:71
static Graph graph;
```

`static`이 파일 최상위 레벨에 쓰이면 "이 파일 안에서만 보이는 전역 변수"를 의미합니다. `graph`는 프로그램 전체에서 하나만 존재합니다.

모든 연산 함수(`add`, `mul`, `log`, ...)가 이 전역 `graph`를 사용합니다. 함수를 호출할 때마다 아레나에 접근하기 위해 `graph`를 인수로 전달할 필요가 없습니다.

---

## 완전한 실습 코드

아레나 패턴을 직접 만들고 10개의 `Value`를 생성한 뒤 `clear()`로 해제하는 실습입니다.

```cpp
// arena_demo.cpp
// Graph 아레나 패턴 실습:
//   - make()로 10개의 Value 생성 (자동 등록)
//   - clear()로 한꺼번에 해제
//   - 소멸자(RAII)로 자동 정리

#include <iostream>
#include <vector>

// ─── Value 구조체 ────────────────────────────────────────────
struct Value {
    double data;
    double grad = 0.0;
    std::vector<Value*> children;
    std::vector<double> local_grads;
};

// ─── Graph 아레나 ────────────────────────────────────────────
struct Graph {
    std::vector<Value*> nodes;

    // 생성과 등록을 한 번에
    Value* make(double data,
                std::vector<Value*> children = {},
                std::vector<double> local_grads = {}) {
        auto* v = new Value{data, 0.0,
                            std::move(children),
                            std::move(local_grads)};
        nodes.push_back(v);
        std::cout << "  [make] data=" << data
                  << "  아레나 크기=" << nodes.size() << "\n";
        return v;
    }

    // 한꺼번에 해제
    void clear() {
        std::cout << "  [clear] " << nodes.size() << "개 노드 해제\n";
        for (auto* v : nodes) delete v;
        nodes.clear();
    }

    // RAII 소멸자
    ~Graph() {
        if (!nodes.empty()) {
            std::cout << "  [~Graph] 소멸자 호출, "
                      << nodes.size() << "개 노드 자동 해제\n";
            clear();
        }
    }
};

// 전역 아레나 (프로그램 전체에서 하나)
static Graph graph;

// ─── 자유 함수: 아레나를 통해 노드 생성 ──────────────────────
Value* add(Value* a, Value* b) {
    return graph.make(a->data + b->data, {a, b}, {1.0, 1.0});
}

Value* mul(Value* a, Value* b) {
    return graph.make(a->data * b->data, {a, b}, {b->data, a->data});
}

int main() {
    std::cout << "=== 파라미터 노드 (아레나 밖) ===\n";
    // 파라미터는 new로 직접 생성 (아레나에 등록되지 않음)
    Value* x = new Value{2.0, 0.0, {}, {}};
    Value* y = new Value{3.0, 0.0, {}, {}};
    std::cout << "파라미터 생성: x=" << x->data << ", y=" << y->data << "\n\n";

    std::cout << "=== 스텝 1: 순전파 ===\n";
    // 계산 그래프 노드는 graph.make()로 생성 (아레나에 등록)
    Value* t1 = mul(x, y);             // x * y = 6.0  → 아레나 등록
    Value* t2 = add(t1, x);            // t1 + x = 8.0 → 아레나 등록
    Value* t3 = mul(t2, graph.make(2.0)); // t2 * 2.0  → 아레나 등록 (상수도 make로)
    std::cout << "loss = " << t3->data << "\n\n";  // 16.0

    std::cout << "=== 스텝 1: 역전파 후 그래프 해제 ===\n";
    graph.clear();  // 임시 노드 4개 해제 (파라미터 x, y는 영향 없음)

    std::cout << "\n파라미터는 살아 있음: x=" << x->data
              << ", y=" << y->data << "\n\n";

    std::cout << "=== 스텝 2: 새 순전파 ===\n";
    // 아레나가 비어 있으므로 새 그래프 구성 가능
    Value* s1 = add(x, y);             // x + y = 5.0 → 새 아레나에 등록
    std::cout << "step2 result = " << s1->data << "\n\n";

    std::cout << "=== 스텝 2 해제 ===\n";
    graph.clear();

    std::cout << "\n=== 아레나 추가 생성으로 10개 만들기 ===\n";
    for (int i = 0; i < 10; i++) {
        graph.make(static_cast<double>(i));
    }
    std::cout << "아레나에 " << graph.nodes.size() << "개 있음\n";
    graph.clear();

    std::cout << "\n=== 파라미터 정리 ===\n";
    delete x;
    delete y;
    std::cout << "파라미터 delete 완료\n";

    std::cout << "\n=== 프로그램 종료: ~Graph() 자동 호출 ===\n";
    // main 함수가 끝나면 전역 graph 소멸 → ~Graph() → clear() 자동 호출
    // (지금은 이미 비어 있으므로 출력 없음)

    return 0;
}
```

**컴파일 및 실행:**

```bash
g++ -std=c++17 -O2 -o arena_demo arena_demo.cpp && ./arena_demo
```

**실행 결과:**
```
=== 파라미터 노드 (아레나 밖) ===
파라미터 생성: x=2, y=3

=== 스텝 1: 순전파 ===
  [make] data=6  아레나 크기=1
  [make] data=8  아레나 크기=2
  [make] data=2  아레나 크기=3
  [make] data=16  아레나 크기=4
loss = 16

=== 스텝 1: 역전파 후 그래프 해제 ===
  [clear] 4개 노드 해제

파라미터는 살아 있음: x=2, y=3

=== 스텝 2: 새 순전파 ===
  [make] data=5  아레나 크기=1
step2 result = 5

=== 스텝 2 해제 ===
  [clear] 1개 노드 해제

=== 아레나 추가 생성으로 10개 만들기 ===
  [make] data=0  아레나 크기=1
  [make] data=1  아레나 크기=2
  ...
  [make] data=9  아레나 크기=10
아레나에 10개 있음
  [clear] 10개 노드 해제

=== 파라미터 정리 ===
파라미터 delete 완료

=== 프로그램 종료: ~Graph() 자동 호출 ===
```

`clear()` 후에도 파라미터 `x`, `y`가 살아 있음을 확인하세요. 아레나는 자기가 `make()`로 만든 것만 해제합니다.

---

## 메모리 누수 검사 (valgrind)

메모리 누수가 없는지 검증하려면 `valgrind`를 쓸 수 있습니다.

```bash
# valgrind 설치 (Ubuntu)
sudo apt install valgrind

# 메모리 누수 검사
valgrind --leak-check=full ./arena_demo
```

**정상 출력:**
```
LEAK SUMMARY:
   definitely lost: 0 bytes in 0 blocks
   indirectly lost: 0 bytes in 0 blocks
```

`definitely lost: 0`이면 메모리 누수가 없는 것입니다.

---

## 실제 microgpt.cpp의 학습 루프와 연결

```cpp
// microgpt.cpp:420-473 (핵심 부분 발췌)
for (int step = 0; step < num_steps; step++) {

    // 순전파: graph.make()가 내부적으로 수천 번 호출됨
    //         모든 임시 노드가 graph.nodes에 등록됨
    Vec logits = gpt(token_id, pos, keys, vals, state_dict);
    Vec probs  = softmax(logits);
    Value* loss = scale(total_loss, 1.0 / n);  // graph.make() 내부 호출

    // 역전파
    backward(loss);

    // 파라미터 업데이트 (파라미터는 아레나 밖, 영향 없음)
    for (size_t i = 0; i < params.size(); i++) {
        params[i]->data -= /* Adam 업데이트 */;
        params[i]->grad = 0.0;
    }

    // 아레나 해제: 임시 노드만 삭제, 파라미터는 그대로
    graph.clear();  // ← 스텝마다 호출
}
```

이 `graph.clear()` 한 줄이 수천 개의 임시 노드를 한 번에 해제합니다.

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 아레나 패턴 | `make()`로 등록, `clear()`로 한꺼번에 해제 |
| `graph.make(data, children, local_grads)` | `new Value` + `nodes.push_back` |
| `graph.clear()` | 아레나의 모든 노드 `delete` + `nodes.clear()` |
| `~Graph()` | 소멸자 — Graph 객체 소멸 시 자동으로 `clear()` 호출 |
| RAII | 자원 수명을 객체 수명에 묶는 C++ 핵심 패턴 |
| `static Graph graph` | 전역 아레나 — 프로그램 전체에서 하나만 존재 |
| 파라미터 수명 | 프로그램 전체 — `new`로 생성, 끝에서 `delete` |
| 계산 그래프 수명 | 한 스텝 — `graph.make()`, 스텝마다 `graph.clear()` |

다음 챕터에서는 아레나를 활용해 `add`, `mul`, `log`, `exp`, `relu` 등의 **미분 가능한 연산**들을 구현합니다.


---
[< 이전: Ch08: Value 구조체](ch08-value-struct.md) | [목차](../README.md) | [다음: Ch10: 미분 가능한 연산 >](ch10-differentiable-ops.md)
