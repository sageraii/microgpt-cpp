# Ch13: 데이터 로딩과 토크나이저

## 학습 목표

- Python 3줄의 데이터 로딩이 C++에서 왜 더 많은 코드가 필요한지 이해한다
- `std::ifstream`으로 파일을 읽고, 공백을 제거하는 방법을 직접 작성한다
- `std::mt19937`과 `std::shuffle`로 데이터를 섞는 방법을 배운다
- `std::set<char>`으로 중복 없는 문자 집합을 만드는 이유를 파악한다
- `std::unordered_map<char, int>`으로 문자 → ID 매핑 테이블을 구현한다
- "emma"를 `[26, 4, 12, 12, 0, 26]`로 변환하는 토크나이저를 완성한다

---

## Python 데이터 로딩 복습 (microgpt.py 15~21줄)

```python
# 단 5줄로 데이터 준비 완료
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
```

Python은 간결합니다:
- `urllib.request` — 표준 라이브러리에 HTTP 클라이언트가 내장
- 리스트 컴프리헨션 한 줄로 파일 읽기 + 공백 제거 + 빈 줄 필터링

C++에서는 각 단계를 명시적으로 작성해야 합니다.

---

## 데이터 다운로드

### Python — 표준 HTTP 클라이언트 있음

```python
import urllib.request
urllib.request.urlretrieve(url, 'input.txt')
```

Python 표준 라이브러리에는 HTTP 클라이언트가 내장되어 있습니다.

### C++ — 표준 HTTP 클라이언트 없음

C++17 표준 라이브러리에는 HTTP 클라이언트가 없습니다. 대신 운영체제에 이미 설치된 `curl` 명령을 실행하는 방법을 씁니다.

```cpp
// microgpt.cpp 327~340줄
{
    std::ifstream test("input.txt");   // ①: 파일 열기 시도
    if (!test) {                        // ②: 파일이 없으면
        std::cout << "input.txt 다운로드 중...\n";
        int ret = std::system(          // ③: curl 명령 실행
            "curl -sL -o input.txt "
            "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt");
        if (ret != 0) {                 // ④: 실패 처리
            std::cerr << "다운로드 실패.\n";
            return 1;
        }
    }
}
```

각 줄 설명:

**① `std::ifstream test("input.txt")`**

`std::ifstream`(input file stream)은 파일을 읽기 위한 객체입니다. 생성자에 파일 이름을 전달하면 즉시 파일 열기를 시도합니다.

**② `if (!test)`**

`!test`는 파일 열기에 실패했을 때 `true`입니다. 파일이 존재하지 않거나 읽기 권한이 없을 때 해당됩니다.

```
Python:   if not os.path.exists('input.txt'):
C++:      std::ifstream test("input.txt");
          if (!test) {
```

**③ `std::system(...)`**

`std::system(명령어)`은 셸에서 명령어를 실행합니다. `<cstdlib>` 헤더에 있습니다.

```
curl 옵션:
  -s  : silent (진행 상황 출력 안 함)
  -L  : 리다이렉트 따라가기
  -o  : 저장할 파일 이름 지정
```

**④ 반환값 확인**

`std::system()`은 명령이 성공하면 0을, 실패하면 다른 값을 반환합니다.

```
Python:   urlretrieve()가 실패하면 예외(exception)를 던짐
C++:      ret != 0 이면 실패 → return 1로 프로그램 종료
```

---

## 파일 읽기

### Python — 한 줄

```python
docs = [line.strip() for line in open('input.txt') if line.strip()]
```

### C++ — 명시적 루프

```cpp
// microgpt.cpp 344~353줄
std::vector<std::string> docs;
{
    std::ifstream file("input.txt");
    std::string line;
    while (std::getline(file, line)) {     // ①: 한 줄씩 읽기
        auto start = line.find_first_not_of(" \t\r\n");  // ②: 앞 공백 위치
        auto end   = line.find_last_not_of(" \t\r\n");   // ③: 뒤 공백 위치
        if (start != std::string::npos)    // ④: 빈 줄이 아니면
            docs.push_back(line.substr(start, end - start + 1));  // ⑤: 추가
    }
}
```

각 줄 설명:

**① `std::getline(file, line)`**

파일에서 한 줄을 읽어 `line`에 저장합니다. 줄 끝(`\n`)은 저장하지 않습니다. 더 읽을 줄이 없으면 `false`를 반환하여 `while` 루프가 종료됩니다.

```
Python:   for line in open('input.txt'):   (자동으로 한 줄씩)
C++:      while (std::getline(file, line)) (명시적으로 한 줄씩)
```

**② `find_first_not_of(" \t\r\n")`**

문자열에서 공백 문자(스페이스 `" "`, 탭 `\t`, 캐리지리턴 `\r`, 줄바꿈 `\n`)가 **아닌** 첫 번째 문자의 위치를 반환합니다. 이것이 앞 공백 제거의 시작점입니다.

**③ `find_last_not_of(" \t\r\n")`**

공백 문자가 아닌 **마지막** 문자의 위치입니다. 이것이 뒤 공백 제거의 끝점입니다.

```
"  emma  \r\n"
   ↑         ↑
  start=2   end=5  (0부터 시작)
```

**④ `start != std::string::npos`**

`npos`는 "찾지 못함"을 나타내는 특수값입니다. `start == npos`이면 줄 전체가 공백인 빈 줄입니다.

```
Python:   if line.strip():   (빈 줄이면 빈 문자열 → False)
C++:      if (start != std::string::npos)
```

**⑤ `line.substr(start, end - start + 1)`**

`substr(시작위치, 길이)`는 문자열의 일부를 추출합니다. `end - start + 1`은 앞뒤 공백을 제거한 실제 내용의 길이입니다.

```
line = "  emma  "
start = 2, end = 5
substr(2, 5-2+1) = substr(2, 4) = "emma"
```

```
Python:   line.strip()
C++:      line.substr(start, end - start + 1)
```

**중괄호 블록 `{}`**

`std::ifstream file`은 블록이 끝날 때 자동으로 닫힙니다(RAII 패턴). C++에서 파일을 명시적으로 닫지 않아도 됩니다.

---

## 데이터 섞기

### Python

```python
random.seed(42)
random.shuffle(docs)
```

### C++

```cpp
// microgpt.cpp 355~357줄
std::mt19937 rng(42);                           // ①: 난수 생성기
std::shuffle(docs.begin(), docs.end(), rng);    // ②: 섞기
```

**① `std::mt19937 rng(42)`**

`mt19937`은 메르센 트위스터(Mersenne Twister) 알고리즘을 구현한 난수 생성기입니다. 42가 시드(seed)입니다. 같은 시드를 쓰면 항상 같은 순서의 난수가 나옵니다.

```
Python:   random.seed(42)     → 전역 난수 상태 설정
C++:      std::mt19937 rng(42) → 독립적인 난수 생성기 객체 생성
```

**② `std::shuffle(docs.begin(), docs.end(), rng)`**

`begin()`과 `end()`는 컨테이너의 시작과 끝을 가리키는 반복자입니다. `std::shuffle`은 이 범위 안의 원소들을 `rng`를 이용해 무작위로 섞습니다.

```
Python:   random.shuffle(docs)            (리스트를 제자리에서 섞음)
C++:      std::shuffle(docs.begin(), docs.end(), rng)  (동일한 효과)
```

**주의: 같은 시드여도 결과가 다를 수 있다**

Python과 C++의 `std::mt19937`은 같은 메르센 트위스터 알고리즘을 쓰지만, 셔플 알고리즘의 구현이 다를 수 있습니다. 따라서 `random.seed(42)` 후의 Python 셔플 결과와 `mt19937(42)` 후의 C++ 셔플 결과는 **다를 수 있습니다.** 이는 정상입니다 — 중요한 것은 "재현 가능하게 섞인다"는 특성이지, 두 언어에서 같은 순서가 나오는 것이 아닙니다.

---

## 토크나이저 구현

### Python 토크나이저 (microgpt.py 24~27줄)

```python
uchars = sorted(set(''.join(docs)))  # 고유 문자, 정렬됨
BOS = len(uchars)                    # BOS 토큰 ID
vocab_size = len(uchars) + 1
```

### C++ 토크나이저

**1단계: 고유 문자 집합 만들기**

```cpp
// microgpt.cpp 362~365줄
std::set<char> char_set;            // ①: 자동 정렬되는 집합
for (const auto& doc : docs)
    for (char c : doc)
        char_set.insert(c);         // ②: 문자 추가 (중복 자동 제거)
std::vector<char> uchars(char_set.begin(), char_set.end());  // ③: 벡터로 변환
```

**① `std::set<char> char_set`**

`std::set`은 **자동으로 정렬**되는 집합입니다. 중복을 허용하지 않습니다.

```
Python:   sorted(set(''.join(docs)))
          → set: 중복 제거
          → sorted: 알파벳 순 정렬

C++:      std::set<char>
          → insert: 중복 자동 제거 + 자동 정렬
```

`std::unordered_set`과 달리 `std::set`은 내부적으로 정렬된 이진 탐색 트리를 사용합니다. 그래서 원소가 항상 오름차순으로 유지됩니다.

**② `char_set.insert(c)`**

같은 문자를 여러 번 `insert`해도 집합에는 한 번만 저장됩니다.

**③ `std::vector<char> uchars(char_set.begin(), char_set.end())`**

`set`의 반복자를 이용해 `vector`를 초기화합니다. `set`이 이미 정렬되어 있으므로 `uchars`도 정렬된 상태입니다.

```
Python:   sorted(set(...)) → uchars = ['a', 'b', ..., 'z']
C++:      std::set<char>   → vector 변환 → uchars = {'a', 'b', ..., 'z'}
```

**2단계: BOS 토큰 ID와 vocab_size 계산**

```cpp
// microgpt.cpp 367~368줄
int BOS = static_cast<int>(uchars.size());  // uchars.size() = 26이면 BOS = 26
int vocab_size = BOS + 1;                   // 전체 토큰 수 = 26 + 1 = 27
```

```
Python:                          C++:
BOS = len(uchars)                int BOS = static_cast<int>(uchars.size());
vocab_size = len(uchars) + 1     int vocab_size = BOS + 1;
```

`static_cast<int>(uchars.size())`를 쓰는 이유: `uchars.size()`는 `size_t`(부호 없는 정수)이고, `BOS`는 `int`(부호 있는 정수)입니다. 나중에 `BOS`를 음수와 비교하거나 음수 산술에 쓸 수 있으므로 `int`로 명시적 변환합니다.

**3단계: 문자 → ID 매핑 테이블**

```cpp
// microgpt.cpp 372~374줄
std::unordered_map<char, int> char_to_id;
for (int i = 0; i < static_cast<int>(uchars.size()); i++)
    char_to_id[uchars[i]] = i;
```

```
Python:
  (별도 딕셔너리 없음)
  uchars.index(ch)로 그때그때 찾음  → O(n) 선형 탐색

C++:
  std::unordered_map<char, int> char_to_id
  char_to_id[uchars[i]] = i         → O(1) 해시 탐색
```

`unordered_map`은 해시 테이블 기반이므로 주어진 문자의 ID를 평균 O(1)에 찾습니다. 학습 루프에서 매번 호출되므로 성능이 중요합니다.

사용법:

```cpp
// 문자 'e'의 ID 얻기
int id = char_to_id['e'];   // → 4 (a=0, b=1, c=2, d=3, e=4, ...)

// Python의 uchars.index('e')와 동일
```

**4단계: 토큰화**

```cpp
// microgpt.cpp 428~430줄 (학습 루프 내부)
std::vector<int> tokens;
tokens.push_back(BOS);
for (char c : doc) tokens.push_back(char_to_id[c]);
tokens.push_back(BOS);
```

```
Python:
  tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

C++:
  tokens.push_back(BOS);
  for (char c : doc) tokens.push_back(char_to_id[c]);
  tokens.push_back(BOS);
```

---

## 토크나이저 실습: "emma" 변환

이름 데이터셋의 uchars는 `['a', 'b', 'c', ..., 'z']`입니다.

```
a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8, j=9, k=10,
l=11, m=12, n=13, o=14, p=15, q=16, r=17, s=18, t=19, u=20,
v=21, w=22, x=23, y=24, z=25, BOS=26
```

"emma" 토크나이징 과정:

```
문자  →  ID
BOS  →  26   (시작 신호)
'e'  →   4   (uchars[4])
'm'  →  12   (uchars[12])
'm'  →  12   (uchars[12])
'a'  →   0   (uchars[0])
BOS  →  26   (종료 신호)

결과: [26, 4, 12, 12, 0, 26]
```

GPT는 이 시퀀스를 보고 다음을 학습합니다:

```
위치 0: BOS(26)가 입력 → 'e'(4)를 예측
위치 1: 'e'(4)가 입력  → 'm'(12)을 예측
위치 2: 'm'(12)이 입력 → 'm'(12)을 예측
위치 3: 'm'(12)이 입력 → 'a'(0)를 예측
위치 4: 'a'(0)가 입력  → BOS(26)를 예측 (이름 끝)
```

---

## 컴파일 가능한 전체 테스트 프로그램

```cpp
// ch13_test.cpp — 데이터 로딩과 토크나이저 테스트
//
// 컴파일:
//   g++ -std=c++17 -O2 -o ch13_test ch13_test.cpp
// 실행 (input.txt가 필요 없는 독립 테스트):
//   ./ch13_test

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

bool approx(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// ---- 테스트 1: 파일 읽기 및 공백 제거 ----
void test_file_reading() {
    std::cout << "--- 테스트 1: 파일 읽기 ---\n";

    // 임시 테스트 파일 생성
    {
        std::ofstream f("ch13_test_input.txt");
        f << "emma\n";
        f << "olivia\n";
        f << "  ava  \n";      // 앞뒤 공백
        f << "\n";             // 빈 줄
        f << "isabella\r\n";  // Windows 줄바꿈
        f << "  \t  \n";      // 공백만 있는 줄
        f << "sophia\n";
    }

    // 파일 읽기 (microgpt.cpp와 동일한 로직)
    std::vector<std::string> docs;
    {
        std::ifstream file("ch13_test_input.txt");
        std::string line;
        while (std::getline(file, line)) {
            auto start = line.find_first_not_of(" \t\r\n");
            auto end   = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos)
                docs.push_back(line.substr(start, end - start + 1));
        }
    }

    // 검증
    std::cout << "  읽은 문서 수: " << docs.size() << "  (기대: 5)\n";
    assert(docs.size() == 5);

    std::cout << "  docs[0] = \"" << docs[0] << "\"  (기대: \"emma\")\n";
    assert(docs[0] == "emma");

    std::cout << "  docs[1] = \"" << docs[1] << "\"  (기대: \"olivia\")\n";
    assert(docs[1] == "olivia");

    std::cout << "  docs[2] = \"" << docs[2] << "\"  (기대: \"ava\", 공백 제거)\n";
    assert(docs[2] == "ava");   // 공백이 제거되어야 함

    std::cout << "  docs[3] = \"" << docs[3] << "\"  (기대: \"isabella\", \\r 제거)\n";
    assert(docs[3] == "isabella");   // \r이 제거되어야 함

    std::cout << "  docs[4] = \"" << docs[4] << "\"  (기대: \"sophia\")\n";
    assert(docs[4] == "sophia");

    // 임시 파일 삭제
    std::remove("ch13_test_input.txt");
    std::cout << "  통과!\n\n";
}

// ---- 테스트 2: std::shuffle 재현성 ----
void test_shuffle() {
    std::cout << "--- 테스트 2: 셔플 재현성 ---\n";

    std::vector<std::string> docs = {"alice", "bob", "charlie", "dave", "eve"};
    std::vector<std::string> docs2 = docs;   // 동일한 복사본

    // 같은 시드로 두 번 섞으면 같은 결과
    std::mt19937 rng1(42);
    std::shuffle(docs.begin(), docs.end(), rng1);

    std::mt19937 rng2(42);
    std::shuffle(docs2.begin(), docs2.end(), rng2);

    std::cout << "  첫 번째 섞기:  ";
    for (const auto& d : docs) std::cout << d << " ";
    std::cout << "\n";

    std::cout << "  두 번째 섞기: ";
    for (const auto& d : docs2) std::cout << d << " ";
    std::cout << "\n";

    assert(docs == docs2);   // 동일한 결과여야 함
    std::cout << "  (두 결과가 동일 — 재현 가능)\n";
    std::cout << "  통과!\n\n";
}

// ---- 테스트 3: 토크나이저 구축 ----
void test_tokenizer_build() {
    std::cout << "--- 테스트 3: 토크나이저 구축 ---\n";

    std::vector<std::string> docs = {"emma", "olivia", "ava"};

    // 고유 문자 집합 (std::set은 자동 정렬)
    std::set<char> char_set;
    for (const auto& doc : docs)
        for (char c : doc)
            char_set.insert(c);

    std::vector<char> uchars(char_set.begin(), char_set.end());

    std::cout << "  고유 문자: ";
    for (char c : uchars) std::cout << c;
    std::cout << "\n";

    // a, e, i, l, m, o, v 가 있어야 함 (emma + olivia + ava)
    // std::set이 자동 정렬하므로 순서: a, e, i, l, m, o, v
    assert(uchars[0] == 'a');
    assert(uchars[1] == 'e');
    assert(std::is_sorted(uchars.begin(), uchars.end()));   // 정렬 확인

    int BOS = static_cast<int>(uchars.size());
    int vocab_size = BOS + 1;

    std::cout << "  고유 문자 수: " << uchars.size() << "\n";
    std::cout << "  BOS ID: " << BOS << "\n";
    std::cout << "  vocab_size: " << vocab_size << "\n";
    assert(vocab_size == static_cast<int>(uchars.size()) + 1);

    // char_to_id 구축
    std::unordered_map<char, int> char_to_id;
    for (int i = 0; i < static_cast<int>(uchars.size()); i++)
        char_to_id[uchars[i]] = i;

    // uchars[i]로 찾은 ID와 char_to_id로 찾은 ID가 일치해야 함
    for (int i = 0; i < static_cast<int>(uchars.size()); i++) {
        assert(char_to_id[uchars[i]] == i);
    }
    std::cout << "  char_to_id 매핑 정합성: OK\n";
    std::cout << "  통과!\n\n";
}

// ---- 테스트 4: 이름 데이터셋 토크나이저로 "emma" 변환 ----
void test_encode_emma() {
    std::cout << "--- 테스트 4: \"emma\" 토크나이징 ---\n";

    // 이름 데이터셋의 uchars: a~z 26개 문자
    std::vector<char> uchars;
    for (char c = 'a'; c <= 'z'; c++) uchars.push_back(c);
    // uchars = ['a', 'b', ..., 'z']

    int BOS = static_cast<int>(uchars.size());   // = 26

    std::unordered_map<char, int> char_to_id;
    for (int i = 0; i < static_cast<int>(uchars.size()); i++)
        char_to_id[uchars[i]] = i;

    // "emma" 인코딩
    std::string doc = "emma";
    std::vector<int> tokens;
    tokens.push_back(BOS);
    for (char c : doc) tokens.push_back(char_to_id[c]);
    tokens.push_back(BOS);

    // 결과 출력
    std::cout << "  encode(\"emma\") = [";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]\n";
    std::cout << "  기대:            [26, 4, 12, 12, 0, 26]\n";

    assert(tokens.size() == 6);
    assert(tokens[0] == 26);   // BOS
    assert(tokens[1] ==  4);   // 'e'
    assert(tokens[2] == 12);   // 'm'
    assert(tokens[3] == 12);   // 'm'
    assert(tokens[4] ==  0);   // 'a'
    assert(tokens[5] == 26);   // BOS

    // 디코딩: 토큰 → 문자열 (BOS 제외)
    std::string decoded;
    for (int t : tokens) {
        if (t != BOS)
            decoded += uchars[t];
    }
    std::cout << "  decode([26, 4, 12, 12, 0, 26]) = \"" << decoded << "\"\n";
    assert(decoded == "emma");

    std::cout << "  통과!\n\n";
}

// ---- 테스트 5: std::set vs std::unordered_set 정렬 비교 ----
void test_set_vs_unordered_set() {
    std::cout << "--- 테스트 5: std::set 자동 정렬 확인 ---\n";

    // 순서 없이 삽입
    std::set<char> sorted_set;
    sorted_set.insert('z');
    sorted_set.insert('a');
    sorted_set.insert('m');
    sorted_set.insert('b');

    std::vector<char> result(sorted_set.begin(), sorted_set.end());

    std::cout << "  삽입 순서: z, a, m, b\n";
    std::cout << "  set 결과: ";
    for (char c : result) std::cout << c << " ";
    std::cout << "\n";
    std::cout << "  (항상 알파벳 순: a b m z)\n";

    assert(result[0] == 'a');
    assert(result[1] == 'b');
    assert(result[2] == 'm');
    assert(result[3] == 'z');
    assert(std::is_sorted(result.begin(), result.end()));

    std::cout << "  통과!\n\n";
}

int main() {
    std::cout << "=== Ch13 데이터 로딩과 토크나이저 테스트 ===\n\n";
    test_file_reading();
    test_shuffle();
    test_tokenizer_build();
    test_encode_emma();
    test_set_vs_unordered_set();
    std::cout << "모든 테스트 통과!\n";
    return 0;
}
```

### 컴파일과 실행

```bash
g++ -std=c++17 -O2 -o ch13_test ch13_test.cpp
./ch13_test
```

예상 출력:

```
=== Ch13 데이터 로딩과 토크나이저 테스트 ===

--- 테스트 1: 파일 읽기 ---
  읽은 문서 수: 5  (기대: 5)
  docs[0] = "emma"  (기대: "emma")
  docs[1] = "olivia"  (기대: "olivia")
  docs[2] = "ava"  (기대: "ava", 공백 제거)
  docs[3] = "isabella"  (기대: "isabella", \r 제거)
  docs[4] = "sophia"  (기대: "sophia")
  통과!

--- 테스트 2: 셔플 재현성 ---
  첫 번째 섞기:  charlie eve alice dave bob
  두 번째 섞기: charlie eve alice dave bob
  (두 결과가 동일 — 재현 가능)
  통과!

--- 테스트 3: 토크나이저 구축 ---
  고유 문자: aeilmov
  고유 문자 수: 7
  BOS ID: 7
  vocab_size: 8
  char_to_id 매핑 정합성: OK
  통과!

--- 테스트 4: "emma" 토크나이징 ---
  encode("emma") = [26, 4, 12, 12, 0, 26]
  기대:            [26, 4, 12, 12, 0, 26]
  decode([26, 4, 12, 12, 0, 26]) = "emma"
  통과!

--- 테스트 5: std::set 자동 정렬 확인 ---
  삽입 순서: z, a, m, b
  set 결과: a b m z
  (항상 알파벳 순: a b m z)
  통과!

모든 테스트 통과!
```

---

## Python vs C++ 전체 비교 요약

```
[데이터 다운로드]
Python:  urllib.request.urlretrieve(url, 'input.txt')   (1줄)
C++:     std::ifstream test("input.txt");
         if (!test) std::system("curl -sL -o input.txt " + url);  (4줄)

[파일 읽기 + 공백 제거]
Python:  docs = [line.strip() for line in open('input.txt') if line.strip()]  (1줄)
C++:     while (std::getline(file, line)) {         (10줄)
             auto start = line.find_first_not_of(...)
             auto end   = line.find_last_not_of(...)
             if (start != npos) docs.push_back(...)
         }

[섞기]
Python:  random.seed(42); random.shuffle(docs)       (2줄)
C++:     std::mt19937 rng(42);                       (2줄)
         std::shuffle(docs.begin(), docs.end(), rng)

[고유 문자 집합]
Python:  uchars = sorted(set(''.join(docs)))         (1줄)
C++:     std::set<char> char_set;                    (6줄)
         for (doc) for (c) char_set.insert(c);
         std::vector<char> uchars(char_set.begin(), char_set.end())

[문자 → ID 변환]
Python:  uchars.index(ch)  (O(n) 선형 탐색)
C++:     char_to_id[ch]    (O(1) 해시 탐색)
```

---

## 핵심 정리

| 개념 | Python | C++ |
|------|--------|-----|
| HTTP 다운로드 | `urllib.request` (내장) | `std::system("curl ...")` |
| 파일 읽기 | `open('input.txt')` | `std::ifstream file("input.txt")` |
| 한 줄 읽기 | `for line in file:` | `std::getline(file, line)` |
| 공백 제거 | `line.strip()` | `find_first/last_not_of` + `substr` |
| 빈 줄 필터 | `if line.strip()` | `if (start != std::string::npos)` |
| 난수 생성기 | `random.seed(42)` | `std::mt19937 rng(42)` |
| 셔플 | `random.shuffle(docs)` | `std::shuffle(docs.begin(), docs.end(), rng)` |
| 정렬된 집합 | `sorted(set(...))` | `std::set<char>` |
| 문자→ID 매핑 | `dict` / `list.index()` | `std::unordered_map<char, int>` |
| "찾지 못함" | `-1` (index 반환) | `std::string::npos` |

C++의 주요 컨테이너 선택 기준:

```
std::set<T>             → 정렬된 고유 원소 집합 (순서 중요할 때)
std::unordered_set<T>   → 정렬 불필요, 빠른 존재 확인 (Ch11 역전파에서 사용)
std::unordered_map<K,V> → 빠른 키→값 조회 (평균 O(1))
std::vector<T>          → 순서 있는 배열, 인덱스 접근
```

다음 챕터에서는 여기서 만든 데이터, 토크나이저, 그리고 Ch10~12에서 만든 신경망 빌딩 블록을 연결하여 **GPT 전체 순전파와 학습 루프**를 완성합니다.


> **직접 체험하기** — 시각화 도구에서 데이터셋 통계와 토크나이저를 인터랙티브하게 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#tokenizer)

---
[< 이전: Ch12: 신경망 빌딩 블록](ch12-nn-building-blocks.md) | [목차](../README.md) | [다음: Ch14: 파라미터 초기화 >](ch14-parameters.md)
