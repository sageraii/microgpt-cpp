# Ch06: C++ 환경 설정

## 학습 목표

- g++ 컴파일러가 설치되어 있는지 확인하고 C++17 지원 여부를 검증한다
- `std::cout`으로 화면에 글자를 출력하는 첫 번째 C++ 프로그램을 작성한다
- `g++` 명령어로 직접 컴파일하고 실행하는 방법을 익힌다
- 이 튜토리얼에서 사용하는 C++ 표준 라이브러리 헤더들을 파악한다
- CMake로 빌드 시스템을 구성하고 Hello World를 빌드한다

---

## 개념 설명

### C++란? Python과 무엇이 다른가?

Part 1에서 Python으로 GPT를 만들었습니다. 이제 같은 것을 C++로 만듭니다. 왜?

```
Python                          C++
─────────────────────────────   ─────────────────────────────
"편한 자동차"                    "수동 변속기 스포츠카"
코드 작성이 쉽다                 코드 작성이 까다롭다
느리다 (인터프리터)              빠르다 (네이티브 기계어)
메모리 자동 관리                 메모리를 직접 관리
외부 라이브러리로 배포            단일 실행 파일로 배포
```

Python은 코드를 한 줄씩 읽어 실행하는 **인터프리터** 방식입니다. C++는 먼저 전체 코드를 **기계어**로 번역(컴파일)한 뒤 실행합니다. 이 차이 때문에 C++가 훨씬 빠릅니다.

```
[Python 방식]
.py 파일 → Python 인터프리터 → 실행 (매번 해석)

[C++ 방식]
.cpp 파일 → g++ 컴파일러 → 실행 파일(.exe/.out) → 실행 (이미 기계어)
```

### 이 튜토리얼의 C++: 외부 의존성 제로

중요한 약속이 있습니다. **이 튜토리얼에서 사용하는 것은 오직 C++ 표준 라이브러리뿐입니다!**

PyTorch, TensorFlow, Eigen 같은 외부 라이브러리를 전혀 사용하지 않습니다. g++ 한 가지만 있으면 됩니다. Python 파트에서 `os`, `math`, `random` 세 가지만 썼던 것과 같은 철학입니다.

---

## 코드 작성

### 1단계: g++ 설치 확인

터미널을 열고 다음을 입력합니다.

```bash
g++ --version
```

**실행 결과 예시:**
```
g++ (Ubuntu 13.2.0-23ubuntu4) 13.2.0
Copyright (C) 2023 Free Software Foundation, Inc.
```

버전 숫자가 나오면 성공입니다. C++17은 g++ 7 이상에서 지원합니다. g++ 10 이상을 권장합니다.

**g++가 없다면?**

```bash
# Ubuntu / Debian
sudo apt install g++

# macOS (Xcode 커맨드라인 도구)
xcode-select --install

# macOS (Homebrew)
brew install gcc
```

C++17이 지원되는지 직접 확인하려면:

```bash
echo '#include <optional>' | g++ -std=c++17 -x c++ - -o /dev/null && echo "C++17 OK"
```

```
C++17 OK
```

---

### 2단계: 첫 번째 프로그램 — Hello World

새 파일 `hello.cpp`를 만들고 다음을 입력합니다.

```cpp
// hello.cpp
// C++ 최초의 프로그램: 화면에 글자를 출력한다

#include <iostream>   // std::cout을 사용하기 위한 헤더

int main() {
    std::cout << "hello, GPT!" << std::endl;
    return 0;
}
```

**한 줄씩 읽기:**

- `// ...` — 슬래시 두 개 뒤는 **주석**입니다. Python의 `# ...`과 같습니다.
- `#include <iostream>` — `iostream`(입출력 스트림) 헤더를 포함합니다. Python의 `import`와 비슷합니다.
- `int main()` — 프로그램의 시작점입니다. C++는 반드시 `main` 함수에서 시작합니다.
- `std::cout` — 화면으로 출력하는 스트림입니다. Python의 `print`에 해당합니다.
- `<<` — 스트림에 데이터를 밀어 넣는 연산자입니다. "화살표 방향으로 흘러 들어간다"고 읽으면 됩니다.
- `std::endl` — 줄바꿈을 출력합니다. Python의 `\n`에 해당합니다.
- `return 0;` — 프로그램이 정상 종료됐음을 운영체제에 알립니다.
- `;` — 문장의 끝. C++에서는 모든 문장 끝에 세미콜론이 필요합니다.

**Python과 비교:**

```python
# Python
print("hello, GPT!")
```

```cpp
// C++
std::cout << "hello, GPT!" << std::endl;
```

---

### 3단계: 컴파일과 실행

C++는 실행 전에 반드시 **컴파일** 과정이 필요합니다.

```bash
g++ -std=c++17 -O2 -o hello hello.cpp
```

**플래그 설명:**

| 플래그 | 의미 |
|--------|------|
| `-std=c++17` | C++17 표준 사용 (이 튜토리얼의 필수 요건) |
| `-O2` | 최적화 수준 2 (적당히 빠른 코드 생성) |
| `-o hello` | 출력 파일 이름을 `hello`로 지정 |
| `hello.cpp` | 컴파일할 소스 파일 |

컴파일 성공 후 실행:

```bash
./hello
```

**실행 결과:**
```
hello, GPT!
```

컴파일과 실행을 한 줄에:

```bash
g++ -std=c++17 -O2 -o hello hello.cpp && ./hello
```

`&&`는 "앞 명령이 성공하면 뒷 명령을 실행"하는 의미입니다. 컴파일 오류가 나면 실행은 자동으로 건너뜁니다.

---

### 4단계: 표준 라이브러리 헤더 소개

C++ 표준 라이브러리는 수십 개의 헤더로 나뉩니다. `microgpt.cpp`에서 사용하는 것들을 미리 파악해 둡시다.

```cpp
// microgpt.cpp 상단의 헤더들
#include <algorithm>     // std::max, std::min, std::shuffle
#include <cassert>       // assert() — 조건 검사 (디버깅용)
#include <cmath>         // std::sqrt, std::pow, std::log, std::exp
#include <cstdio>        // std::printf — C 스타일 포맷 출력
#include <cstdlib>       // std::system — 시스템 명령 실행
#include <fstream>       // std::ifstream — 파일 읽기
#include <functional>    // std::function — 함수 객체, 람다 저장
#include <iostream>      // std::cout, std::cerr
#include <random>        // std::mt19937, std::normal_distribution
#include <set>           // std::set — 정렬된 집합
#include <string>        // std::string — 문자열
#include <unordered_map> // std::unordered_map — 해시맵 (Python의 dict)
#include <unordered_set> // std::unordered_set — 해시 집합
#include <vector>        // std::vector — 동적 배열 (Python의 list)
```

각각 Python 대응:

| C++ 헤더 | 주요 기능 | Python 대응 |
|----------|----------|------------|
| `<vector>` | 동적 배열 | `list` |
| `<string>` | 문자열 | `str` |
| `<unordered_map>` | 해시맵 | `dict` |
| `<cmath>` | 수학 함수 | `import math` |
| `<random>` | 난수 생성 | `import random` |
| `<fstream>` | 파일 입출력 | `open(...)` |
| `<algorithm>` | 정렬, 탐색 등 | 내장 함수들 |

지금 당장 다 외울 필요는 없습니다. 각 챕터에서 실제로 사용할 때 다시 설명합니다.

---

### 5단계: CMake 소개

프로젝트가 커지면 `g++` 명령을 직접 타이핑하기 어려워집니다. **CMake**는 빌드 설정을 파일로 관리하는 도구입니다.

이미 프로젝트 루트에 `CMakeLists.txt`가 있습니다. 내용을 살펴봅시다.

```cmake
# CMakeLists.txt — microgpt 프로젝트의 빌드 설정

cmake_minimum_required(VERSION 3.16)   # CMake 최소 버전 요구
project(microgpt LANGUAGES CXX)        # 프로젝트 이름과 사용 언어

set(CMAKE_CXX_STANDARD 17)             # C++17 사용
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # 반드시 C++17이어야 함
set(CMAKE_CXX_EXTENSIONS OFF)         # 컴파일러 확장 기능 사용 안 함

add_executable(microgpt microgpt.cpp) # microgpt.cpp로 실행 파일 생성
```

**Hello World용 CMake 실습:**

실습을 위해 임시 디렉토리를 만들어 봅시다.

```bash
mkdir hello_cmake
cd hello_cmake
```

`hello_cmake/CMakeLists.txt` 작성:

```cmake
cmake_minimum_required(VERSION 3.16)
project(hello LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hello hello.cpp)
```

`hello_cmake/hello.cpp` 작성:

```cpp
// hello.cpp
#include <iostream>

int main() {
    std::cout << "hello, GPT!" << std::endl;
    return 0;
}
```

**CMake로 빌드:**

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

**실행 결과:**
```
-- The CXX compiler identification is GNU 13.2.0
-- Detecting CXX compiler ABI info
-- Check for working CXX compiler: /usr/bin/g++
-- Build files have been written to: .../hello_cmake/build
[ 50%] Building CXX object CMakeFiles/hello.dir/hello.cpp.o
[100%] Linking CXX executable hello
[100%] Built target hello
```

빌드 완료 후:

```bash
./hello
```

```
hello, GPT!
```

**CMake 빌드 흐름 정리:**

```
CMakeLists.txt          ← 빌드 설명서
       ↓ cmake ..
build/Makefile          ← 실제 빌드 규칙 생성
       ↓ cmake --build .
build/hello             ← 실행 파일 완성
```

`build/` 디렉토리를 따로 만드는 이유: 소스 파일과 빌드 결과물을 섞지 않기 위해서입니다. `build/`만 지우면 깨끗하게 초기화됩니다.

---

## 실습: CMake로 microgpt 빌드

프로젝트 루트의 `CMakeLists.txt`를 사용해 실제 microgpt를 빌드해 봅시다.

```bash
# 프로젝트 루트에서
mkdir build
cd build
cmake ..
cmake --build .
```

빌드가 완료되면 `build/microgpt` 실행 파일이 생깁니다.

```bash
cd ..
cp input.txt build/   # 데이터 파일 복사 (없으면 자동 다운로드)
cd build
./microgpt
```

**실행 결과 (처음 몇 줄):**
```
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 3.2917
step    2 / 1000 | loss 3.1843
...
```

학습이 진행되면서 loss가 점점 줄어드는 것이 보이면 성공입니다.

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `g++ --version` | 컴파일러 설치 및 버전 확인 |
| `g++ -std=c++17 -O2 -o 출력 소스.cpp` | C++ 파일 컴파일 |
| `./실행파일` | 컴파일된 프로그램 실행 |
| `#include <헤더>` | 표준 라이브러리 포함 (Python의 `import`) |
| `std::cout << "..." << std::endl;` | 화면 출력 (Python의 `print`) |
| `int main() { ... }` | 프로그램 시작점 |
| `CMakeLists.txt` | CMake 빌드 설정 파일 |
| `cmake .. && cmake --build .` | CMake 빌드 실행 |

다음 챕터에서는 Python과 C++의 핵심 차이점을 비교하며, 왜 C++에서 포인터와 자유 함수를 써야 하는지 이해합니다.


---
[< 이전: Ch05: Transformer와 학습](../part1-python/ch05-transformer-and-training.md) | [목차](../README.md) | [다음: Ch07: Python→C++ 브릿지 >](ch07-python-to-cpp-bridge.md)
