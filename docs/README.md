# Zero에서 GPT까지: Python과 C++로 만드는 나만의 GPT

> @karpathy의 microgpt.py를 기반으로, GPT의 모든 것을 처음부터 직접 구현하는 튜토리얼

## 이 튜토리얼은 누구를 위한 것인가?

- C++/Python 기초 문법을 아는 프로그래밍 입문자
- 딥러닝, GPT, 신경망이 뭔지 전혀 모르는 사람
- "ChatGPT가 어떻게 동작하는지" 코드로 이해하고 싶은 사람

## 사전 요구사항

- Python 3.10+ 설치
- g++ (C++17 지원) 또는 CMake 3.16+
- 수학: 사칙연산, 함수 개념 (미적분은 본문에서 설명)

## 튜토리얼 구조

### Part 1: Python으로 GPT 개념 이해하기

GPT가 무엇인지, 어떻게 학습하는지를 Python으로 빠르게 체험합니다.

| 챕터 | 제목 | 배우는 것 |
|------|------|----------|
| [Ch00](part1-python/ch00-setup.md) | 환경 설정 | Python 설치, Hello World |
| [Ch01](part1-python/ch01-data-tokenizer.md) | 데이터와 토크나이저 | 문자를 숫자로 바꾸기 |
| [Ch02](part1-python/ch02-first-neural-net.md) | 신경망 첫걸음 | linear, softmax 맛보기 |
| [Ch03](part1-python/ch03-why-learning-works.md) | 왜 학습이 되는가 | 손실함수, 경사하강법 |
| [Ch04](part1-python/ch04-value-and-backprop.md) | Value와 역전파 | 자동 미분 엔진 |
| [Ch05](part1-python/ch05-transformer-and-training.md) | Transformer와 학습 | 어텐션, MLP, Adam |

### Part 2: C++로 GPT 구현하기 (메인)

Part 1에서 이해한 개념을 C++ 표준 라이브러리만으로 직접 구현합니다.

| 챕터 | 제목 | 배우는 것 |
|------|------|----------|
| [Ch06](part2-cpp/ch06-cpp-setup.md) | C++ 환경 설정 | g++, CMake, Hello World |
| [Ch07](part2-cpp/ch07-python-to-cpp-bridge.md) | Python→C++ 브릿지 | 언어 차이, 메모리 모델 |
| [Ch08](part2-cpp/ch08-value-struct.md) | Value 구조체 | struct, 포인터, 필드 설계 |
| [Ch09](part2-cpp/ch09-graph-arena.md) | Graph 아레나 | new/delete, RAII, 아레나 패턴 |
| [Ch10](part2-cpp/ch10-differentiable-ops.md) | 미분 가능한 연산 | add, mul, log, exp, relu |
| [Ch11](part2-cpp/ch11-backward.md) | 역전파 구현 | 위상정렬, 체인룰, 검증 |
| [Ch12](part2-cpp/ch12-nn-building-blocks.md) | 신경망 빌딩 블록 | linear, softmax, rmsnorm |
| [Ch13](part2-cpp/ch13-data-loading.md) | 데이터 로딩 | fstream, set, unordered_map |
| [Ch14](part2-cpp/ch14-parameters.md) | 파라미터 초기화 | random, state_dict |
| [Ch15](part2-cpp/ch15-gpt-forward.md) | GPT 순전파 | KV 캐시, 멀티헤드 어텐션 |
| [Ch16](part2-cpp/ch16-training-loop.md) | 학습 루프 | Adam, 크로스엔트로피 |
| [Ch17](part2-cpp/ch17-inference.md) | 추론 | Temperature sampling |
| [Ch18](part2-cpp/ch18-build-and-wrap-up.md) | 빌드와 마무리 | CMake, 성능 비교, 다음 단계 |

## 빠른 시작

```bash
# 완성된 코드 컴파일 및 실행
g++ -std=c++17 -O2 -o microgpt microgpt.cpp
./microgpt
```

## 파일 구조

```
microgpt-cpp/
├── microgpt.py          # 원본 Python (@karpathy)
├── microgpt.cpp         # C++17 완성본 (한국어 주석)
├── CMakeLists.txt       # CMake 빌드 설정
└── docs/
    ├── README.md        # 이 파일
    ├── part1-python/    # Part 1: Python 개념 설명
    └── part2-cpp/       # Part 2: C++ 구현 (메인)
```

## 크레딧

- 원본: [@karpathy](https://github.com/karpathy)의 microgpt.py
- C++ 변환 및 튜토리얼: Claude Code
