# MicroGPT — Zero에서 GPT까지

> 순수 Python과 C++ 표준 라이브러리만으로 GPT를 처음부터 구현하고, 브라우저에서 실시간으로 시각화하는 교육 프로젝트

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://python.org)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-GitHub%20Pages-blue.svg)](https://sageraii.github.io/microgpt-cpp/)

<p align="center">
  <strong>4,192개의 파라미터로 이해하는 GPT의 모든 것</strong><br>
  <a href="https://sageraii.github.io/microgpt-cpp/">라이브 데모 바로가기</a>
</p>

```
microgpt.py (200줄)     microgpt.cpp (515줄)     viz/index.html (1,984줄)
Python 원본              C++17 변환                브라우저 시각화
───────────────────     ───────────────────      ───────────────────
외부 라이브러리: 0개     외부 라이브러리: 0개      외부 라이브러리: 0개
import os, math, random  #include <vector> 등      순수 HTML/CSS/JS
```

## 무엇을 하는 프로젝트인가?

이 프로젝트는 [@karpathy](https://github.com/karpathy)의 [microgpt.py](https://github.com/karpathy/microgpt)를 기반으로:

1. **Python 200줄**로 GPT를 구현합니다 (외부 라이브러리 없이 `os`, `math`, `random`만 사용)
2. **C++ 515줄**로 동일한 GPT를 재구현합니다 (C++ 표준 라이브러리만 사용)
3. **19개 챕터의 한국어 튜토리얼**로 Hello World부터 완성된 GPT까지 설명합니다
4. **브라우저 시각화 도구**로 학습/추론/역전파 과정을 실시간으로 보여줍니다

## 빠른 시작

### 1. C++ 버전 빌드 & 실행

```bash
# 방법 1: g++ 직접 컴파일
g++ -std=c++17 -O2 -o microgpt microgpt.cpp
./microgpt

# 방법 2: CMake
mkdir build && cd build
cmake .. && cmake --build .
./microgpt
```

### 2. Python 버전 실행

```bash
python3 microgpt.py
```

### 3. 시각화 도구

**온라인:** https://sageraii.github.io/microgpt-cpp/ (바로 사용 가능)

**로컬 실행:**
```bash
cd viz
python3 -m http.server 8080
# 브라우저에서 http://localhost:8080 접속
```

> `data.js`가 없는 경우 먼저 가중치를 내보냅니다:
> ```bash
> cd viz
> g++ -std=c++17 -O2 -o export_weights export_weights.cpp
> ln -sf ../input.txt input.txt
> ./export_weights    # data.js 생성 (~66KB)
> ```

### 실행 결과

```
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 3.2917
step    2 / 1000 | loss 3.1843
...
step 1000 / 1000 | loss 2.0080
--- inference (new, hallucinated names) ---
sample  1: kaspen
sample  2: maede
sample  3: shan
...
```

## 프로젝트 구조

```
microgpt-cpp/
├── README.md                 # 이 파일
├── microgpt.py               # Python 원본 (200줄, @karpathy)
├── microgpt.cpp              # C++17 변환 (515줄)
├── CMakeLists.txt            # CMake 빌드 설정
├── input.txt                 # 학습 데이터 (32,033개 영어 이름)
│
├── docs/                     # 한국어 튜토리얼 (19 챕터)
│   ├── README.md             # 튜토리얼 목차
│   ├── part1-python/         # Part 1: Python으로 GPT 이해하기
│   │   ├── ch00-setup.md          # 환경 설정
│   │   ├── ch01-data-tokenizer.md # 데이터와 토크나이저
│   │   ├── ch02-first-neural-net.md # 신경망 첫걸음
│   │   ├── ch03-why-learning-works.md # 왜 학습이 되는가
│   │   ├── ch04-value-and-backprop.md # Value와 역전파
│   │   └── ch05-transformer-and-training.md # Transformer와 학습
│   └── part2-cpp/            # Part 2: C++로 GPT 구현하기
│       ├── ch06-cpp-setup.md      # C++ 환경 설정
│       ├── ch07-python-to-cpp-bridge.md # Python→C++ 브릿지
│       ├── ch08-value-struct.md   # Value 구조체
│       ├── ch09-graph-arena.md    # Graph 아레나
│       ├── ch10-differentiable-ops.md # 미분 가능한 연산
│       ├── ch11-backward.md       # 역전파 구현
│       ├── ch12-nn-building-blocks.md # 신경망 빌딩 블록
│       ├── ch13-data-loading.md   # 데이터 로딩
│       ├── ch14-parameters.md     # 파라미터 초기화
│       ├── ch15-gpt-forward.md    # GPT 순전파
│       ├── ch16-training-loop.md  # 학습 루프
│       ├── ch17-inference.md      # 추론
│       └── ch18-build-and-wrap-up.md # 빌드와 마무리
│
└── viz/                      # 브라우저 시각화 도구
    ├── index.html            # 4탭 대시보드 (1,984줄)
    ├── data.js               # 학습된 모델 데이터 (~66KB)
    └── export_weights.cpp    # 가중치 내보내기 도구
```

## 튜토리얼 가이드

19개 챕터가 Hello World부터 시작하여 완성된 GPT까지 한 줄씩 설명합니다.

### Part 1: Python으로 GPT 이해하기

| 챕터 | 제목 | 핵심 내용 |
|------|------|----------|
| [Ch00](docs/part1-python/ch00-setup.md) | 환경 설정 | Python 설치, Hello World |
| [Ch01](docs/part1-python/ch01-data-tokenizer.md) | 데이터와 토크나이저 | 문자를 숫자로 바꾸기 |
| [Ch02](docs/part1-python/ch02-first-neural-net.md) | 신경망 첫걸음 | linear, softmax 맛보기 |
| [Ch03](docs/part1-python/ch03-why-learning-works.md) | 왜 학습이 되는가 | 손실함수, 경사하강법 |
| [Ch04](docs/part1-python/ch04-value-and-backprop.md) | Value와 역전파 | 자동 미분 엔진 |
| [Ch05](docs/part1-python/ch05-transformer-and-training.md) | Transformer와 학습 | 어텐션, MLP, Adam |

### Part 2: C++로 GPT 구현하기

| 챕터 | 제목 | 핵심 내용 |
|------|------|----------|
| [Ch06](docs/part2-cpp/ch06-cpp-setup.md) | C++ 환경 설정 | g++, CMake, Hello World |
| [Ch07](docs/part2-cpp/ch07-python-to-cpp-bridge.md) | Python→C++ 브릿지 | 언어 차이, 메모리 모델 |
| [Ch08](docs/part2-cpp/ch08-value-struct.md) | Value 구조체 | struct, 포인터, 필드 |
| [Ch09](docs/part2-cpp/ch09-graph-arena.md) | Graph 아레나 | new/delete, RAII |
| [Ch10](docs/part2-cpp/ch10-differentiable-ops.md) | 미분 가능한 연산 | add, mul, log, exp, relu |
| [Ch11](docs/part2-cpp/ch11-backward.md) | 역전파 구현 | 위상정렬, 체인룰 |
| [Ch12](docs/part2-cpp/ch12-nn-building-blocks.md) | 신경망 빌딩 블록 | linear, softmax, rmsnorm |
| [Ch13](docs/part2-cpp/ch13-data-loading.md) | 데이터 로딩 | fstream, set, unordered_map |
| [Ch14](docs/part2-cpp/ch14-parameters.md) | 파라미터 초기화 | random, state_dict |
| [Ch15](docs/part2-cpp/ch15-gpt-forward.md) | GPT 순전파 | KV 캐시, 멀티헤드 어텐션 |
| [Ch16](docs/part2-cpp/ch16-training-loop.md) | 학습 루프 | Adam, 크로스엔트로피 |
| [Ch17](docs/part2-cpp/ch17-inference.md) | 추론 | Temperature sampling |
| [Ch18](docs/part2-cpp/ch18-build-and-wrap-up.md) | 빌드와 마무리 | CMake, 성능 비교 |

## 시각화 도구

**MicroGPT Visual Explorer** — 4개 탭으로 구성된 다크 테마 대시보드

| 탭 | 기능 |
|----|------|
| **데이터 & 토크나이저** | 인터랙티브 토큰화, 어휘 테이블 |
| **학습 Training** | Loss 곡선, 학습률, 어텐션 히트맵 스냅샷 |
| **추론 Inference** | JS 실시간 추론, 확률분포, 어텐션 시각화, 온도 조절 |
| **역전파 Backprop** | 계산 그래프 애니메이션, 그래디언트 플로우, 체인룰 |

추론 탭은 학습된 가중치를 JavaScript로 직접 로딩하여 **브라우저에서 실시간 GPT 순전파**를 실행합니다.

## 모델 사양

```
아키텍처: GPT-2 (간소화)
─────────────────────────────
파라미터 수:      4,192
레이어 수:        1
임베딩 차원:      16
어텐션 헤드 수:   4
헤드 차원:        4
컨텍스트 길이:    16
어휘 크기:        27 (a-z + BOS)
─────────────────────────────
GPT-2와 차이점:
  LayerNorm → RMSNorm
  GeLU → ReLU
  바이어스 없음
```

## 요구사항

- **C++**: g++ 7+ (C++17), 권장 g++ 10+
- **Python**: 3.10+
- **시각화**: 모던 브라우저 (Chrome, Firefox, Safari, Edge)
- **외부 라이브러리**: 없음 (전부 표준 라이브러리만 사용)

## 감사의 말

- [@karpathy](https://github.com/karpathy) — [microgpt.py](https://github.com/karpathy/microgpt) 원본 코드
- [makemore](https://github.com/karpathy/makemore) — 이름 데이터셋 (names.txt)

## 라이선스

MIT License
