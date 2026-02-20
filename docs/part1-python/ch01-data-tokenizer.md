# Ch01: 데이터와 토크나이저

## 학습 목표

- GPT에게 어떤 데이터를 "먹이"로 주는지 이해한다
- 파일을 읽고 섞는 코드를 직접 작성한다
- 토크나이저가 무엇인지, 왜 필요한지 파악한다
- "emma"를 숫자 시퀀스로 변환하고, 다시 되돌리는 실습을 완성한다

---

## 개념 설명

### GPT에게 먹이를 주자 — 데이터셋

GPT는 태어날 때 아무것도 모릅니다. 사람이 언어를 배우듯, 수많은 예시를 보면서 패턴을 익힙니다. 우리가 줄 "먹이"는 영어 이름 32,000개 목록입니다.

```
데이터셋 (input.txt)
┌───────────────────┐
│ emma              │
│ olivia            │
│ ava               │
│ isabella          │
│ sophia            │
│ ... (32,000줄)    │
└───────────────────┘
```

GPT는 이 이름들을 보면서 스스로 질문합니다.

```
"e 다음에는 뭐가 올까?"  →  주로 m, l, v 등이 온다
"em 다음에는 뭐가 올까?" →  주로 m, i, a 등이 온다
"emm 다음에는 뭐가 올까?" → 주로 a, e, y 등이 온다
```

이 패턴을 모두 학습하면, 나중에 혼자서 새 이름을 만들어 낼 수 있게 됩니다.

### 토크나이저란?

컴퓨터는 글자를 직접 이해하지 못합니다. 컴퓨터가 이해하는 것은 오직 숫자입니다. 그래서 **글자를 숫자로 바꿔 주는 번역기**가 필요합니다. 이것이 토크나이저(Tokenizer)입니다.

```
[토크나이저 — 글자 ↔ 숫자 사전]

글자 → 숫자 (인코딩)          숫자 → 글자 (디코딩)
  a  →   0                      0  →   a
  b  →   1                      1  →   b
  c  →   2                      2  →   c
  ...                           ...
  z  →  25                     25  →   z
 BOS →  26                     26  →  BOS
```

`BOS`(Beginning Of Sequence)는 특수 토큰입니다. "이름의 시작"과 "이름의 끝"을 알려 주는 신호입니다. 마치 문장 부호처럼요.

```
"emma" 의 토큰 시퀀스:

[BOS] e  m  m  a  [BOS]
 26   4  12  12  0  26

의미: "지금 이름이 시작됩니다 → e → m → m → a → 이름이 끝났습니다"
```

---

## 코드 작성

새 파일 `ch01_tokenizer.py`를 만들고 한 줄씩 따라 써 보세요.

### 1단계: 데이터 다운로드와 읽기

```python
# ch01_tokenizer.py

import os       # 파일 존재 확인
import random   # 리스트 섞기

# ① 파일이 없으면 인터넷에서 자동으로 다운로드합니다
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
    print("input.txt 다운로드 완료!")

# ② 파일을 읽어서 docs 리스트에 저장합니다
docs = [line.strip() for line in open('input.txt') if line.strip()]
print(f"이름 개수: {len(docs)}")   # num docs: 32033
```

각 줄 설명:

- `os.path.exists('input.txt')` — `input.txt` 파일이 현재 폴더에 있는지 확인합니다. 있으면 `True`, 없으면 `False`입니다.
- `urllib.request.urlretrieve(url, 파일이름)` — URL에서 파일을 다운로드합니다.
- `[line.strip() for line in open('input.txt') if line.strip()]` — 파일의 모든 줄을 읽어서 리스트로 만듭니다. `strip()`은 줄 끝의 줄바꿈 문자(`\n`)를 제거하고, `if line.strip()`은 빈 줄을 건너뜁니다.

### 2단계: 데이터 섞기

```python
# ③ 데이터를 무작위로 섞습니다
random.seed(42)          # 재현 가능한 결과를 위해 시드를 고정합니다
random.shuffle(docs)     # docs 리스트의 순서를 뒤섞습니다

# 섞인 후 처음 5개 확인
print("섞인 첫 5개:", docs[:5])
```

- `random.seed(42)` — 무작위의 시작 값을 고정합니다. 이렇게 하면 여러분이 실행해도 저와 같은 결과가 나옵니다. 숫자 42는 특별한 의미 없이 관례적으로 자주 쓰입니다.
- `random.shuffle(docs)` — 리스트 안의 원소 순서를 뒤섞습니다. 이름의 알파벳 순서 등 편향을 없애 학습 품질을 높입니다.

### 3단계: 토크나이저 만들기

```python
# ④ 데이터셋에 등장하는 모든 고유 문자를 뽑아 정렬합니다
uchars = sorted(set(''.join(docs)))
print(f"고유 문자: {uchars}")     # ['a', 'b', 'c', ..., 'z']
print(f"고유 문자 수: {len(uchars)}")  # 26

# ⑤ BOS 토큰 ID는 고유 문자 개수와 같습니다 (마지막 번호)
BOS = len(uchars)               # 26
print(f"BOS 토큰 ID: {BOS}")

# ⑥ 전체 어휘 크기: 26개 문자 + 1개 BOS 토큰
vocab_size = len(uchars) + 1    # 27
print(f"vocab_size: {vocab_size}")
```

각 줄 설명:

- `''.join(docs)` — `docs` 리스트의 모든 이름을 하나의 긴 문자열로 합칩니다. `"emma" + "olivia" + ...` 형태.
- `set(...)` — 중복을 제거합니다. `"emma"` 안의 `m`이 두 개여도 `set`에는 한 번만 들어갑니다.
- `sorted(...)` — 알파벳 순으로 정렬합니다. `['a', 'b', 'c', ..., 'z']`
- `BOS = len(uchars)` — 0부터 25까지가 `a`~`z`에 할당되었으므로, 그 다음 번호인 26이 BOS의 번호가 됩니다.

### 4단계: 인코딩과 디코딩

```python
# ⑦ 인코딩: 글자 → 숫자
def encode(name):
    """이름 문자열을 토큰 ID 리스트로 변환합니다."""
    return [BOS] + [uchars.index(ch) for ch in name] + [BOS]

# ⑧ 디코딩: 숫자 → 글자
def decode(tokens):
    """토큰 ID 리스트를 문자열로 복원합니다. BOS 토큰은 건너뜁니다."""
    return ''.join(uchars[t] for t in tokens if t != BOS)


if __name__ == '__main__':
    # 실습 ①: "emma"를 토큰 시퀀스로 변환하기
    name = "emma"
    tokens = encode(name)
    print(f"encode('{name}') = {tokens}")
    # 출력: encode('emma') = [26, 4, 12, 12, 0, 26]

    # 실습 ②: 토큰 시퀀스를 다시 "emma"로 되돌리기
    restored = decode(tokens)
    print(f"decode({tokens}) = '{restored}'")
    # 출력: decode([26, 4, 12, 12, 0, 26]) = 'emma'

    # 검증: 원본과 동일한지 확인
    assert name == restored, "인코딩/디코딩 오류!"
    print("인코딩 → 디코딩 완벽히 일치!")
```

`encode()` 함수 상세 설명:

```
encode("emma")
  단계 1: [BOS]              →  [26]
  단계 2: [uchars.index('e')]  →  [4]   ('e'는 uchars의 4번 위치)
  단계 3: [uchars.index('m')]  →  [12]  ('m'는 uchars의 12번 위치)
  단계 4: [uchars.index('m')]  →  [12]
  단계 5: [uchars.index('a')]  →  [0]   ('a'는 uchars의 0번 위치)
  단계 6: [BOS]              →  [26]
  결과: [26, 4, 12, 12, 0, 26]
```

`decode()` 함수 상세 설명:

```
decode([26, 4, 12, 12, 0, 26])
  26 → BOS  → 건너뜀
   4 → uchars[4]  = 'e'
  12 → uchars[12] = 'm'
  12 → uchars[12] = 'm'
   0 → uchars[0]  = 'a'
  26 → BOS  → 건너뜀
  결과: "emma"
```

---

## 실행 결과 확인

```bash
python ch01_tokenizer.py
```

예상 출력:

```
이름 개수: 32033
섞인 첫 5개: ['aditya', 'mykola', 'taish', 'raelynn', 'neeraj']
고유 문자: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
고유 문자 수: 26
BOS 토큰 ID: 26
vocab_size: 27
encode('emma') = [26, 4, 12, 12, 0, 26]
decode([26, 4, 12, 12, 0, 26]) = 'emma'
인코딩 → 디코딩 완벽히 일치!
```

---

## 핵심 정리

| 개념 | 설명 | 코드 |
|------|------|------|
| 데이터셋 | GPT의 학습 재료 | `docs = [line.strip() for line in open('input.txt')]` |
| 셔플 | 순서 편향 제거 | `random.shuffle(docs)` |
| `uchars` | 고유 문자 목록 (a~z, 26개) | `sorted(set(''.join(docs)))` |
| `BOS` | 시작/끝 특수 토큰 (ID = 26) | `BOS = len(uchars)` |
| `vocab_size` | 전체 토큰 종류 수 (27개) | `len(uchars) + 1` |
| 인코딩 | 글자 → 숫자 | `uchars.index(ch)` |
| 디코딩 | 숫자 → 글자 | `uchars[token_id]` |

다음 챕터에서는 이 숫자 시퀀스를 받아서 처리하는 **신경망의 기본 연산**을 배웁니다.


> **직접 체험하기** — 시각화 도구에서 토큰화 과정을 인터랙티브하게 확인할 수 있습니다: [라이브 데모에서 직접 체험](https://sageraii.github.io/microgpt-cpp/#tokenizer)

---
[< 이전: Ch00: 환경 설정](ch00-setup.md) | [목차](../README.md) | [다음: Ch02: 신경망 첫걸음 >](ch02-first-neural-net.md)
