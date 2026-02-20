/*
 * microgpt.cpp — 순수 C++17로 구현한 최소 GPT
 *
 * GPT를 학습하고 추론하는 가장 원자적인(atomic) 방법.
 * C++ 표준 라이브러리만 사용하며, 외부 의존성이 전혀 없습니다.
 * 이 파일이 알고리즘의 전부입니다. 나머지는 효율성의 문제일 뿐.
 *
 * @karpathy의 microgpt.py를 C++로 변환
 *
 * 컴파일: g++ -std=c++17 -O2 -o microgpt microgpt.cpp
 * 실행:   ./microgpt
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// 자동 미분 엔진 (Autograd): 스칼라 단위 자동 미분
// ============================================================================
//
// 핵심 아이디어: 모든 연산을 Value 노드로 기록하여 계산 그래프(DAG)를 구축한다.
// 순전파(forward)에서 값을 계산하고, 역전파(backward)에서 체인룰로 기울기를 구한다.
//
// 예시: z = x * y 일 때
//   - 순전파: z.data = x.data * y.data
//   - 역전파: x.grad += y.data * z.grad  (dz/dx = y)
//             y.grad += x.data * z.grad  (dz/dy = x)

struct Value {
    double data;                       // 스칼라 값 (순전파에서 계산)
    double grad = 0.0;                 // dLoss/d(this) (역전파에서 계산)
    std::vector<Value*> children;      // 계산 그래프에서의 자식 노드들
    std::vector<double> local_grads;   // 각 자식에 대한 국소 미분값 (d(this)/d(child))
};

// 그래프 아레나: 순전파 중 생성되는 모든 임시 Value 노드를 소유한다.
// 학습 스텝 사이에 clear()로 계산 그래프를 해제한다.
// 모델 파라미터는 별도로 할당되어 스텝 간 유지된다.
struct Graph {
    std::vector<Value*> nodes;

    // 새 Value 노드를 생성하고 아레나에 등록
    Value* make(double data, std::vector<Value*> children = {},
                std::vector<double> local_grads = {}) {
        auto* v = new Value{data, 0.0, std::move(children), std::move(local_grads)};
        nodes.push_back(v);
        return v;
    }

    // 아레나의 모든 임시 노드를 해제 (파라미터는 영향 없음)
    void clear() {
        for (auto* v : nodes) delete v;
        nodes.clear();
    }

    ~Graph() { clear(); }
};

static Graph graph; // 현재 스텝의 계산 그래프를 위한 전역 아레나

// --- 미분 가능한 연산들 ---
// 각 연산은 결과 Value를 생성하면서 자식 노드와 국소 미분을 기록한다.
// 이 정보가 역전파 시 체인룰 적용에 사용된다.

// 덧셈: d(a+b)/da = 1, d(a+b)/db = 1
inline Value* add(Value* a, Value* b) {
    return graph.make(a->data + b->data, {a, b}, {1.0, 1.0});
}

// 곱셈: d(a*b)/da = b, d(a*b)/db = a
inline Value* mul(Value* a, Value* b) {
    return graph.make(a->data * b->data, {a, b}, {b->data, a->data});
}

// 스칼라 곱: d(a*s)/da = s (상수 s에는 기울기가 흐르지 않음)
inline Value* scale(Value* a, double s) {
    return graph.make(a->data * s, {a}, {s});
}

// 거듭제곱: d(a^n)/da = n * a^(n-1)
inline Value* power(Value* a, double n) {
    return graph.make(std::pow(a->data, n), {a}, {n * std::pow(a->data, n - 1)});
}

// 자연로그: d(ln a)/da = 1/a
inline Value* log(Value* a) {
    return graph.make(std::log(a->data), {a}, {1.0 / a->data});
}

// 지수함수: d(e^a)/da = e^a
inline Value* exp(Value* a) {
    double e = std::exp(a->data);
    return graph.make(e, {a}, {e});
}

// ReLU 활성화 함수: d(relu(a))/da = 1 if a > 0, else 0
inline Value* relu(Value* a) {
    return graph.make(std::max(0.0, a->data), {a}, {a->data > 0 ? 1.0 : 0.0});
}

// 복합 연산 (위 기본 연산들의 조합)
inline Value* neg(Value* a)              { return scale(a, -1.0); }
inline Value* sub(Value* a, Value* b)    { return add(a, neg(b)); }
inline Value* div(Value* a, Value* b)    { return mul(a, power(b, -1.0)); }

// 상수 빼기: 상수를 위한 노드를 만들지 않아 더 효율적
inline Value* sub_const(Value* a, double c) {
    return graph.make(a->data - c, {a}, {1.0});
}

// --- 역전파: 위상 정렬 + 역방향 체인룰 ---
//
// 1. 계산 그래프를 위상 정렬하여 노드 순서를 결정한다.
// 2. 역순으로 순회하며 각 노드에서 자식으로 기울기를 전파한다.
//    child.grad += local_grad * parent.grad  (체인룰)
void backward(Value* root) {
    // 위상 정렬 (DFS 기반)
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

    // 역방향 누적: loss의 자기 자신에 대한 미분은 1.0
    root->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Value* v = *it;
        for (size_t i = 0; i < v->children.size(); i++)
            v->children[i]->grad += v->local_grads[i] * v->grad;
    }
}

// ============================================================================
// 신경망 텐서를 위한 타입 별칭
// ============================================================================

using Vec = std::vector<Value*>;  // 1차원 벡터 (예: 임베딩 하나)
using Mat = std::vector<Vec>;     // 2차원 행렬 (예: 가중치 행렬)

// ============================================================================
// 신경망 구성 요소 (빌딩 블록)
// ============================================================================

// 선형 변환: y = W @ x (행렬-벡터 곱, 바이어스 없음)
// W의 각 행과 x의 내적을 구해 출력 벡터를 만든다.
Vec linear(const Vec& x, const Mat& w) {
    Vec out;
    out.reserve(w.size());
    for (const auto& row : w) {
        Value* s = graph.make(0.0);
        for (size_t i = 0; i < x.size(); i++)
            s = add(s, mul(row[i], x[i]));
        out.push_back(s);
    }
    return out;
}

// 소프트맥스: 로짓을 확률 분포로 변환
// 수치 안정성을 위해 최댓값을 빼고 exp를 취한다.
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
Vec softmax(const Vec& logits) {
    // 수치 안정성을 위한 최댓값 계산 (오버플로 방지)
    double max_val = -1e30;
    for (auto* v : logits)
        max_val = std::max(max_val, v->data);

    // exp(logit - max) 계산 및 합산
    Vec exps;
    exps.reserve(logits.size());
    Value* total = graph.make(0.0);
    for (auto* v : logits) {
        Value* e = exp(sub_const(v, max_val));
        exps.push_back(e);
        total = add(total, e);
    }

    // 각 exp를 합계로 나누어 확률 산출
    Vec probs;
    probs.reserve(logits.size());
    for (auto* e : exps)
        probs.push_back(div(e, total));
    return probs;
}

// RMSNorm: 제곱평균제곱근(RMS)으로 정규화
// LayerNorm과 달리 평균을 빼지 않고, 학습 가능한 파라미터도 없다.
// rmsnorm(x) = x / sqrt(mean(x^2) + eps)
Vec rmsnorm(const Vec& x) {
    // 제곱의 평균 계산
    Value* ms = graph.make(0.0);
    for (auto* xi : x)
        ms = add(ms, mul(xi, xi));
    ms = scale(ms, 1.0 / static_cast<double>(x.size()));

    // 스케일 팩터: 1 / sqrt(ms + eps)
    Value* s = power(add(ms, graph.make(1e-5)), -0.5);

    // 각 원소에 스케일 적용
    Vec out;
    out.reserve(x.size());
    for (auto* xi : x)
        out.push_back(mul(xi, s));
    return out;
}

// ============================================================================
// 하이퍼파라미터
// ============================================================================

constexpr int N_LAYER    = 1;   // 트랜스포머 깊이 (레이어 수)
constexpr int N_EMBD     = 16;  // 네트워크 너비 (임베딩 차원)
constexpr int BLOCK_SIZE = 16;  // 최대 컨텍스트 길이 (가장 긴 이름: 15자)
constexpr int N_HEAD     = 4;   // 어텐션 헤드 수
constexpr int HEAD_DIM   = N_EMBD / N_HEAD; // 헤드당 차원 (= 4)

// ============================================================================
// GPT 순전파 (한 번에 토큰 하나씩, KV 캐시를 사용한 자기회귀 방식)
// GPT-2 구조를 따르되: LayerNorm→RMSNorm, 바이어스 없음, GeLU→ReLU
// ============================================================================

Vec gpt(int token_id, int pos_id,
        std::vector<std::vector<Vec>>& keys,
        std::vector<std::vector<Vec>>& vals,
        std::unordered_map<std::string, Mat>& sd) {

    // 토큰 임베딩 + 위치 임베딩을 합산하여 초기 표현 생성
    Vec x(N_EMBD);
    for (int i = 0; i < N_EMBD; i++)
        x[i] = add(sd["wte"][token_id][i], sd["wpe"][pos_id][i]);
    x = rmsnorm(x); // 잔차 연결을 통한 역전파 때문에 중복이 아님

    for (int li = 0; li < N_LAYER; li++) {
        std::string pfx = "layer" + std::to_string(li);

        // ---- 1) 멀티헤드 어텐션 블록 ----
        Vec x_residual = x; // 잔차 연결을 위해 저장
        x = rmsnorm(x);

        // Q, K, V 프로젝션: 입력을 쿼리, 키, 밸류로 변환
        Vec q = linear(x, sd[pfx + ".attn_wq"]); // 쿼리 (Query): "내가 찾는 것"
        Vec k = linear(x, sd[pfx + ".attn_wk"]); // 키 (Key): "내가 가진 것의 라벨"
        Vec v = linear(x, sd[pfx + ".attn_wv"]); // 밸류 (Value): "내가 가진 것의 내용"

        // KV 캐시에 추가 (이전 위치의 K, V를 재사용하여 효율적 생성)
        keys[li].push_back(k);
        vals[li].push_back(v);

        Vec x_attn;
        x_attn.reserve(N_EMBD);
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM; // 현재 헤드의 시작 인덱스

            // 현재 헤드에 해당하는 쿼리 슬라이스 추출
            Vec q_h(q.begin() + hs, q.begin() + hs + HEAD_DIM);
            int T = static_cast<int>(keys[li].size()); // 캐시된 위치 수

            // 스케일드 닷-프로덕트 어텐션
            // attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
            double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
            Vec attn_logits;
            attn_logits.reserve(T);
            for (int t = 0; t < T; t++) {
                // Q와 K의 내적 → 유사도 점수
                Value* dot = graph.make(0.0);
                for (int j = 0; j < HEAD_DIM; j++)
                    dot = add(dot, mul(q_h[j], keys[li][t][hs + j]));
                // sqrt(d_k)로 나누어 스케일링 (기울기 안정화)
                attn_logits.push_back(scale(dot, inv_sqrt));
            }

            // 어텐션 가중치 계산 (softmax로 확률 분포 변환)
            Vec attn_weights = softmax(attn_logits);

            // 가중 합산: 어텐션 가중치로 Value를 결합
            for (int j = 0; j < HEAD_DIM; j++) {
                Value* s = graph.make(0.0);
                for (int t = 0; t < T; t++)
                    s = add(s, mul(attn_weights[t], vals[li][t][hs + j]));
                x_attn.push_back(s);
            }
        }

        // 출력 프로젝션 + 잔차 연결 (skip connection)
        x = linear(x_attn, sd[pfx + ".attn_wo"]);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = add(x[i], x_residual[i]);

        // ---- 2) MLP 블록 (피드포워드 네트워크) ----
        x_residual = x;
        x = rmsnorm(x);
        x = linear(x, sd[pfx + ".mlp_fc1"]); // 확장: n_embd → 4*n_embd
        for (auto& xi : x) xi = relu(xi);    // 비선형 활성화
        x = linear(x, sd[pfx + ".mlp_fc2"]); // 축소: 4*n_embd → n_embd
        for (int i = 0; i < N_EMBD; i++)
            x[i] = add(x[i], x_residual[i]); // 잔차 연결
    }

    // 최종 선형 프로젝션: 임베딩 → 어휘 크기 로짓
    return linear(x, sd["lm_head"]);
}

// ============================================================================
// 메인: 데이터 로딩, 학습, 추론
// ============================================================================

int main() {
    // --- 데이터셋 준비 ---
    // input.txt가 없으면 Karpathy의 makemore 이름 데이터셋을 다운로드
    {
        std::ifstream test("input.txt");
        if (!test) {
            std::cout << "input.txt 다운로드 중...\n";
            int ret = std::system(
                "curl -sL -o input.txt "
                "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt");
            if (ret != 0) {
                std::cerr << "다운로드 실패. 수동으로 다운로드하세요:\n"
                          << "  curl -o input.txt https://raw.githubusercontent.com/"
                          << "karpathy/makemore/988aa59/names.txt\n";
                return 1;
            }
        }
    }

    // 파일에서 문서(이름) 목록 읽기 — 한 줄이 하나의 "문서"
    std::vector<std::string> docs;
    {
        std::ifstream file("input.txt");
        std::string line;
        while (std::getline(file, line)) {
            auto start = line.find_first_not_of(" \t\r\n");
            auto end = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos)
                docs.push_back(line.substr(start, end - start + 1));
        }
    }

    std::mt19937 rng(42); // 혼돈 속에 질서가 있기를
    std::shuffle(docs.begin(), docs.end(), rng);
    std::cout << "num docs: " << docs.size() << "\n";

    // --- 토크나이저 ---
    // 문자 단위(char-level) 토크나이징: 각 고유 문자 → 하나의 토큰 ID
    std::set<char> char_set;
    for (const auto& doc : docs)
        for (char c : doc)
            char_set.insert(c);
    std::vector<char> uchars(char_set.begin(), char_set.end()); // 정렬된 고유 문자

    int BOS = static_cast<int>(uchars.size()); // BOS (문장 시작) 특수 토큰
    int vocab_size = BOS + 1;                  // 전체 어휘 크기 = 문자 수 + BOS
    std::cout << "vocab size: " << vocab_size << "\n";

    // 문자 → 토큰 ID 변환 테이블
    std::unordered_map<char, int> char_to_id;
    for (int i = 0; i < static_cast<int>(uchars.size()); i++)
        char_to_id[uchars[i]] = i;

    // --- 모델 파라미터 초기화 ---
    // 가우시안 분포(평균=0, 표준편차=0.08)로 가중치를 무작위 초기화
    std::normal_distribution<double> normal(0.0, 0.08);
    std::vector<Value*> params; // 모든 학습 가능한 파라미터의 평탄화 리스트

    // 행렬 생성 헬퍼: nout x nin 크기의 가중치 행렬을 만들고 params에 등록
    auto make_matrix = [&](int nout, int nin) -> Mat {
        Mat m(nout, Vec(nin));
        for (int i = 0; i < nout; i++)
            for (int j = 0; j < nin; j++) {
                auto* v = new Value{normal(rng), 0.0, {}, {}};
                m[i][j] = v;
                params.push_back(v);
            }
        return m;
    };

    // state_dict: Python의 딕셔너리처럼 이름으로 가중치 행렬에 접근
    std::unordered_map<std::string, Mat> state_dict;
    state_dict["wte"]     = make_matrix(vocab_size, N_EMBD);  // 토큰 임베딩
    state_dict["wpe"]     = make_matrix(BLOCK_SIZE, N_EMBD);  // 위치 임베딩
    state_dict["lm_head"] = make_matrix(vocab_size, N_EMBD);  // 출력 프로젝션

    for (int i = 0; i < N_LAYER; i++) {
        std::string pfx = "layer" + std::to_string(i);
        state_dict[pfx + ".attn_wq"] = make_matrix(N_EMBD, N_EMBD);       // 어텐션 Q 가중치
        state_dict[pfx + ".attn_wk"] = make_matrix(N_EMBD, N_EMBD);       // 어텐션 K 가중치
        state_dict[pfx + ".attn_wv"] = make_matrix(N_EMBD, N_EMBD);       // 어텐션 V 가중치
        state_dict[pfx + ".attn_wo"] = make_matrix(N_EMBD, N_EMBD);       // 어텐션 출력 가중치
        state_dict[pfx + ".mlp_fc1"] = make_matrix(4 * N_EMBD, N_EMBD);   // MLP 확장층
        state_dict[pfx + ".mlp_fc2"] = make_matrix(N_EMBD, 4 * N_EMBD);   // MLP 축소층
    }
    std::cout << "num params: " << params.size() << "\n";

    // --- Adam 옵티마이저 (축복받은 최적화기) 및 버퍼 ---
    // Adam은 각 파라미터에 대해 1차 모멘트(평균)와 2차 모멘트(분산)를 추적하여
    // 적응적 학습률로 업데이트한다.
    constexpr double learning_rate = 0.01;
    constexpr double beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
    std::vector<double> m_buf(params.size(), 0.0); // 1차 모멘트 버퍼 (기울기의 이동 평균)
    std::vector<double> v_buf(params.size(), 0.0); // 2차 모멘트 버퍼 (기울기 제곱의 이동 평균)

    // --- 학습 루프 ---
    constexpr int num_steps = 1000;
    for (int step = 0; step < num_steps; step++) {

        // 문서 하나를 선택하고 토큰화, 양쪽을 BOS 토큰으로 감싼다
        // 예: "emma" → [BOS, 'e', 'm', 'm', 'a', BOS]
        const std::string& doc = docs[step % docs.size()];
        std::vector<int> tokens;
        tokens.reserve(doc.size() + 2);
        tokens.push_back(BOS);
        for (char c : doc) tokens.push_back(char_to_id[c]);
        tokens.push_back(BOS);
        int n = std::min(BLOCK_SIZE, static_cast<int>(tokens.size()) - 1);

        // 순전파: 토큰 시퀀스를 모델에 통과시켜 손실까지의 계산 그래프 구축
        std::vector<std::vector<Vec>> keys(N_LAYER), vals(N_LAYER);
        std::vector<Value*> losses;
        for (int pos = 0; pos < n; pos++) {
            int token_id  = tokens[pos];       // 입력 토큰
            int target_id = tokens[pos + 1];   // 예측 대상 (다음 토큰)
            Vec logits = gpt(token_id, pos, keys, vals, state_dict);
            Vec probs  = softmax(logits);
            // 크로스 엔트로피 손실: -log(정답 토큰의 확률)
            losses.push_back(neg(log(probs[target_id])));
        }

        // 시퀀스에 대한 평균 손실. 그대의 손실이 낮기를.
        Value* total_loss = graph.make(0.0);
        for (auto* l : losses) total_loss = add(total_loss, l);
        Value* loss = scale(total_loss, 1.0 / n);

        // 역전파: 모든 파라미터에 대한 기울기를 계산
        backward(loss);

        // Adam 옵티마이저 업데이트
        double lr_t = learning_rate * (1.0 - static_cast<double>(step) / num_steps); // 선형 학습률 감쇠
        for (size_t i = 0; i < params.size(); i++) {
            Value* p = params[i];
            // 1차 모멘트 업데이트 (기울기의 지수 이동 평균)
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p->grad;
            // 2차 모멘트 업데이트 (기울기 제곱의 지수 이동 평균)
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p->grad * p->grad;
            // 바이어스 보정 (초기 스텝에서의 편향 제거)
            double m_hat = m_buf[i] / (1.0 - std::pow(beta1, step + 1));
            double v_hat = v_buf[i] / (1.0 - std::pow(beta2, step + 1));
            // 파라미터 업데이트: p -= lr * m_hat / (sqrt(v_hat) + eps)
            p->data -= lr_t * m_hat / (std::sqrt(v_hat) + eps_adam);
            p->grad = 0.0; // 다음 스텝을 위해 기울기 초기화
        }

        std::printf("\rstep %4d / %4d | loss %.4f", step + 1, num_steps, loss->data);
        std::fflush(stdout);

        // 계산 그래프 해제 (임시 노드만 삭제; 파라미터는 유지)
        graph.clear();
    }

    // --- 추론: 모델이 우리에게 중얼거리기를 ---
    // temperature: (0, 1] 범위, 낮을수록 보수적, 높을수록 창의적
    constexpr double temperature = 0.5;
    std::cout << "\n--- inference (new, hallucinated names) ---\n";

    for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
        graph.clear(); // 이전 샘플의 계산 그래프 해제
        std::vector<std::vector<Vec>> keys(N_LAYER), vals(N_LAYER);
        int token_id = BOS; // BOS 토큰에서 시작
        std::string sample;

        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            Vec logits = gpt(token_id, pos, keys, vals, state_dict);

            // 온도 스케일링: 로짓을 temperature로 나누어 분포 날카로움 조절
            Vec scaled;
            scaled.reserve(logits.size());
            for (auto* l : logits)
                scaled.push_back(scale(l, 1.0 / temperature));
            Vec probs = softmax(scaled);

            // 확률 분포에서 다음 토큰을 샘플링
            std::vector<double> weights;
            weights.reserve(probs.size());
            for (auto* p : probs) weights.push_back(p->data);
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            token_id = dist(rng);

            if (token_id == BOS) break; // BOS를 다시 만나면 생성 종료
            sample += uchars[token_id];
        }

        std::printf("sample %2d: %s\n", sample_idx + 1, sample.c_str());
    }

    // 정리: 모든 파라미터 노드 해제
    graph.clear();
    for (auto* p : params) delete p;

    return 0;
}
