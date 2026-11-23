// mmvk-check: strict correctness checks for the Kepler matvec (mmvk) kernels, after Wafer.ai's guards
// against reward-hacked AI-written kernels. test-backend-ops judges MUL_MAT by NMSE <= 5e-4 over ALL
// outputs, so a handful of wrong or unwritten rows in a large matrix pass it. This tool checks, per case:
//   1. NaN-poisoned output: dst is filled with NaN before every run, so a row the kernel never writes fails.
//   2. Bitwise-identical repeat: the same inputs run twice must give identical bits (races, uninitialised reads).
//   3. Per-element error against an fp64 reference computed from ggml's own dequantisation:
//      |gpu - ref| <= tol * sum_j |w_j * y_j| for EVERY row (tol 2e-5; a wrong row errs by ~1.0 x that sum).
//   4. Weights-buffer mode (--weights): marks the buffer GGML_BACKEND_BUFFER_USAGE_WEIGHTS so caching or
//      repacking paths run; cases reuse freed addresses with NEW contents, which catches stale caches.
// Build and run: scripts/test.sh (builds this against the llama.cpp build it tests)
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ch: channels of y and the output; ch_a: channels of the weights (0 = ch; ch % ch_a == 0 broadcasts weight
// channel c / (ch / ch_a)); off: the weights are a view starting `off` rows into a larger tensor (with an odd
// number of Q8_0 blocks per row that start is only 2-byte aligned).
// n: columns of y and the output (r21: multi-column mmvk; 1 = one-token matvec)
struct shape { ggml_type type; int64_t m, k, ch; int64_t ch_a = 0, off = 0, n = 1; };

static int run_case(ggml_backend_t be, const shape & s, bool weights, std::mt19937 & rng, double & worst) {
    const int64_t m = s.m, k = s.k, ch = s.ch, ch_a = s.ch_a ? s.ch_a : s.ch, off = s.off, mt = m + off, n = s.n;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // weights (k x mt per weight channel), quantised with ggml's reference quantiser
    std::vector<float> wf(k * mt * ch_a), yf(k * n * ch);
    for (auto & v : wf) v = nd(rng);
    for (auto & v : yf) v = nd(rng);
    const size_t row_sz = ggml_row_size(s.type, k);
    std::vector<uint8_t> wq(row_sz * mt * ch_a);
    ggml_quantize_chunk(s.type, wf.data(), wq.data(), 0, mt * ch_a, k, nullptr);
    if (s.type == GGML_TYPE_Q8_0) {
        // uniform int8 quants, -128 and +127 included (the quantiser never emits -128): every byte lane sees
        // both sign boundaries; d stays the quantiser's
        std::uniform_int_distribution<int> bd(-128, 127);
        const size_t bs = ggml_type_size(s.type);
        for (size_t b = 0; b < wq.size() / bs; ++b) for (size_t j = 2; j < bs; ++j) wq[b * bs + j] = (uint8_t) (int8_t) bd(rng);
    }
    // fp64 reference from ggml's own dequantisation of the quantised bytes
    const auto * tt = ggml_get_type_traits(s.type);
    std::vector<float> wd(k);
    std::vector<double> ref(m * n * ch), mag(m * n * ch);
    for (int64_t c = 0; c < ch; ++c) {
        const int64_t ca = c / (ch / ch_a);
        for (int64_t r = 0; r < m; ++r) {
            tt->to_float(wq.data() + (ca * mt + off + r) * row_sz, wd.data(), k);
            for (int64_t col = 0; col < n; ++col) {   // ggml layout: y (k, n, ch), out (m, n, ch)
                double acc = 0.0, a = 0.0;
                for (int64_t j = 0; j < k; ++j) {
                    const double p = (double) wd[j] * (double) yf[(c * n + col) * k + j];
                    acc += p; a += std::fabs(p);
                }
                ref[(c * n + col) * m + r] = acc; mag[(c * n + col) * m + r] = a;
            }
        }
    }
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * AF = ggml_new_tensor_3d(ctx, s.type, k, mt, ch_a);
    ggml_tensor * A = off ? ggml_view_3d(ctx, AF, k, m, ch_a, AF->nb[1], AF->nb[2], off * AF->nb[1]) : AF;
    ggml_tensor * Y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n, ch);
    ggml_tensor * O = ggml_mul_mat(ctx, A, Y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, O);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (weights) ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(AF, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(Y, yf.data(), 0, yf.size() * sizeof(float));
    std::vector<float> nanv(m * n * ch, std::nanf("")), out1(m * n * ch), out2(m * n * ch);
    for (int rep = 0; rep < 2; ++rep) {
        ggml_backend_tensor_set(O, nanv.data(), 0, nanv.size() * sizeof(float));
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(O, rep == 0 ? out1.data() : out2.data(), 0, out1.size() * sizeof(float));
    }
    int nan_rows = 0, bad_rows = 0, first_bad = -1;
    double worst_rel = 0.0;
    for (int64_t i = 0; i < m * n * ch; ++i) {
        if (!std::isfinite(out1[i])) { ++nan_rows; if (first_bad < 0) first_bad = (int) i; continue; }
        const double rel = std::fabs((double) out1[i] - ref[i]) / (mag[i] > 0 ? mag[i] : 1.0);
        if (rel > worst_rel) worst_rel = rel;
        if (rel > 2e-5) { ++bad_rows; if (first_bad < 0) first_bad = (int) i; }
    }
    const bool same = std::memcmp(out1.data(), out2.data(), out1.size() * sizeof(float)) == 0;
    const bool ok = nan_rows == 0 && bad_rows == 0 && same;
    if (worst_rel > worst) worst = worst_rel;
    printf("%s %-5s m=%-6lld k=%-5lld n=%lld ch=%lld/%lld off=%lld %s  max_rel=%.2e  unwritten=%d  wrong=%d  repeat=%s%s\n",
           ok ? "OK  " : "FAIL", ggml_type_name(s.type), (long long) m, (long long) k, (long long) n, (long long) ch, (long long) ch_a, (long long) off,
           weights ? "weights" : "plain  ", worst_rel, nan_rows, bad_rows, same ? "identical" : "DIFFERS",
           first_bad >= 0 ? (std::string("  first_bad_row=") + std::to_string(first_bad)).c_str() : "");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    bool weights = false;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--weights")) weights = true;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 2; }
    // types to check: MMVK_CHECK_TYPES="q4_K,q6_K" (default) or e.g. "q4_K,q6_K,q8_0". Only list a type once
    // mmvk handles it: the stock kernels quantise y to q8_1 and miss the 2e-5 limit by design.
    std::vector<ggml_type> types;
    {
        const char * env = std::getenv("MMVK_CHECK_TYPES");
        const std::string list = env && *env ? env : "q4_K,q6_K";
        // r27: Q5_K added (Q5_K_M files; 176-byte blocks, 16-byte aligned rows, the generic shapes below apply)
        for (ggml_type t : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_F16, GGML_TYPE_Q3_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q5_K }) {
            if (("," + list + ",").find("," + std::string(ggml_type_name(t)) + ",") != std::string::npos) types.push_back(t);
        }
    }
    std::vector<shape> cases;
    for (ggml_type t : types) {
        if (t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q5_0) {
            // Q8_0 blocks are 32 weights (34 bytes, 2-byte aligned rows): k not a multiple of 256, odd block
            // counts, one block, and the Qwen3-0.6B shapes
            // (lists after AGY Neon's review, consult/neon-r8-q8.md)
            for (int64_t k : { 32, 64, 96, 128, 160, 192, 256, 288, 640, 1056, 2048, 3072 })
                for (int64_t m : { 1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65 }) cases.push_back({ t, m, k, 1 });
            cases.push_back({ t, 2048, 1024, 1 });
            // views starting 2 bytes off a 4-byte boundary (k = 32 or 96: odd blocks per row, off = 1 row)
            for (int64_t k : { 32, 96, 1056 }) for (int64_t m : { 5, 64 }) cases.push_back({ t, m, k, 1, 0, 1 });
            // channel broadcast: 2 weight channels, 6 y channels; odd and even blocks per row
            for (int64_t k : { 96, 1024 }) cases.push_back({ t, 17, k, 6, 2, 0 });
        }
        if (t == GGML_TYPE_F16) {
            // attention in one-token generation (non-flash): KQ = K (k = head dim, m = n_kv) and KQV = V^T
            // (k = n_kv, m = head dim), 16 query heads over 8 or 2 KV heads (GQA broadcast), odd n_kv tails
            for (int64_t hd : { 64, 128, 256 }) for (int64_t kv : { 1, 7, 33, 256, 1000, 4096 }) {
                cases.push_back({ t, kv, hd, 16, 8, 0 });
                cases.push_back({ t, hd, kv, 16, 8, 0 });
            }
            for (int64_t kv : { 5, 300 }) { cases.push_back({ t, kv, 64, 16, 2, 0 }); cases.push_back({ t, 64, kv, 16, 2, 0 }); }
            continue;   // the quantised-weight shapes below do not apply
        }
        // the model's shapes, the head, tails (rows not a multiple of 2/4/8/16), odd and tiny block counts,
        // and channel batches (ne2 > 1)
        for (int64_t m : { 512, 1024, 3072, 4608 }) cases.push_back({ t, m, 1024, 1 });
        cases.push_back({ t, 1024, 4608, 1 });
        cases.push_back({ t, 65536, 1024, 1 });
        for (int64_t m : { 1, 2, 3, 5, 7, 9, 13, 17, 33, 4097 }) cases.push_back({ t, m, 1024, 1 });
        for (int64_t k : { 256, 512, 768, 2304, 4352 }) for (int64_t m : { 7, 64 }) cases.push_back({ t, m, k, 1 });
        cases.push_back({ t, 64, 14336, 1 });   // nb 56: too long for Q4_K `sy` shared memory (fallback path)
        cases.push_back({ t, 33, 1024, 3 });
        cases.push_back({ t, 64, 768, 2 });
        if (t == GGML_TYPE_Q3_K || t == GGML_TYPE_IQ3_XXS) {
            // LFM3-1.7B shapes (attn_gate 16 rows, kv 512, ffn 7680); block_iq3_xxs is 98 bytes (2-byte aligned rows)
            for (int64_t m : { 16, 512, 7680 }) cases.push_back({ t, m, 2048, 1 });
            cases.push_back({ t, 2048, 7680, 1 });
            // r15: Qwen3-1.7B Q3_K_S shapes (hidden 2048, kv 1024, ffn 6144); block_q3_K is 110 bytes, so rows are
            // only 2-byte aligned when nb is odd and views can start 2 bytes off (off = 1 row at nb = 1, 3, 9)
            for (int64_t m : { 1024, 2048, 6144 }) cases.push_back({ t, m, 2048, 1 });
            cases.push_back({ t, 2048, 6144, 1 });
            for (int64_t k : { 256, 768, 2304 }) for (int64_t m : { 5, 64 }) cases.push_back({ t, m, k, 1, 0, 1 });
            for (int64_t k : { 768, 2048 }) cases.push_back({ t, 17, k, 6, 2, 0 });
        }
    }
    // r21: multi-column cases (n = 2..8) for the types listed in MMVK_CHECK_NCOLS (e.g. "q3_K"): only once mmvk
    // takes ne11 > 1 for that type; the stock multi-column kernels quantise y to q8_1 and miss 2e-5 by design.
    {
        const char * env = std::getenv("MMVK_CHECK_NCOLS");
        const std::string list = env ? env : "";
        std::vector<shape> extra;
        for (ggml_type t : types) {
            if (list.empty() || ("," + list + ",").find("," + std::string(ggml_type_name(t)) + ",") == std::string::npos) continue;
            for (int64_t n : { 2, 3, 4, 5, 8 }) {
                for (int64_t m : { 1, 7, 64, 2048 }) extra.push_back({ t, m, 2048, 1, 0, 0, n });
                extra.push_back({ t, 1536, 8960, 1, 0, 0, n });                              // odd block count (Q3_K: 35)
                for (int64_t k : { 256, 768 }) extra.push_back({ t, 64, k, 1, 0, 1, n });   // 2-byte-offset views
                extra.push_back({ t, 17, 2048, 6, 2, 0, n });                                // channel broadcast
            }
        }
        cases.insert(cases.end(), extra.begin(), extra.end());
    }
    std::mt19937 rng(1234);
    int fails = 0; double worst = 0.0;
    for (const auto & s : cases) fails += run_case(be, s, weights, rng, worst);
    printf("SUMMARY %s: %d cases, %d FAIL, worst max_rel %.2e (limit 2e-5)\n",
           weights ? "weights" : "plain", (int) cases.size(), fails, worst);
    ggml_backend_free(be);
    return fails ? 1 : 0;
}
