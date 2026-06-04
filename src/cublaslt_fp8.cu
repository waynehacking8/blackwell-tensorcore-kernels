// cuBLASLt FP8 (E4M3) baseline — the library ceiling for Phase 3's FP8 kernel.
//
// cuBLASLt FP8 requires the "TN" layout: both operands K-contiguous. That is the
// same layout our hand-written FP8 kernel uses (A row-major, B transposed), so the
// comparison is layout-for-layout fair. Inputs are quantized once outside the timed
// region (same staging discipline as every other baseline in this harness).
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "util.cuh"

namespace {

__global__ void q_fp8_a(const float* in, uint8_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __nv_fp8_e4m3(in[i]).__x;
}
__global__ void q_fp8_bt(const float* in, uint8_t* out, int K, int N) {
  int k = blockIdx.x * blockDim.x + threadIdx.x, n = blockIdx.y * blockDim.y + threadIdx.y;
  if (k < K && n < N) out[(size_t)n * K + k] = __nv_fp8_e4m3(in[(size_t)k * N + n]).__x;
}

uint8_t *g_A = nullptr, *g_Bt = nullptr;
const float *g_cA = nullptr, *g_cB = nullptr;
int g_M = 0, g_N = 0, g_K = 0;
cublasLtHandle_t g_lt = nullptr;
void* g_ws = nullptr;
constexpr size_t WS_SIZE = 64u << 20;  // 64 MiB workspace (recommended for Lt heuristics)

}  // namespace

void launch_cublaslt_fp8(const float* A, const float* B, float* C, int M, int N, int K) {
  if (M != g_M || N != g_N || K != g_K) {
    cudaFree(g_A); cudaFree(g_Bt);
    g_A = g_Bt = nullptr; g_cA = g_cB = nullptr;
  }
  if (!g_A) {
    CUDA_CHECK(cudaMalloc(&g_A, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&g_Bt, (size_t)N * K));
    g_M = M; g_N = N; g_K = K;
  }
  if (A != g_cA) { int n = M * K; q_fp8_a<<<(n + 255) / 256, 256>>>(A, g_A, n); g_cA = A; }
  if (B != g_cB) {
    dim3 t(16, 16), g((K + 15) / 16, (N + 15) / 16);
    q_fp8_bt<<<g, t>>>(B, g_Bt, K, N);
    g_cB = B;
  }
  if (!g_lt) { cublasLtCreate(&g_lt); CUDA_CHECK(cudaMalloc(&g_ws, WS_SIZE)); }

  // Row-major C = A x B  ==  column-major C^T[N,M] = op(X) * op(Y) with
  //   X = Bt buffer seen as col-major [K x N], opX = T  -> X^T = [N x K]
  //   Y = A  buffer seen as col-major [K x M], opY = N
  // Both operands are K-major = the TN layout FP8 requires.

  // Descriptors are cached per problem size to avoid per-call allocation overhead.
  static cublasLtMatmulDesc_t s_op = nullptr;
  static cublasLtMatrixLayout_t s_la = nullptr, s_lb = nullptr, s_lc = nullptr;
  static cublasLtMatmulPreference_t s_pref = nullptr;
  static int s_dM = 0, s_dN = 0, s_dK = 0;

  if (M != s_dM || N != s_dN || K != s_dK) {
    // Destroy stale descriptors before recreating.
    if (s_pref) { cublasLtMatmulPreferenceDestroy(s_pref); s_pref = nullptr; }
    if (s_lc)   { cublasLtMatrixLayoutDestroy(s_lc);        s_lc   = nullptr; }
    if (s_lb)   { cublasLtMatrixLayoutDestroy(s_lb);        s_lb   = nullptr; }
    if (s_la)   { cublasLtMatrixLayoutDestroy(s_la);        s_la   = nullptr; }
    if (s_op)   { cublasLtMatmulDescDestroy(s_op);          s_op   = nullptr; }

    cublasLtMatmulDescCreate(&s_op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(s_op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(s_op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));

    cublasLtMatrixLayoutCreate(&s_la, CUDA_R_8F_E4M3, K, N, K);  // X: K x N col-major, ld=K
    cublasLtMatrixLayoutCreate(&s_lb, CUDA_R_8F_E4M3, K, M, K);  // Y: K x M col-major, ld=K
    cublasLtMatrixLayoutCreate(&s_lc, CUDA_R_32F,     N, M, N);  // C^T: N x M col-major, ld=N

    cublasLtMatmulPreferenceCreate(&s_pref);
    cublasLtMatmulPreferenceSetAttribute(s_pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &WS_SIZE, sizeof(WS_SIZE));
    s_dM = M; s_dN = N; s_dK = K;
  }

  float alpha = 1.f, beta = 0.f;
  cublasLtMatmulHeuristicResult_t heur;
  int found = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, s_op, s_la, s_lb, s_lc, s_lc, s_pref, 1, &heur, &found);

  cublasStatus_t st = cublasLtMatmul(g_lt, s_op, &alpha, g_Bt, s_la, g_A, s_lb, &beta,
                                     C, s_lc, C, s_lc, found ? &heur.algo : nullptr,
                                     g_ws, WS_SIZE, 0);
  if (st != CUBLAS_STATUS_SUCCESS)
    fprintf(stderr, "cublasLtMatmul FP8 failed: %d (heuristics found=%d)\n", (int)st, found);
}
