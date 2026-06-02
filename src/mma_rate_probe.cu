// FP32-accumulate tensor rate probe (roadmap Phase 4, last item).
//
// Question: is dense FP16-in/FP32-acc mma.sync on this card full-rate or half-rate
// relative to FP16-in/FP16-acc? Consumer GB202 (RTX 5090) runs FP32-acc at HALF the
// FP16-acc tensor rate; the workstation RTX PRO 6000 (same die) has no public spec.
//
// Method: register-resident back-to-back mma.sync.aligned.m16n8k16 loop — no global
// or shared memory traffic in the timed region, one block per SM, 8 warps per block.
// Each warp drives CHAINS independent accumulator chains (D += A*B with the
// accumulator feeding back), so each mma instruction depends only on its own chain
// — the loop measures issue *throughput*, not instruction latency. The only
// difference between the two variants is the accumulator type (.f32 vs .f16).
//
// Output: TFLOP/s for both variants, interleaved over several reps. The ratio
// FP32acc / FP16acc is the read-out (1.0 = full-rate, 0.5 = half-rate). The
// interleave makes the ratio robust against clock drift on this Max-Q (300 W)
// card, where clocks cannot be pinned without root.
//
// Usage: mma_rate_probe [csv_out]
//   Each rep prints: variant, ms, TFLOP/s, SM clock (MHz, sampled via NVML).
#include <cuda_fp16.h>
#include <nvml.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "mma_ptx.cuh"  // mma_m16n8k16 (FP32-acc) — reuse the exact kernel-path PTX wrapper

#define CK(call)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA error %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

// FP16-accumulate variant: D(16x8,f16; 2 regs) += A(16x16,f16; 4 regs) x B(16x8,f16; 2 regs).
// Identical instruction shape to mma_m16n8k16 in mma_ptx.cuh — only the accumulator type differs.
__device__ __forceinline__ void mma_m16n8k16_f16acc(unsigned d[2], const unsigned a[4],
                                                    const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 8 independent accumulator chains per warp: enough ILP to saturate the tensor pipe,
// few enough registers (FP32: 8*4 + 6 operands = 38/thread) to stay far from spilling.
constexpr int kChains = 8;
constexpr int kInnerIters = 32768;  // ~5-10 ms per launch: long enough for a clean clock sample
constexpr int kWarpsPerBlock = 8;
constexpr int kFlopsPerMma = 16 * 8 * 16 * 2;  // m16n8k16 MACs * 2

// Each block also measures its own elapsed SM cycles with clock64(): cycles are counted by
// the same clock the SM executes at, so FLOP/SM/clk derived from them is exact regardless of
// clock-frequency oscillation under the 300 W power cap (NVML clock readings lag ~1 s, too
// coarse for back-to-back launches). Block 0's cycle count is written to cycles_out[launch].
__global__ void __launch_bounds__(32 * kWarpsPerBlock, 1) rate_f32acc(float* out,
                                                                      long long* cycles_out,
                                                                      int launch_idx) {
  // Operand values: small magnitudes derived from the lane id; never read from memory.
  unsigned lane = threadIdx.x & 31u;
  unsigned a[4], b[2];
  half2 av = __floats2half2_rn(0.001f * lane, 0.002f);
  half2 bv = __floats2half2_rn(0.001f, -0.001f * lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<unsigned*>(&av);
#pragma unroll
  for (int i = 0; i < 2; ++i) b[i] = *reinterpret_cast<unsigned*>(&bv);

  long long t0 = clock64();
  float d[kChains][4] = {};
  for (int it = 0; it < kInnerIters; ++it) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) mma_m16n8k16(d[c], a, b);
  }

  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < kChains; ++c) acc += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  if (acc == -1.f) out[blockIdx.x] = acc;  // never true: keeps the loop from being eliminated
  if (blockIdx.x == 0 && threadIdx.x == 0) cycles_out[launch_idx] = clock64() - t0;
}

__global__ void __launch_bounds__(32 * kWarpsPerBlock, 1) rate_f16acc(float* out,
                                                                      long long* cycles_out,
                                                                      int launch_idx) {
  unsigned lane = threadIdx.x & 31u;
  unsigned a[4], b[2];
  half2 av = __floats2half2_rn(0.001f * lane, 0.002f);
  half2 bv = __floats2half2_rn(0.001f, -0.001f * lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<unsigned*>(&av);
#pragma unroll
  for (int i = 0; i < 2; ++i) b[i] = *reinterpret_cast<unsigned*>(&bv);

  long long t0 = clock64();
  unsigned d[kChains][2] = {};
  for (int it = 0; it < kInnerIters; ++it) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) mma_m16n8k16_f16acc(d[c], a, b);
  }

  unsigned acc = 0;
#pragma unroll
  for (int c = 0; c < kChains; ++c) acc += d[c][0] + d[c][1];
  if (acc == 0xdeadbeefu) out[blockIdx.x] = 1.f;  // never true: keeps the loop live
  if (blockIdx.x == 0 && threadIdx.x == 0) cycles_out[launch_idx] = clock64() - t0;
}

struct Rep {
  double ms, tflops;        // per-launch average over the sustained stream
  double flop_per_sm_clk;   // from in-kernel clock64() cycles — clock-oscillation-immune
  unsigned power_mw_mid;
};

// Launch `launches` kernels back-to-back (~0.7 s of sustained work) and time the
// whole stream. A single 3.6 ms launch finishes before the GPU leaves idle clocks;
// the sustained stream is what reaches the boost clock the GEMM benchmarks run at.
// Per-clock throughput comes from in-kernel clock64() cycle counts (exact at any
// clock); host-side power is sampled mid-stream for the session record.
template <typename Kernel>
Rep run_rep(Kernel kernel, int blocks, float* d_out, long long* d_cycles, nvmlDevice_t nvml_dev,
            int launches) {
  cudaEvent_t start, stop;
  CK(cudaEventCreate(&start));
  CK(cudaEventCreate(&stop));

  kernel<<<blocks, 32 * kWarpsPerBlock>>>(d_out, d_cycles, 0);  // warm-up
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  CK(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i) kernel<<<blocks, 32 * kWarpsPerBlock>>>(d_out, d_cycles, i);
  CK(cudaEventRecord(stop));

  // Sample power while the stream runs (clock comes from in-kernel cycle counts).
  double mw_sum = 0;
  long n = 0;
  while (cudaEventQuery(stop) == cudaErrorNotReady) {
    unsigned mw_now = 0;
    nvmlDeviceGetPowerUsage(nvml_dev, &mw_now);
    mw_sum += mw_now;
    ++n;
    usleep(1000);
  }
  unsigned mw = n ? unsigned(mw_sum / n) : 0;

  CK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, start, stop));

  // Median per-launch cycle count (block 0) -> FLOP per SM per clock.
  std::vector<long long> cycles(launches);
  CK(cudaMemcpy(cycles.data(), d_cycles, launches * sizeof(long long), cudaMemcpyDeviceToHost));
  std::sort(cycles.begin(), cycles.end());
  double med_cycles = double(cycles[launches / 2]);
  double mmas_per_block = double(kWarpsPerBlock) * kInnerIters * kChains;
  double flop_per_sm_clk = mmas_per_block * kFlopsPerMma / med_cycles;

  double mmas = double(blocks) * kWarpsPerBlock * kInnerIters * kChains * launches;
  double tflops = mmas * kFlopsPerMma / (ms * 1e-3) / 1e12;
  CK(cudaEventDestroy(start));
  CK(cudaEventDestroy(stop));
  return {ms / launches, tflops, flop_per_sm_clk, mw};
}

int main(int argc, char** argv) {
  const char* csv_path = argc > 1 ? argv[1] : nullptr;

  int dev = 0;
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, dev));

  nvmlInit();
  nvmlDevice_t nvml_dev{};
  nvmlDeviceGetHandleByIndex(0, &nvml_dev);

  int blocks = prop.multiProcessorCount;  // one block per SM
  constexpr int kReps = 5;
  constexpr int kLaunches = 200;

  float* d_out = nullptr;
  long long* d_cycles = nullptr;
  CK(cudaMalloc(&d_out, blocks * sizeof(float)));
  CK(cudaMalloc(&d_cycles, kLaunches * sizeof(long long)));

  printf("# %s, %d SMs, one block/SM x %d warps, %d chains x %d iters\n", prop.name,
         blocks, kWarpsPerBlock, kChains, kInnerIters);
  printf("rep,variant,ms_per_launch,tflops,flop_per_sm_clk,power_w\n");

  // Warm up to sustained (power-capped) clocks before the first timed rep, so rep 0
  // is measured under the same conditions as the rest.
  for (int i = 0; i < 400; ++i) rate_f32acc<<<blocks, 32 * kWarpsPerBlock>>>(d_out, d_cycles, 0);
  CK(cudaDeviceSynchronize());

  // Interleave the two variants so clock drift hits both equally. Each rep is a
  // sustained ~0.7 s stream of launches; power sampled while it runs.
  std::vector<Rep> f16r, f32r;
  std::string csv = "rep,variant,ms_per_launch,tflops,flop_per_sm_clk,power_w\n";
  char line[128];
  for (int r = 0; r < kReps; ++r) {
    Rep h = run_rep(rate_f16acc, blocks, d_out, d_cycles, nvml_dev, kLaunches);
    Rep s = run_rep(rate_f32acc, blocks, d_out, d_cycles, nvml_dev, kLaunches);
    f16r.push_back(h);
    f32r.push_back(s);
    snprintf(line, sizeof line, "%d,f16acc,%.3f,%.1f,%.1f,%.1f\n", r, h.ms, h.tflops,
             h.flop_per_sm_clk, h.power_mw_mid / 1000.0);
    printf("%s", line);
    csv += line;
    snprintf(line, sizeof line, "%d,f32acc,%.3f,%.1f,%.1f,%.1f\n", r, s.ms, s.tflops,
             s.flop_per_sm_clk, s.power_mw_mid / 1000.0);
    printf("%s", line);
    csv += line;
  }

  auto med = [](std::vector<Rep> v, auto key) {
    std::sort(v.begin(), v.end(), [&](const Rep& a, const Rep& b) { return key(a) < key(b); });
    return key(v[v.size() / 2]);
  };
  auto tfl = [](const Rep& r) { return r.tflops; };
  // FLOP per SM per clock (from in-kernel cycle counts) removes the clock difference between
  // the two runs — on a power-capped card the FP32-acc variant draws more power and clocks a
  // few % lower, so the raw TFLOP/s ratio under-reports the per-clock instruction rate.
  auto per_clk = [](const Rep& r) { return r.flop_per_sm_clk; };

  double f16_med = med(f16r, tfl), f32_med = med(f32r, tfl);
  double f16_clk = med(f16r, per_clk), f32_clk = med(f32r, per_clk);
  double ratio = f32_med / f16_med;
  double ratio_clk = f32_clk / f16_clk;

  printf("#\n# median f16acc: %.1f TFLOP/s  (%.0f FLOP/SM/clk)\n", f16_med, f16_clk);
  printf("# median f32acc: %.1f TFLOP/s  (%.0f FLOP/SM/clk)\n", f32_med, f32_clk);
  printf("# ratio f32acc/f16acc (raw):       %.3f\n", ratio);
  printf("# ratio f32acc/f16acc (per clock): %.3f  (1.0 = full-rate, 0.5 = half-rate)\n", ratio_clk);

  if (csv_path) {
    FILE* f = fopen(csv_path, "w");
    if (!f) {
      fprintf(stderr, "cannot open %s\n", csv_path);
      return 1;
    }
    fprintf(f, "# %s, %d SMs, %d warps/block, %d chains, %d iters\n", prop.name, blocks,
            kWarpsPerBlock, kChains, kInnerIters);
    fputs(csv.c_str(), f);
    fprintf(f, "# median_f16acc_tflops,%.1f\n# median_f32acc_tflops,%.1f\n", f16_med, f32_med);
    fprintf(f, "# median_f16acc_flop_per_sm_clk,%.1f\n# median_f32acc_flop_per_sm_clk,%.1f\n",
            f16_clk, f32_clk);
    fprintf(f, "# ratio_f32_over_f16_raw,%.3f\n# ratio_f32_over_f16_per_clock,%.3f\n", ratio,
            ratio_clk);
    fclose(f);
  }

  CK(cudaFree(d_out));
  CK(cudaFree(d_cycles));
  nvmlShutdown();
  return 0;
}
