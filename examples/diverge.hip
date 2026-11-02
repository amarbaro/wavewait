#include <hip/hip_runtime.h>
#include <cstdio>
// Wave-level divergence for wavewait's `divergent` hint: waves 0-3 of a workgroup loop once, waves 4-7 three
// times. The waves go round robin over the 4 SIMDs, so every SIMD holds both kinds and the traced one sees the
// waves disagree on how often they take the loop branch (0 vs 2 times).
extern "C" __global__ void diverge(float* out, const float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = threadIdx.x / 128 * 2 + 1;
  float s = x[i];
  for (int k = 0; k < n; k++) s = s * 1.5f + x[(i + k * 32) % 49152];
  out[i] = s;
}
int main() {
  const int N = 49152;
  float *dout, *dx;
  hipMalloc(&dout, N * sizeof(float));
  hipMalloc(&dx, N * sizeof(float));
  float* hx = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) hx[i] = (float)(i % 7);
  hipMemcpy(dx, hx, N * sizeof(float), hipMemcpyHostToDevice);
  hipLaunchKernelGGL(diverge, dim3(192), dim3(256), 0, 0, dout, dx);
  hipDeviceSynchronize();
  float ho[2];
  hipMemcpy(ho, dout, 2 * sizeof(float), hipMemcpyDeviceToHost);
  printf("out[0]=%f out[1]=%f\n", ho[0], ho[1]);
  return 0;
}
