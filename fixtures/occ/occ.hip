#include <hip/hip_runtime.h>
template <int V, int L> __global__ __launch_bounds__(256) void k(float* o, const float* a) {
  __shared__ float s[L > 0 ? L : 1];
  float r[V];
#pragma unroll
  for (int i = 0; i < V; i++) r[i] = a[threadIdx.x * V + i];
  if (L > 0) { s[threadIdx.x % (L > 0 ? L : 1)] = r[0]; __syncthreads(); r[1] += s[(threadIdx.x + 1) % (L > 0 ? L : 1)]; }
  float t = 0;
#pragma unroll
  for (int i = 0; i < V; i++) t += r[i] * r[(i * 7 + 3) % V];
  o[threadIdx.x] = t;
}
template __global__ void k<4, 0>(float*, const float*);
template __global__ void k<64, 0>(float*, const float*);
template __global__ void k<100, 0>(float*, const float*);
template __global__ void k<200, 0>(float*, const float*);
template __global__ void k<4, 4096>(float*, const float*);
template __global__ void k<4, 12000>(float*, const float*);
template __global__ void k<4, 5450>(float*, const float*);
template __global__ void k<4, 5100>(float*, const float*);
