// A probe kernel for the wavewait hint rules: it carries the patterns the rules look for.
//
//   hoist      a load that is waited on right after it, with nothing to overlap the latency
//   idle wait  a wait with nothing in flight
//   vopd       two independent 32 bit VALU ops the compiler leaves in two slots
//   lds bank   an LDS access whose stride is 16 bytes (four lanes per bank)
//
// The launch is 192 workgroups of 256 threads: a single workgroup does not reach CU 3, the only
// compute unit that emits instruction tokens under the rocprofv3 --att capture (see README.md).
#include <hip/hip_runtime.h>
#include <cstdio>
extern "C" __global__ void probe(float* out, const float* x, const float* y) {
  __shared__ float t[1024];
  int i = threadIdx.x;
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  t[i * 4] = x[g];  // the byte address of t[i * 4] steps by 16 bytes per lane
  __syncthreads();
  float a = y[g];  // a load whose latency the wait right after it exposes
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
  int c = i << 2;
  int d = i >> 3;
  out[g] = t[i * 4] + a + (float)c + (float)d;
}
int main() {
  const int N = 49152;
  float *dout, *dx, *dy;
  hipMalloc(&dout, N*sizeof(float));
  hipMalloc(&dx, N*sizeof(float));
  hipMalloc(&dy, N*sizeof(float));
  float *hx = (float*)malloc(N*sizeof(float));
  float *hy = (float*)malloc(N*sizeof(float));
  for (int i=0;i<N;i++){hx[i]=(float)i; hy[i]=(float)(i+1);}
  hipMemcpy(dx,hx,N*sizeof(float),hipMemcpyHostToDevice);
  hipMemcpy(dy,hy,N*sizeof(float),hipMemcpyHostToDevice);
  hipLaunchKernelGGL(probe, dim3(192), dim3(256), 0, 0, dout, dx, dy);
  hipDeviceSynchronize();
  float ho[2];
  hipMemcpy(ho,dout,2*sizeof(float),hipMemcpyDeviceToHost);
  printf("out[0]=%f out[1]=%f\n", ho[0], ho[1]);
  return 0;
}
