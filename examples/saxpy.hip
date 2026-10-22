#include <hip/hip_runtime.h>
#include <cstdio>
extern "C" __global__ void saxpy(float* out, const float* x, const float* y) {
  __shared__ float t[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  t[threadIdx.x] = x[i];
  __syncthreads();
  out[i] = 2.0f * t[255 - threadIdx.x] + y[i];
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
  hipLaunchKernelGGL(saxpy, dim3(192), dim3(256), 0, 0, dout, dx, dy);
  hipDeviceSynchronize();
  float ho[2];
  hipMemcpy(ho,dout,2*sizeof(float),hipMemcpyDeviceToHost);
  printf("out[0]=%f out[1]=%f\n", ho[0], ho[1]);
  return 0;
}
