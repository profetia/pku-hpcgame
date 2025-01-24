#include <stdint.h>
#include <stdio.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define FLOAT4(a) *((float4*)&(a))
#define DOUBLE4(a) *((double4*)&(a))

using Point = double3;

static __device__ __forceinline__ double distance(Point a, Point b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  double dz = a.z - b.z;
  return sqrt(dx * dx + dy * dy + dz * dz);
}

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = WARP_SIZE;
constexpr int WINDOW_SIZE = 8;

__global__ void focus_ab(const Point kSrc, const Point* mirs, const int kMirN,
                         const Point* sens, const int kSenN, float* a,
                         float* b) {
  int kMirOffset = blockIdx.x * blockDim.x * WINDOW_SIZE;
  int kSenOffset = blockIdx.y * blockDim.y * WINDOW_SIZE;
  int kWarpIdx = threadIdx.y;
  int kWarpLane = threadIdx.x;

  __shared__ Point local_mirs[BLOCK_SIZE * WINDOW_SIZE];
  __shared__ Point local_sens[BLOCK_SIZE * WINDOW_SIZE];

  if (kWarpIdx < WINDOW_SIZE) {
    int kMirLocalIdx = kWarpIdx * BLOCK_SIZE + kWarpLane;
    local_mirs[kMirLocalIdx] = mirs[kMirOffset + kMirLocalIdx];
  } else if (kWarpIdx < WINDOW_SIZE * 2) {
    int kSenLocalIdx = (kWarpIdx - WINDOW_SIZE) * BLOCK_SIZE + kWarpLane;
    local_sens[kSenLocalIdx] = sens[kSenOffset + kSenLocalIdx];
  }

  __syncthreads();

  float local_a[WINDOW_SIZE] = {0};
  float local_b[WINDOW_SIZE] = {0};

  for (int i = 0; i < WINDOW_SIZE; ++i) {
    Point sen = local_sens[kWarpIdx * WINDOW_SIZE + i];
#pragma unroll
    for (int j = 0; j < WINDOW_SIZE; ++j) {
      Point mir = local_mirs[kWarpLane * WINDOW_SIZE + j];
      double l = distance(mir, kSrc) + distance(mir, sen);
      double a, b;
      sincos(6.283185307179586 * 2000 * l, &b, &a);

      local_a[i] += static_cast<float>(a);
      local_b[i] += static_cast<float>(b);
    }

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      local_a[i] += __shfl_down_sync(0xffffffff, local_a[i], offset);
      local_b[i] += __shfl_down_sync(0xffffffff, local_b[i], offset);
    }
  }

  if (kWarpLane == 0) {
#pragma unroll
    for (int i = 0; i < WINDOW_SIZE; ++i) {
      atomicAdd(&a[kSenOffset + kWarpIdx * WINDOW_SIZE + i], local_a[i]);
      atomicAdd(&b[kSenOffset + kWarpIdx * WINDOW_SIZE + i], local_b[i]);
    }
  }
}

__global__ void focus_illum(const float* a, const float* b, const int kSenN,
                            double* illums) {
  int kSenIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (kSenIdx < kSenN) {
    float4 a4 = FLOAT4(a[kSenIdx]);
    float4 b4 = FLOAT4(b[kSenIdx]);

    double4 illum4;
    illum4.x = static_cast<double>(sqrt(a4.x * a4.x + b4.x * b4.x));
    illum4.y = static_cast<double>(sqrt(a4.y * a4.y + b4.y * b4.y));
    illum4.z = static_cast<double>(sqrt(a4.z * a4.z + b4.z * b4.z));
    illum4.w = static_cast<double>(sqrt(a4.w * a4.w + b4.w * b4.w));

    DOUBLE4(illums[kSenIdx]) = illum4;
  }
}

int main() {
  Point kSrc;
  int64_t kMirN, kSenN;
  Point *mirs, *sens;

  {
    FILE* infile = fopen("in.data", "rb");
    fread(&kSrc, sizeof(Point), 1, infile);

    fread(&kMirN, sizeof(int64_t), 1, infile);
    cudaHostAlloc(&mirs, kMirN * sizeof(Point), cudaHostAllocDefault);
    fread(mirs, sizeof(Point), kMirN, infile);

    fread(&kSenN, sizeof(int64_t), 1, infile);
    cudaHostAlloc(&sens, kSenN * sizeof(Point), cudaHostAllocDefault);
    fread(sens, sizeof(Point), kSenN, infile);

    fclose(infile);
  }

  double* illums;
  cudaHostAlloc(&illums, kSenN * sizeof(double), cudaHostAllocDefault);

  Point *d_mirs, *d_sens;
  cudaMalloc(&d_mirs, kMirN * sizeof(Point));
  cudaMalloc(&d_sens, kSenN * sizeof(Point));

  float *d_a, *d_b;
  cudaMalloc(&d_a, kSenN * sizeof(float));
  cudaMalloc(&d_b, kSenN * sizeof(float));

  double* d_illums;
  cudaMalloc(&d_illums, kSenN * sizeof(double));

  cudaMemcpy(d_mirs, mirs, kMirN * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sens, sens, kSenN * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemset(d_a, 0, kSenN * sizeof(float));
  cudaMemset(d_b, 0, kSenN * sizeof(float));

  {
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(CEIL(CEIL(kMirN, WINDOW_SIZE), BLOCK_SIZE),
                  CEIL(CEIL(kSenN, WINDOW_SIZE), BLOCK_SIZE));

    focus_ab<<<grid_dim, block_dim>>>(kSrc, d_mirs, kMirN, d_sens, kSenN, d_a,
                                      d_b);
  };

  {
    int block_dim = 1024;
    int grid_dim = CEIL(CEIL(kSenN, 4), block_dim);

    focus_illum<<<grid_dim, block_dim>>>(d_a, d_b, kSenN, d_illums);
  }

  cudaMemcpy(illums, d_illums, kSenN * sizeof(double), cudaMemcpyDeviceToHost);

  {
    FILE* outfile = fopen("out.data", "wb");
    fwrite(illums, sizeof(double), kSenN, outfile);
    fclose(outfile);
  }

  cudaFree(d_mirs);
  cudaFree(d_sens);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_illums);

  cudaFreeHost(mirs);
  cudaFreeHost(sens);
  cudaFreeHost(illums);

  return 0;
}