#include <stdint.h>
#include <stdio.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
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

__global__ void focus_ab(const Point kSrc, const Point* mirs, const int kMirN,
                         const Point* sens, const int kSenN, double* a,
                         double* b) {
  int kMirOffset = blockIdx.x * blockDim.x;
  int kSenOffset = blockIdx.y * blockDim.y;
  int kWarpIdx = threadIdx.y;
  int kWarpLane = threadIdx.x;

  __shared__ Point mirs_local[BLOCK_SIZE];
  __shared__ Point sens_local[BLOCK_SIZE];

  if (kWarpIdx == 0) {
    mirs_local[kWarpLane] = mirs[kMirOffset + kWarpLane];
  } else if (kWarpIdx == 1) {
    sens_local[kWarpLane] = sens[kSenOffset + kWarpLane];
  }

  __syncthreads();

  double l = distance(mirs_local[kWarpLane], kSrc) +
             distance(mirs_local[kWarpLane], sens_local[kWarpIdx]);
  double local_a = cos(6.283185307179586 * 2000 * l);
  double local_b = sin(6.283185307179586 * 2000 * l);

  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    local_a += __shfl_down_sync(0xffffffff, local_a, offset);
    local_b += __shfl_down_sync(0xffffffff, local_b, offset);
  }

  if (kWarpLane == 0) {
    atomicAdd(&a[kSenOffset + kWarpIdx], local_a);
    atomicAdd(&b[kSenOffset + kWarpIdx], local_b);
  }
}

__global__ void focus_illum(const double* a, const double* b, const int kSenN,
                            double* illums) {
  int kSenIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (kSenIdx < kSenN) {
    double4 a4 = DOUBLE4(a[kSenIdx]);
    double4 b4 = DOUBLE4(b[kSenIdx]);

    double4 illum4;
    illum4.x = sqrt(a4.x * a4.x + b4.x * b4.x);
    illum4.y = sqrt(a4.y * a4.y + b4.y * b4.y);
    illum4.z = sqrt(a4.z * a4.z + b4.z * b4.z);
    illum4.w = sqrt(a4.w * a4.w + b4.w * b4.w);

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

  double *d_a, *d_b;
  cudaMalloc(&d_a, kSenN * sizeof(double));
  cudaMalloc(&d_b, kSenN * sizeof(double));

  double* d_illums;
  cudaMalloc(&d_illums, kSenN * sizeof(double));

  cudaMemcpy(d_mirs, mirs, kMirN * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sens, sens, kSenN * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemset(d_a, 0, kSenN * sizeof(double));
  cudaMemset(d_b, 0, kSenN * sizeof(double));

  {
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(CEIL(kMirN, BLOCK_SIZE), CEIL(kSenN, BLOCK_SIZE));

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
  cudaFree(d_illums);

  cudaFreeHost(mirs);
  cudaFreeHost(sens);
  cudaFreeHost(illums);

  return 0;
}