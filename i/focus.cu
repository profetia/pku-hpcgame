#include <stdint.h>
#include <stdio.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

using Point = double3;

static __device__ __forceinline__ double distance(Point a, Point b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  double dz = a.z - b.z;
  return sqrt(dx * dx + dy * dy + dz * dz);
}

constexpr int BLOCK_SIZE = 1024;

__global__ void focus(const Point kSrc, const Point* mirs, const int64_t kMirN,
                      const Point* sens, const int64_t kSenN, double* illums) {
  int sen_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (sen_idx >= kSenN) return;

  double a = 0, b = 0;
  for (int mir_idx = 0; mir_idx < kMirN; ++mir_idx) {
    double l =
        distance(mirs[mir_idx], kSrc) + distance(mirs[mir_idx], sens[sen_idx]);

    a += cos(6.283185307179586 * 2000 * l);
    b += sin(6.283185307179586 * 2000 * l);
  }

  illums[sen_idx] = sqrt(a * a + b * b);
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

  double* d_illums;
  cudaMalloc(&d_illums, kSenN * sizeof(double));

  cudaMemcpy(d_mirs, mirs, kMirN * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sens, sens, kSenN * sizeof(Point), cudaMemcpyHostToDevice);

  {
    int kBlockSize = BLOCK_SIZE;
    int kGridSize = CEIL(kSenN, kBlockSize);
    focus<<<kGridSize, kBlockSize>>>(kSrc, d_mirs, kMirN, d_sens, kSenN,
                                     d_illums);
  };

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