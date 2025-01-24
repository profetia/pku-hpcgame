#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>

const int ROUNDS = 5;

void cuda_posit_gemm_d(const uint64_t *dA, const uint64_t *dB, uint64_t *dC,
                       int n, int m, int k);

void cuda_posit_gemm(const uint64_t *hA, const uint64_t *hB, uint64_t *hC,
                     int n, int m, int k) {
    uint64_t *dA, *dB, *dC;
    cudaMalloc((void **)&dA, (size_t)n * k * sizeof(uint64_t));
    cudaMalloc((void **)&dB, (size_t)k * m * sizeof(uint64_t));
    cudaMalloc((void **)&dC, (size_t)n * m * sizeof(uint64_t));
    cudaMemcpy(dA, hA, (size_t)n * k * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)k * m * sizeof(uint64_t),
               cudaMemcpyHostToDevice);

    cuda_posit_gemm_d(dA, dB, dC, n, m, k);
    cudaDeviceSynchronize();

    double elapsed[ROUNDS];
    double mean = 0, std = 0;
    for (int i = 0; i < ROUNDS; ++i) {
        auto start = std::chrono::steady_clock::now();
        cuda_posit_gemm_d(dA, dB, dC, n, m, k);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        elapsed[i] = std::chrono::duration<double>(end - start).count();
        mean += elapsed[i];
    }
    mean /= ROUNDS;
    for (int i = 0; i < ROUNDS; ++i) {
        std += (elapsed[i] - mean) * (elapsed[i] - mean);
    }
    std = std::sqrt(std / ROUNDS);
    printf("mean %.3lf ms\nstd %.3lf ms\n", mean * 1e3, std::sqrt(std) * 1e3);

    cudaMemcpy(hC, dC, (size_t)n * m * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char **argv) {
    assert(argc == 4);
    std::ifstream in1(argv[1], std::ios::binary),
        in2(argv[2], std::ios::binary);
    std::ofstream out(argv[3], std::ios::binary);
    int n, m, k, k2;
    in1.read((char *)&n, sizeof(int));
    in1.read((char *)&k, sizeof(int));
    in2.read((char *)&k2, sizeof(int));
    in2.read((char *)&m, sizeof(int));
    assert(k == k2);

    uint64_t *hA, *hB, *hC;
    hA = new uint64_t[n * k];
    hB = new uint64_t[k * m];
    hC = new uint64_t[n * m];
    in1.read((char *)hA, (size_t)n * k * 8);
    in2.read((char *)hB, (size_t)k * m * 8);
    assert(in1);
    assert(in2);

    cuda_posit_gemm(hA, hB, hC, n, m, k);
    out.write((char *)&n, sizeof(int));
    out.write((char *)&m, sizeof(int));
    out.write((char *)hC, (size_t)n * m * 8);
}