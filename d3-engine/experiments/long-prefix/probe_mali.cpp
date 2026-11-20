// Time only QK^T with resident inputs. If it already loses to full attention,
// this FP32 CLBlast path cannot improve the complete segment on this workload.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "gpu_host.cpp"

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: probe_mali gpu_kernels.cl qk.bin");
        std::ifstream input(argv[2], std::ios::binary);
        input.exceptions(std::ios::badbit | std::ios::failbit);
        uint32_t shape[4];
        float scale;
        input.read(reinterpret_cast<char *>(shape), sizeof(shape));
        input.read(reinterpret_cast<char *>(&scale), sizeof(scale));
        const size_t heads = shape[0], rows = shape[1], keys = shape[2], dim = shape[3];
        if (heads != 2 || !rows || rows > 8192 || !keys || keys > 16384 || dim != 256 || !std::isfinite(scale))
            throw std::runtime_error("unexpected question capture dimensions");
        std::vector<float> q(heads * rows * dim), k(heads * keys * dim);
        input.read(reinterpret_cast<char *>(q.data()), q.size() * sizeof(float));
        input.read(reinterpret_cast<char *>(k.data()), k.size() * sizeof(float));
        std::unique_ptr<D3GPU> gpu(static_cast<D3GPU *>(d3_gpu_create(argv[1])));
        if (!gpu) throw std::runtime_error(d3_gpu_error());
        char name[256] = {};
        check(clGetDeviceInfo(gpu->device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "device name");
        gpu->write(0, q.data(), q.size() * sizeof(float));
        gpu->write(1, k.data(), k.size() * sizeof(float));
        gpu->buffer(2, heads * rows * keys * sizeof(float));
        check(clFinish(gpu->queue), "initial upload");
        std::cout << std::setprecision(9);
        for (int trial = 0; trial < 3; ++trial) {
            auto start = std::chrono::steady_clock::now();
            auto rc = CLBlastSgemmStridedBatched(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes,
                rows, keys, dim, scale, gpu->bufs[0], 0, dim, rows * dim, gpu->bufs[1], 0, dim, keys * dim,
                0.f, gpu->bufs[2], 0, keys, rows * keys, heads, &gpu->queue, nullptr);
            check(static_cast<cl_int>(rc), "CLBlast QK");
            check(clFinish(gpu->queue), "wait for GPU");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            double max_error = 0;
            for (size_t h = 0; h < heads; ++h) {
                for (size_t i = 0; i < 4; ++i) {
                    size_t row = (rows - 1) * i / 3, key = (keys - 1) * (3 - i) / 3;
                    double expected = 0;
                    for (size_t d = 0; d < dim; ++d)
                        expected += double(q[(h * rows + row) * dim + d]) * k[(h * keys + key) * dim + d];
                    float score;
                    size_t offset = ((h * rows + row) * keys + key) * sizeof(float);
                    check(clEnqueueReadBuffer(gpu->queue, gpu->bufs[2], CL_TRUE, offset, sizeof(score), &score,
                                              0, nullptr, nullptr), "verify GPU score");
                    double delta = std::abs(score - expected * scale);
                    if (!std::isfinite(delta) || delta > 1e-4 * (1 + std::abs(expected * scale)))
                        throw std::runtime_error("GPU score differs from the captured Q/K dot product");
                    max_error = std::max(max_error, delta);
                }
            }
            std::cout << "{\"device\":\"" << name << "\",\"operation\":\"resident_fp32_qk\",\"trial\":" << trial
                      << ",\"heads\":" << heads << ",\"rows_per_head\":" << rows << ",\"keys\":" << keys
                      << ",\"dim\":" << dim << ",\"seconds\":" << seconds
                      << ",\"sampled_max_abs_score_error\":" << max_error << "}" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
