// Time only QK^T with resident inputs. If it already loses to full attention,
// this FP32 CLBlast path cannot improve the complete segment on this workload.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "gpu_host.cpp"

// Compact both heads so CLBlast sees only the current tile, including at each ragged edge.
static void copy_head_slices(D3GPU &gpu, int source, int dest, size_t heads, size_t stride,
                             size_t first, size_t count, size_t dim) {
    const size_t row_bytes = dim * sizeof(float);
    for (size_t h = 0; h < heads; ++h) {
        check(clEnqueueCopyBuffer(gpu.queue, gpu.bufs[source], gpu.bufs[dest],
                                  (h * stride + first) * row_bytes, h * count * row_bytes,
                                  count * row_bytes, 0, nullptr, nullptr), "copy resident input tile");
    }
}

struct Tile {
    size_t row, key, rows, keys;
};

static double check_scores(D3GPU &gpu, const std::vector<float> &q, const std::vector<float> &k,
                           size_t heads, size_t rows, size_t keys, size_t dim, float scale, Tile tile) {
    double max_error = 0;
    for (size_t h = 0; h < heads; ++h) {
        for (size_t i = 0; i < 4; ++i) {
            size_t r = (tile.rows - 1) * i / 3, c = (tile.keys - 1) * (3 - i) / 3;
            const float *a = q.data() + (h * rows + tile.row + r) * dim;
            const float *b = k.data() + (h * keys + tile.key + c) * dim;
            double expected = 0;
            for (size_t d = 0; d < dim; ++d) expected += double(a[d]) * b[d];
            expected *= scale;
            float score;
            const size_t offset = ((h * tile.rows + r) * tile.keys + c) * sizeof(float);
            check(clEnqueueReadBuffer(gpu.queue, gpu.bufs[2], CL_TRUE, offset, sizeof(score), &score,
                                      0, nullptr, nullptr), "verify GPU score");
            const double delta = std::abs(score - expected);
            if (!std::isfinite(delta) || delta > 1e-4 * (1 + std::abs(expected)))
                throw std::runtime_error("GPU score differs from the captured Q/K dot product");
            max_error = std::max(max_error, delta);
        }
    }
    return max_error;
}

int main(int argc, char **argv) {
    try {
        const bool tiled = argc == 4 && std::string(argv[3]) == "--tiled";
        if (argc != 3 && !tiled) throw std::runtime_error("usage: probe_mali gpu_kernels.cl qk.bin [--tiled]");
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
        std::cerr << "Opening the OpenCL GPU" << std::endl;
        std::unique_ptr<D3GPU> gpu(static_cast<D3GPU *>(d3_gpu_create(argv[1])));
        if (!gpu) throw std::runtime_error(d3_gpu_error());
        char name[256] = {};
        check(clGetDeviceInfo(gpu->device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "device name");
        gpu->write(0, q.data(), q.size() * sizeof(float));
        gpu->write(1, k.data(), k.size() * sizeof(float));
        const size_t row_tile = tiled ? std::min<size_t>(128, rows) : rows;
        const size_t key_tile = tiled ? std::min<size_t>(1024, keys) : keys;
        const size_t output_bytes = heads * row_tile * key_tile * sizeof(float);
        gpu->buffer(2, output_bytes);
        if (tiled) {
            gpu->buffer(3, heads * row_tile * dim * sizeof(float));
            gpu->buffer(4, heads * key_tile * dim * sizeof(float));
        }
        check(clFinish(gpu->queue), "initial upload");
        std::cerr << "Inputs resident on " << name << ", maximum output tile " << output_bytes << " bytes" << std::endl;
        std::cout << std::setprecision(9);
        using Clock = std::chrono::steady_clock;
        for (int trial = 0; trial < 3; ++trial) {
            const auto wall_start = Clock::now();
            size_t calls = 0;
            double seconds = 0, max_error = 0;
            for (size_t row = 0; row < rows; row += row_tile) {
                const size_t m = std::min(row_tile, rows - row);
                if (tiled) copy_head_slices(*gpu, 0, 3, heads, rows, row, m, dim);
                for (size_t key = 0; key < keys; key += key_tile) {
                    const size_t n = std::min(key_tile, keys - key);
                    if (tiled) {
                        copy_head_slices(*gpu, 1, 4, heads, keys, key, n, dim);
                        check(clFinish(gpu->queue), "wait for resident copies");
                    }
                    const auto start = Clock::now();
                    auto rc = CLBlastSgemmStridedBatched(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes,
                        m, n, dim, scale, gpu->bufs[tiled ? 3 : 0], 0, dim, m * dim,
                        gpu->bufs[tiled ? 4 : 1], 0, dim, n * dim, 0.f, gpu->bufs[2], 0, n, m * n,
                        heads, &gpu->queue, nullptr);
                    check(static_cast<cl_int>(rc), "CLBlast QK");
                    check(clFinish(gpu->queue), "wait for GPU");
                    seconds += std::chrono::duration<double>(Clock::now() - start).count();
                    max_error = std::max(max_error, check_scores(*gpu, q, k, heads, rows, keys, dim, scale, {row, key, m, n}));
                    if (++calls == 1 && trial == 0) std::cerr << "First output tile verified" << std::endl;
                }
            }
            const double wall = std::chrono::duration<double>(Clock::now() - wall_start).count();
            std::cout << "{\"device\":\"" << name << "\",\"operation\":\"resident_fp32_qk\",\"trial\":" << trial
                      << ",\"heads\":" << heads << ",\"rows_per_head\":" << rows << ",\"keys\":" << keys
                      << ",\"dim\":" << dim << ",\"seconds\":" << seconds
                      << ",\"wall_with_copies_and_checks_s\":" << wall << ",\"dispatches\":" << calls
                      << ",\"row_tile\":" << row_tile << ",\"key_tile\":" << key_tile
                      << ",\"output_buffer_bytes\":" << output_bytes << ",\"scores_checked\":" << calls * heads * 4
                      << ",\"sampled_max_abs_score_error\":" << max_error << "}" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
