#include <cstddef>
#include "rkllm.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Capture {
    std::vector<float> last;
    int width = 0;
    int rows = 0;
    bool failed = false;
    RKLLMPerfStat perf{};
};

int callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* capture = static_cast<Capture*>(userdata);
    if (state == RKLLM_RUN_ERROR) capture->failed = true;
    if (result) {
        if (result->perf.prefill_time_ms > 0 || result->perf.generate_time_ms > 0)
            capture->perf = result->perf;
        auto& hidden = result->last_hidden_layer;
        if (hidden.hidden_states && hidden.embd_size > 0 && hidden.num_tokens > 0) {
            capture->width = hidden.embd_size;
            capture->rows = hidden.num_tokens;
            const float* last = hidden.hidden_states +
                                static_cast<size_t>(hidden.num_tokens - 1) * hidden.embd_size;
            capture->last.assign(last, last + hidden.embd_size);
            return 2; // The documented API allows releasing this copied output buffer.
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4 || (argc - 2) % 2 != 0) {
        std::cerr << "Usage: npu_hidden MODEL TOKEN_FILE OUTPUT_F32 [TOKEN_FILE OUTPUT_F32 ...]\n";
        return 2;
    }
    RKLLMParam params = rkllm_createDefaultParam();
    params.model_path = argv[1];
    params.max_context_len = 512;
    params.max_new_tokens = 1;
    params.top_k = 1;
    params.is_async = false;
    params.extend_param.enabled_cpus_num = 4;
    params.extend_param.enabled_cpus_mask = 0x0f;
    Capture capture;
    RKLLMCallback callbacks{};
    callbacks.result_callback = callback;
    callbacks.result_userdata = &capture;
    LLMHandle handle = nullptr;
    int status = rkllm_init(&handle, &params, &callbacks);
    if (status != 0) {
        std::cerr << "rkllm_init failed: " << status << "\n";
        return 3;
    }
    rkllm_set_chat_template(handle, "", "", "");
    for (int file = 2; file < argc; file += 2) {
        std::ifstream input_file(argv[file]);
        std::vector<int32_t> ids;
        int32_t token;
        while (input_file >> token) ids.push_back(token);
        if (ids.empty() || ids.size() >= 512) {
            std::cerr << "Invalid input token count: " << ids.size() << "\n";
            rkllm_destroy(handle);
            return 4;
        }
        capture = Capture{};
        rkllm_clear_kv_cache(handle, 0, nullptr, nullptr);
        RKLLMInput input{};
        input.role = "user";
        input.input_type = RKLLM_INPUT_TOKEN;
        input.token_input.input_ids = ids.data();
        input.token_input.n_tokens = ids.size();
        RKLLMInferParam infer{};
        infer.mode = RKLLM_INFER_GET_LAST_HIDDEN_LAYER;
        infer.keep_history = 0;
        infer.max_new_tokens = 1;
        auto start = std::chrono::steady_clock::now();
        status = rkllm_run(handle, &input, &infer, &capture);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (status != 0 || capture.failed || capture.last.empty()) {
            std::cerr << "rkllm_run failed: " << status << ", hidden_size=" << capture.last.size() << "\n";
            rkllm_destroy(handle);
            return 5;
        }
        std::ofstream output(argv[file + 1], std::ios::binary);
        output.write(reinterpret_cast<const char*>(capture.last.data()),
                     capture.last.size() * sizeof(float));
        if (!output) {
            std::cerr << "Failed to write hidden-state output\n";
            rkllm_destroy(handle);
            return 6;
        }
        std::cout << "RESULT {\"input\":\"" << argv[file] << "\",\"input_tokens\":" << ids.size()
                  << ",\"hidden_size\":" << capture.width << ",\"hidden_rows\":" << capture.rows
                  << ",\"seconds\":" << seconds << ",\"prefill_ms\":" << capture.perf.prefill_time_ms
                  << ",\"memory_mb\":" << capture.perf.memory_usage_mb << "}" << std::endl;
    }
    rkllm_destroy(handle);
    return 0;
}
