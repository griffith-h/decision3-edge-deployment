// Batch the existing RKNN kernels across contiguous token rows in one foreign call.
#include <algorithm>
#include <cstdint>

extern "C" {
int nm_prepare(void *, int, const uint16_t *);
int nm_copy(void *, void *, int);
int nm_raw(void *, int);
int nm_fuse(void *, void *, void *, int, const uint16_t *);
int nm_output(void *, int, uint16_t *);

int d3_tiles(void *gate, void *up, void *down, const uint16_t *input, uint16_t *output,
             const uint16_t *silu, int rows, int in_width, int out_width) {
    if (!gate || !input || !output || rows < 1 || in_width < 1 || out_width < 1 ||
        bool(up) != bool(down) || (up && !silu))
        return -1;
    for (int pos = 0; pos < rows; pos += 256) {
        int n = std::min(rows - pos, 256);
        int rc = nm_prepare(gate, n, input + size_t(pos) * in_width);
        if (rc) return rc;
        if (up) {
            rc = nm_copy(up, gate, n);
            if (rc) return rc;
            int gate_rc = 0, up_rc = 0;
            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                gate_rc = nm_raw(gate, n);
                #pragma omp section
                up_rc = nm_raw(up, n);
            }
            if (gate_rc || up_rc) return gate_rc ? gate_rc : up_rc;
            rc = nm_fuse(down, gate, up, n, silu);
            if (rc) return rc;
        }
        rc = nm_output(up ? down : gate, n, output + size_t(pos) * out_width);
        if (rc) return rc;
    }
    return 0;
}
}
