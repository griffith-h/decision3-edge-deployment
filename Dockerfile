FROM ghcr.io/vllm-project/semantic-router/vllm-sr@sha256:efc356587252c28b8d972d5d27b381df0626ccdc53597f3fccff7ae454041178

# PyTorch 2.10 ARM64 wheels crash on Cortex-A53/A72 at import (upstream #174344).
# Install the fixed official CPU wheel and preserve the rest of the router image.
RUN python -m pip install --no-cache-dir --no-deps --force-reinstall \
    'https://download-r2.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=ce2ddb880b0813fcc91a737f08fdd973a8115a74c64ccb34e9c09a7964b4d448' \
    && python -c 'import torch; assert torch.__version__ == "2.12.0+cpu"; x = torch.ones(8, 8, dtype=torch.bfloat16); assert (x @ x).shape == (8, 8); print("ARMv8 PyTorch and BF16 matmul OK")'
