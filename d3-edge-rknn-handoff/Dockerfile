FROM ghcr.io/vllm-project/semantic-router/vllm-sr@sha256:efc356587252c28b8d972d5d27b381df0626ccdc53597f3fccff7ae454041178

# PyTorch 2.10 ARM64 wheels crash on Cortex-A53/A72 at import (upstream #174344).
# Install the fixed official CPU wheel and preserve the rest of the router image.
RUN python -m pip install --no-cache-dir --no-deps --force-reinstall \
    'https://download-r2.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=ce2ddb880b0813fcc91a737f08fdd973a8115a74c64ccb34e9c09a7964b4d448' \
    && python -c 'import torch; assert torch.__version__ == "2.12.0+cpu"; x = torch.ones(8, 8, dtype=torch.bfloat16); assert (x @ x).shape == (8, 8); print("ARMv8 PyTorch and BF16 matmul OK")'


# Serve through the HTTP implementation shipped with the pinned model package.
# CPU inference uses PyTorch reference kernels; do not install GPU-only FLA.
RUN python -m pip install --no-cache-dir \
    transformers==5.17.0 accelerate==1.10.1 \
    'https://download-r2.pytorch.org/whl/cpu/torchvision-0.27.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=1b06f42d48b62098114923d8a3fe9fa864182715db06584a515155db0aa8eb30' \
    && python -c 'import torch, torchvision, transformers; assert torch.__version__ == "2.12.0+cpu"; assert transformers.__version__ == "5.17.0"; print(torch.__version__, torchvision.__version__, transformers.__version__)'

ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "-u"]

RUN apt-get update && apt-get install -y --no-install-recommends g++ ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock && python -m pip check
WORKDIR /work
COPY tools/fetch_sdk.py sdk-manifest.json /tmp/sdk-fetch/
RUN python /tmp/sdk-fetch/fetch_sdk.py /work/sdk /tmp/sdk-fetch/sdk-manifest.json
COPY app/ /work/
RUN g++ -O2 -std=c++17 -shared -fPIC -I sdk matmul_bridge.cpp -L sdk -lrknnrt -Wl,-rpath,'$ORIGIN/sdk' -o libmatmul_bridge.so
CMD ["/work/npu_server.py", "--model", "/models/d3-edge", "--host", "0.0.0.0", "--port", "8900"]
