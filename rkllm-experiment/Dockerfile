FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ENV DEBIAN_FRONTEND=noninteractive BUILD_CUDA_EXT=0 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 libglib2.0-0 libgl1 libgfortran5 git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
    'setuptools<80' wheel numpy==1.26.4 \
    && python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu torchvision==0.21.0+cpu \
    && python -c 'import platform, torch; print(platform.machine(), torch.__version__); print(torch.ones(2,2) @ torch.ones(2,2))'
COPY toolkit/rkllm_toolkit-1.3.1-cp312-cp312-linux_x86_64.whl /tmp/toolkit.whl
RUN mv /tmp/toolkit.whl /tmp/rkllm_toolkit-1.3.1-cp312-cp312-linux_x86_64.whl \
    && python -m pip install --no-cache-dir --no-build-isolation \
       /tmp/rkllm_toolkit-1.3.1-cp312-cp312-linux_x86_64.whl \
    && python -m pip check \
    && python -c 'from rkllm.api import RKLLM; print("RKLLM import OK")' \
    && rm /tmp/rkllm_toolkit-1.3.1-cp312-cp312-linux_x86_64.whl
WORKDIR /work
ENTRYPOINT ["python", "-u"]
