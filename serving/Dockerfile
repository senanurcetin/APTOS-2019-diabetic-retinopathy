# Image for the Hugging Face Space. Built from the directory that
# serving/stage_space.py assembles, not from the repository root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # The package resolves its root from its own location by default, which
    # after a non-editable install is site-packages. Point it at /app instead.
    APTOS_ROOT=/app \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    HF_HOME=/tmp/hf

WORKDIR /app

# CPU wheels only. The PyPI default pulls a CUDA build several gigabytes larger,
# and a free Space has no GPU to use it. torch, torchvision and timm are pinned
# to the versions the checkpoints were trained with, since they define the
# architecture the weights are loaded into.
RUN pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        timm==1.0.29 \
        "numpy>=2.0,<3" \
        "opencv-python-headless>=4.9,<6" \
        "pandas>=2.2,<3" \
        "scipy>=1.11" \
        "scikit-learn>=1.4" \
        "pyyaml>=6" \
        "pillow>=10" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "python-multipart>=0.0.9"

COPY pyproject.toml README.md ./
COPY src ./src
# --no-deps: the dependencies above are the serving subset. The package's own
# list also names matplotlib and friends, which nothing here imports.
RUN pip install --no-cache-dir --no-deps .

COPY configs ./configs
COPY serving ./serving
COPY models ./models

# Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
