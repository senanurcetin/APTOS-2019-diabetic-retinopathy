# The inference service, as deployed on Render's free tier.
#
# Built from the repository root (see render.yaml). It serves the ONNX export
# through ONNX Runtime and never imports torch: the free tier has 512 MB of RAM,
# and measured with the five fold models loaded the service peaks at ~204 MB.
# The same code with torch did not fit.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # After a non-editable install the package would resolve its root to
    # site-packages; point it at the application directory instead.
    APTOS_ROOT=/app \
    APTOS_BACKEND=onnx \
    HOME=/tmp

WORKDIR /app

# The serving subset only. No torch, no torchvision, no scikit-learn at import
# time, no mlflow - each was checked to be absent from the running process.
RUN pip install --no-cache-dir \
        "onnxruntime>=1.18,<2" \
        "numpy>=2.0,<3" \
        "opencv-python-headless>=4.9,<6" \
        "pillow>=10" \
        "pyyaml>=6" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "python-multipart>=0.0.9"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY configs ./configs

# The weights live in a Hugging Face model repository, not in git: 80 MB of
# binaries do not belong in a source repository. The revision is pinned to a
# commit, so every build serves exactly the model whose parity was checked.
ARG WEIGHTS_REPO=senanurcetin/aptos-retinopathy-grader
ARG WEIGHTS_REVISION=882dfc66076cd143d91002394fff68c1778f8e1b
ARG SWEEP=baseline-6e66147526
COPY serving/fetch_weights.py ./serving/fetch_weights.py
# Refuses an export whose recorded parity check failed.
RUN python serving/fetch_weights.py \
        --repo "${WEIGHTS_REPO}" --revision "${WEIGHTS_REVISION}" \
        --sweep "${SWEEP}" --out models/onnx

# After the weights on purpose: the page and the app change far more often than
# the model, and a change here should not re-download 80 MB.
COPY serving/app.py ./serving/app.py
COPY serving/static ./serving/static

RUN useradd -m -u 1000 app && chown -R app /app
USER app

# Render supplies PORT; 10000 is its documented default.
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
