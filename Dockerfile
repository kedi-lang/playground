FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ARG KEDI_REPOSITORY=https://github.com/kedi-lang/kedi.git
ARG KEDI_REVISION=kedi_playground
ARG KEDI_INSTALL_MODE=dev

RUN --mount=type=secret,id=GITHUB_TOKEN \
    if [ "$KEDI_INSTALL_MODE" = "prod" ]; then \
        kedi_package="kedi[playground]"; \
    elif [ "$KEDI_INSTALL_MODE" = "dev" ]; then \
        if [ ! -s /run/secrets/GITHUB_TOKEN ]; then \
            echo "GITHUB_TOKEN is required for a dev build" >&2; \
            exit 1; \
        fi; \
        token="$(cat /run/secrets/GITHUB_TOKEN)"; \
        git config --global \
            url."https://x-access-token:${token}@github.com/".insteadOf \
            "https://github.com/"; \
        kedi_package="kedi[playground] @ git+${KEDI_REPOSITORY}@${KEDI_REVISION}"; \
    else \
        echo "KEDI_INSTALL_MODE must be prod or dev" >&2; \
        exit 1; \
    fi \
    && uv venv /opt/venv \
    && uv pip install \
        --upgrade \
        --python /opt/venv/bin/python \
        "$kedi_package" \
    && rm -f /root/.gitconfig \
    && /opt/venv/bin/python -c \
        "from kedi.agent_adapter import WebGPUAdapter; from kedi.executors import PyodideExecutor"


FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 1000 user

ENV HOST=0.0.0.0 \
    PATH=/opt/venv/bin:$PATH \
    PORT=7860 \
    PYTHONPATH=/home/user/app/src \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=user:user src/playground ./src/playground

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3)"

CMD ["python", "-m", "playground.server"]
