# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ARG KEDI_INSTALL_MODE=dev
ARG KEDI_REPOSITORY=https://github.com/kedi-lang/kedi.git
ARG KEDI_REVISION=stable

RUN --mount=type=secret,id=KEDI_GITHUB_TOKEN,required=false \
    set -eu; \
    uv venv /opt/venv; \
    if [ "$KEDI_INSTALL_MODE" = "prod" ]; then \
        uv pip install --python /opt/venv/bin/python --compile-bytecode "kedi[playground]"; \
    elif [ "$KEDI_INSTALL_MODE" = "dev" ]; then \
        if [ ! -s /run/secrets/KEDI_GITHUB_TOKEN ]; then \
            echo "KEDI_GITHUB_TOKEN is required for a dev build" >&2; \
            exit 2; \
        fi; \
        token="$(cat /run/secrets/KEDI_GITHUB_TOKEN)"; \
        git config --global \
            url."https://x-access-token:${token}@github.com/".insteadOf \
            "https://github.com/"; \
        uv pip install \
            --python /opt/venv/bin/python \
            --compile-bytecode \
            "kedi[playground] @ git+${KEDI_REPOSITORY}@${KEDI_REVISION}"; \
        rm -f /root/.gitconfig; \
    else \
        echo "KEDI_INSTALL_MODE must be prod or dev" >&2; \
        exit 2; \
    fi; \
    /opt/venv/bin/python -c \
        "import pydantic_monty; from kedi.agent_adapter import WebGPUAdapter; from kedi.executors import NsJailExecutor, PlaygroundExecutor, PyodideExecutor"


FROM python:3.12-slim-bookworm AS nsjail-builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        bison \
        build-essential \
        ca-certificates \
        flex \
        git \
        libnl-route-3-dev \
        libprotobuf-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

ARG NSJAIL_REPOSITORY=https://github.com/google/nsjail.git
ARG NSJAIL_REVISION=master

RUN git clone --depth 1 --branch "$NSJAIL_REVISION" "$NSJAIL_REPOSITORY" /tmp/nsjail \
    && make -C /tmp/nsjail \
    && /tmp/nsjail/nsjail --help >/dev/null


FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libnl-3-200 \
        libnl-route-3-200 \
        libprotobuf32 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nsjail-builder /tmp/nsjail/nsjail /usr/local/bin/nsjail

RUN nsjail --help >/dev/null

RUN useradd --create-home --uid 1000 user

ENV HOST=0.0.0.0 \
    KEDI_NSJAIL_CHROOT=/opt/kedi-jail-root \
    KEDI_NSJAIL_STATIC_CHROOT=1 \
    PATH=/opt/venv/bin:$PATH \
    PORT=7860 \
    PYTHONPATH=/home/user/app/src \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=user:user src ./src

RUN set -eux; \
    chmod 4755 /usr/local/bin/nsjail; \
    mkdir -p \
        "$KEDI_NSJAIL_CHROOT/etc" \
        "$KEDI_NSJAIL_CHROOT/home/user/app" \
        "$KEDI_NSJAIL_CHROOT/lib" \
        "$KEDI_NSJAIL_CHROOT/opt" \
        "$KEDI_NSJAIL_CHROOT/tmp" \
        "$KEDI_NSJAIL_CHROOT/usr"; \
    cp -a /etc/ssl "$KEDI_NSJAIL_CHROOT/etc/ssl"; \
    cp -a /home/user/app/src "$KEDI_NSJAIL_CHROOT/home/user/app/src"; \
    cp -a /lib/. "$KEDI_NSJAIL_CHROOT/lib/"; \
    if [ -e /lib64 ]; then cp -aL /lib64 "$KEDI_NSJAIL_CHROOT/lib64"; fi; \
    cp -a /opt/venv "$KEDI_NSJAIL_CHROOT/opt/venv"; \
    cp -a /usr/lib "$KEDI_NSJAIL_CHROOT/usr/lib"; \
    cp -a /usr/local "$KEDI_NSJAIL_CHROOT/usr/local"; \
    chmod 1777 "$KEDI_NSJAIL_CHROOT/tmp"

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3)"

CMD ["python", "-m", "playground.server"]
