ARG PYTHON_VERSION=3.9
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim
ARG VENV_PATH=/prod_venv

FROM ${BASE_IMAGE} as builder

# Install Poetry
ARG POETRY_HOME=/opt/poetry
ARG POETRY_VERSION=1.6.1

# Required for building packages for arm64 arch
# RUN apt-get update && apt-get install -y --no-install-recommends python3-dev build-essential

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip config set global.timeout 1800 && pip config set global.retries 10
RUN python3 -m venv ${POETRY_HOME} && ${POETRY_HOME}/bin/python3 -m pip install --upgrade pip && ${POETRY_HOME}/bin/pip install poetry==${POETRY_VERSION}
ENV PATH="$PATH:${POETRY_HOME}/bin"
ENV POETRY_REQUESTS_TIMEOUT=3600
ENV POETRY_RETRIES=10

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY tserve/pyproject.toml tserve/poetry.lock tserve/
RUN cd tserve && poetry install --no-root --no-interaction --no-cache
COPY tserve tserve
RUN cd tserve && poetry install --no-interaction --no-cache

COPY yolo_server/pyproject.toml yolo_server/poetry.lock yolo_server/
RUN cd yolo_server && poetry install --no-root --no-interaction --no-cache
COPY yolo_server yolo_server
RUN cd yolo_server && poetry install --no-interaction --no-cache


FROM ${BASE_IMAGE} as prod

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN useradd tserve -m -u 1000 -d /home/tserve

COPY --from=builder --chown=tserve:tserve ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=builder tserve tserve
COPY --from=builder yolo_server yolo_server

USER 1000
# ENTRYPOINT ["python", "-m", "sklearnserver"]