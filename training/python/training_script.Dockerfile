ARG PYTHON_VERSION=3.8.9
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim
ARG VENV_PATH=/prod_venv

FROM ${BASE_IMAGE} as builder

# Install Poetry
ARG POETRY_HOME=/opt/poetry
ARG POETRY_VERSION=1.6.1

# Required for building packages for arm64 arch
# RUN apt-get update && apt-get install -y --no-install-recommends python3-dev build-essential

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set global.timeout 1800
RUN pip config set global.retries 10
RUN python3 -m venv ${POETRY_HOME} && ${POETRY_HOME}/bin/python3 -m pip install --upgrade pip && ${POETRY_HOME}/bin/pip install poetry==${POETRY_VERSION}
ENV PATH="$PATH:${POETRY_HOME}/bin"
ENV POETRY_REQUESTS_TIMEOUT=3600
ENV POETRY_RETRIES=10

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY training_script/pyproject.toml training_script/poetry.lock training_script/
RUN cd training_script && poetry install --no-root --no-interaction --no-cache
COPY training_script training_script
RUN cd training_script && poetry install --no-interaction --no-cache


FROM ${BASE_IMAGE} as prod

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN useradd training_script -m -u 1000 -d /home/training_script

COPY --from=builder --chown=training_script:training_script ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=builder training_script training_script

USER 1000