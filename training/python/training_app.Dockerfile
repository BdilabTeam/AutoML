ARG PYTHON_VERSION=3.8.9
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bullseye
ARG VENV_PATH=/prod_venv

FROM ${BASE_IMAGE} as builder

# Install Poetry
ARG POETRY_HOME=/opt/poetry
ARG POETRY_VERSION=1.6.1

# Required for building packages for arm64 arch
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev build-essential

RUN python3 -m venv ${POETRY_HOME} && ${POETRY_HOME}/bin/pip install poetry==${POETRY_VERSION}
ENV PATH="$PATH:${POETRY_HOME}/bin"

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY training_app/pyproject.toml training_app/poetry.lock training_app/
RUN cd training_app && poetry install --no-root --no-interaction --no-cache
COPY training_app training_app
RUN cd training_app && poetry install --no-interaction --no-cache


FROM ${BASE_IMAGE} as prod

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN useradd training_app -m -u 1000 -d /home/training_app

COPY --from=builder --chown=training_app:training_app ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=builder training_app training_app

USER 1000
# ENTRYPOINT ["python", "/training_app/training_app/training_controller.py"]