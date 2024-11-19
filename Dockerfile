# WIP to load the trainer into the docker file. 
# TODO: Needs arguments to pass to train
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

FROM python:3.11-slim-bookworm

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv 
COPY --from=builder --chown=app:app /app/train.py /app/train.py 
COPY --from=builder --chown=app:app /app/vae_train /app/vae_train

COPY entrypoint.sh /app/

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT ["/entrypoint.sh"]

# CMD ["/app/.venv/bin/accelerate", "launch", "/app/train.py"]
