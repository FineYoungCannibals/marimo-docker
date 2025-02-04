FROM python:3.13.1-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV VIRTUAL_ENV=/home/app_user/venv

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m app_user
RUN uv venv $VIRTUAL_ENV
RUN chown -R app_user:app_user $VIRTUAL_ENV

USER app_user
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --chown=app_user:app_user requirements.txt .
RUN . $VIRTUAL_ENV/bin/activate && uv pip install -r requirements.txt

WORKDIR /home/app_user/notebooks
EXPOSE 2718