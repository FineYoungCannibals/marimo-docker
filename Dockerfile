FROM python:3.13.1-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV VIRTUAL_ENV=/home/app_user/venv

RUN apt update && \
    apt upgrade -y && \
    apt install -y curl \
    iputils-ping && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m app_user
RUN uv venv $VIRTUAL_ENV
RUN chown -R app_user:app_user $VIRTUAL_ENV

USER app_user
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Ensure directories exist
RUN mkdir -p /home/app_user/notebooks/utils
RUN touch /home/app_user/notebooks/utils/__init__.py

# Set PYTHONPATH so Python can find modules in /home/app_user/utils
ENV PYTHONPATH="/home/app_user/notebooks/utils:${PYTHONPATH}"

# Copy requirements (it will be used by the entrypoint).
COPY --chown=app_user:app_user requirements.txt /home/app_user/requirements.txt

# Install standard packages into our image
RUN . $VIRTUAL_ENV/bin/activate && uv pip install -U -r /home/app_user/requirements.txt

# Copy the entrypoint script.
COPY --chown=app_user:app_user entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /home/app_user/notebooks
EXPOSE 2718

# Use the entrypoint script to update dependencies upon container rebuilds
ENTRYPOINT ["/entrypoint.sh"]
