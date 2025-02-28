#!/bin/sh
set -e

# Activate the virtual environment if needed.
# If the PATH is already set properly, this might be unnecessary,
# but it ensures that any activation-specific shell modifications are applied.
. "$VIRTUAL_ENV/bin/activate"

# Install or update dependencies at startup.
if [ -f /home/app_user/requirements.txt ]; then
  echo "Installing/updating Python dependencies..."
  uv pip install -U -r /home/app_user/requirements.txt
fi

# Execute the main command passed via CMD.
exec "$@"
