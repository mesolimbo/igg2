# The MCP server runs inside this image, which extends igg2:local so it can
# invoke the igg2 CLI directly -- no Docker-in-Docker, nothing on the host.
# Build igg2:local first (the Makefile's `igg2-image` target does this).
FROM igg2:local

USER root

# Install the MCP SDK on top of igg2's existing environment. Kept in a
# separate directory so igg2's own Pipfile in /app is left untouched.
WORKDIR /opt/igg2-mcp
COPY Pipfile Pipfile.lock ./
RUN pipenv sync --system --dev

# Add the MCP server and tests alongside the igg2 source already in /app.
WORKDIR /app
COPY src/server.py ./src/server.py
COPY test/ ./test/

USER igg2:igg2

# Speak the Model Context Protocol over stdio.
ENTRYPOINT ["python", "-m", "src.server"]
CMD []
