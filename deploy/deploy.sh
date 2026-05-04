#!/usr/bin/env bash
# deploy/deploy.sh — Deploy neo4j-agent-memory-mcp to hume-prod
#
# Prerequisites:
#   - SSH key at ~/.ssh/hume-demo.pem (or set PEM_FILE env var)
#   - deploy/.env file with production secrets (copy from .env.prod.example)
#
# Usage:
#   ./deploy/deploy.sh              # Full deploy (clone + neo4j + mcp server)
#   ./deploy/deploy.sh update       # Update code only (git pull + restart)

set -euo pipefail

HOST="centos@54.226.102.25"
PEM="${PEM_FILE:-$HOME/.ssh/hume-demo.pem}"
REMOTE_DIR="/opt/neo4j-agent-memory-mcp"
REPO_URL="$(git remote get-url origin 2>/dev/null || echo 'REPO_URL_NOT_SET')"
BRANCH="$(git branch --show-current)"

if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    echo "Set PEM_FILE env var or place key at ~/.ssh/hume-demo.pem"
    exit 1
fi

SSH="ssh -i $PEM -o StrictHostKeyChecking=no"
SCP="scp -i $PEM -o StrictHostKeyChecking=no"

echo "=== Deploying to hume-prod (54.226.102.25) ==="
echo "Branch: $BRANCH"
echo ""

if [ "${1:-}" = "update" ]; then
    echo "--- Update mode: pulling latest code and restarting ---"
    $SSH $HOST <<SCRIPT
cd $REMOTE_DIR
git pull origin $BRANCH
export PATH="\$HOME/.local/bin:\$PATH"
uv sync --frozen --no-dev
uv run baml-cli generate
# Overlay files must overwrite base package (Python 3.14 resolves base first)
SITE_PKG=\$(python3 -c "import neo4j_agent_memory; import os; print(os.path.dirname(neo4j_agent_memory.__file__))" 2>/dev/null || echo "")
if [ -n "\$SITE_PKG" ] && [ -d src/neo4j_agent_memory ]; then
    cd src/neo4j_agent_memory
    find . -name '*.py' | while read f; do
        if [ -f "\$SITE_PKG/\$f" ]; then
            cp "\$f" "\$SITE_PKG/\$f"
        fi
    done
    cd \$REMOTE_DIR
    echo "Overlay files synced to \$SITE_PKG"
fi
sudo systemctl restart neo4j-memory-mcp
echo "=== MCP server restarted ==="
sudo systemctl status neo4j-memory-mcp --no-pager
SCRIPT
    exit 0
fi

echo "--- Step 1: Install uv on remote ---"
$SSH $HOST <<'SCRIPT'
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'source ~/.local/bin/env' >> ~/.bashrc
fi
echo "uv version: $(~/.local/bin/uv --version 2>/dev/null || echo 'installing...')"
SCRIPT

echo "--- Step 2: Clone repository ---"
$SSH $HOST <<SCRIPT
if [ -d "$REMOTE_DIR" ]; then
    echo "Directory exists, pulling latest..."
    cd $REMOTE_DIR
    git pull origin $BRANCH
else
    sudo mkdir -p $REMOTE_DIR
    sudo chown centos:centos $REMOTE_DIR
    git clone -b $BRANCH $REPO_URL $REMOTE_DIR
fi
SCRIPT

echo "--- Step 3: Copy production .env ---"
if [ -f "deploy/.env" ]; then
    $SCP deploy/.env $HOST:$REMOTE_DIR/deploy/.env
    echo "Copied deploy/.env to remote"
else
    echo "WARNING: deploy/.env not found. Copy deploy/.env.prod.example to deploy/.env and fill in secrets."
    echo "Then run: scp -i $PEM deploy/.env $HOST:$REMOTE_DIR/deploy/.env"
fi

echo "--- Step 4: Install dependencies and generate BAML ---"
$SSH $HOST <<SCRIPT
cd $REMOTE_DIR
export PATH="\$HOME/.local/bin:\$PATH"
uv sync --frozen --no-dev
uv run baml-cli generate
# Overlay files must overwrite base package (Python 3.14 resolves base first)
SITE_PKG=\$(python3 -c "import neo4j_agent_memory; import os; print(os.path.dirname(neo4j_agent_memory.__file__))" 2>/dev/null || echo "")
if [ -n "\$SITE_PKG" ] && [ -d src/neo4j_agent_memory ]; then
    cd src/neo4j_agent_memory
    find . -name '*.py' | while read f; do
        if [ -f "\$SITE_PKG/\$f" ]; then
            cp "\$f" "\$SITE_PKG/\$f"
        fi
    done
    cd \$REMOTE_DIR
    echo "Overlay files synced to \$SITE_PKG"
fi
echo "Dependencies installed, BAML generated"
SCRIPT

echo "--- Step 5: Start Neo4j ---"
$SSH $HOST <<SCRIPT
cd $REMOTE_DIR/deploy
if [ -f .env ]; then
    set -a; source .env; set +a
fi
docker compose -f docker-compose.prod.yml up -d
echo "Waiting for Neo4j healthcheck..."
sleep 10
docker compose -f docker-compose.prod.yml ps
SCRIPT

echo "--- Step 6: Install and start systemd service ---"
$SSH $HOST <<SCRIPT
sudo cp $REMOTE_DIR/deploy/neo4j-memory-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable neo4j-memory-mcp
sudo systemctl restart neo4j-memory-mcp
sleep 3
sudo systemctl status neo4j-memory-mcp --no-pager
SCRIPT

echo ""
echo "=== Deployment complete ==="
echo "MCP server: http://54.226.102.25:8080"
echo "Neo4j browser: http://54.226.102.25:7474"
echo "Health check: curl http://54.226.102.25:8080/health"
echo "Logs: ssh -i $PEM $HOST 'journalctl -u neo4j-memory-mcp -f'"
