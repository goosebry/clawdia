#!/usr/bin/env bash
# =============================================================================
# Ella AI Agent — Mac Mini Deployment Script
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Assumptions:
#   - macOS (Apple Silicon M-series)
#   - Docker Desktop is installed and running
#   - Internet access for downloading models and packages
#
# What this script does:
#   1. Checks system prerequisites (macOS, Apple Silicon, Docker, Python)
#   2. Installs Homebrew if missing
#   3. Installs Python 3.11+ via pyenv if not available
#   4. Creates a Python virtual environment
#   5. Installs all Python dependencies
#   6. Starts Redis + Qdrant via Docker Compose
#   7. Initialises Qdrant collections
#   8. Pre-downloads MLX models (LLM, VL, Whisper)
#   9. Creates .env from .env.example if not present
#  10. Creates ~/Ella/ identity files if not already present
#  11. Pre-downloads MLX models
#  12. Creates launchd plist files for auto-start on login
#  13. Prints a getting-started summary
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[ella]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${BLUE}── $* ──${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
ELLA_PLIST_MAIN="com.ella.main.plist"
ELLA_PLIST_WORKER="com.ella.worker.plist"
LOG_DIR="$HOME/Library/Logs/ella"

# ── Step 1: System Checks ─────────────────────────────────────────────────────
step "Checking system requirements"

# macOS only
if [[ "$(uname)" != "Darwin" ]]; then
    err "This script is for macOS only."
fi

# Apple Silicon check
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    warn "Not running on Apple Silicon ($ARCH). MLX requires arm64. Continuing anyway..."
fi
log "Architecture: $ARCH"

# macOS version
MACOS_VERSION="$(sw_vers -productVersion)"
log "macOS: $MACOS_VERSION"

# Docker
if ! command -v docker &>/dev/null; then
    err "Docker not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
fi
if ! docker info &>/dev/null; then
    err "Docker daemon is not running. Start Docker Desktop and re-run this script."
fi
log "Docker: $(docker --version)"

# ── Step 2: Homebrew ──────────────────────────────────────────────────────────
step "Checking Homebrew and system tools"
if ! command -v brew &>/dev/null; then
    log "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    log "Homebrew: $(brew --version | head -1)"
fi

# ffmpeg is required by mlx-whisper for audio decoding
if ! command -v ffmpeg &>/dev/null; then
    log "Installing ffmpeg..."
    brew install ffmpeg
else
    log "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
fi

# ── Step 3: Python ────────────────────────────────────────────────────────────
step "Checking Python (3.11+ required)"

PYTHON_BIN=""

# Prefer Python 3.12; 3.13 also works (no XTTS-v2 constraint any more)
for PY in python3.12 python3.11 python3.13; do
    if command -v "$PY" &>/dev/null; then
        PY_VER="$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        log "Found $PY ($PY_VER)"
        PYTHON_BIN="$(command -v $PY)"
        break
    fi
done

# Install via pyenv if no suitable Python found
if [[ -z "$PYTHON_BIN" ]]; then
    warn "Python 3.11+ not found. Installing via pyenv..."
    if ! command -v pyenv &>/dev/null; then
        brew install pyenv
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi
    PYTHON_TARGET="3.12.3"
    pyenv install --skip-existing "$PYTHON_TARGET"
    pyenv global "$PYTHON_TARGET"
    PYTHON_BIN="$(pyenv which python3)"
    log "Installed Python $PYTHON_TARGET via pyenv"
fi

log "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# ── Step 4: Virtual Environment ───────────────────────────────────────────────
step "Setting up Python virtual environment"

if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    log "Virtual environment already exists at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
log "Activated venv: $(python --version)"

# Upgrade pip
pip install --quiet --upgrade pip setuptools wheel

# ── Step 5: Python Dependencies ───────────────────────────────────────────────
step "Installing Python dependencies"
log "This may take several minutes on first run..."

log "Installing packages..."
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
log "Dependencies installed."

# ── Step 6: Infrastructure (Docker) ──────────────────────────────────────────
step "Starting Redis + Qdrant via Docker Compose"
cd "$SCRIPT_DIR"
docker compose pull --quiet
docker compose up -d
log "Waiting for services to become healthy..."

# Wait for Redis
REDIS_READY=0
for i in $(seq 1 30); do
    if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
        REDIS_READY=1
        break
    fi
    sleep 1
done
[[ "$REDIS_READY" -eq 1 ]] && log "Redis: ready" || warn "Redis health check timed out — continuing anyway"

# Wait for Qdrant
QDRANT_READY=0
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz &>/dev/null; then
        QDRANT_READY=1
        break
    fi
    sleep 1
done
[[ "$QDRANT_READY" -eq 1 ]] && log "Qdrant: ready" || warn "Qdrant health check timed out — continuing anyway"

# Wait for MySQL
MYSQL_READY=0
for i in $(seq 1 40); do
    if docker compose exec -T mysql mysqladmin ping -h localhost -uella -pella_pass --silent 2>/dev/null; then
        MYSQL_READY=1
        break
    fi
    sleep 2
done
[[ "$MYSQL_READY" -eq 1 ]] && log "MySQL: ready" || warn "MySQL health check timed out — continuing anyway"

# ── Step 7: Qdrant Collections ────────────────────────────────────────────────
step "Initialising Qdrant collections"
cd "$SCRIPT_DIR"
python scripts/init_qdrant.py
log "Qdrant collections ready."

# ── Step 7b: Emotion Engine DB Tables ─────────────────────────────────────────
step "Initialising emotion engine MySQL tables"
if [[ "$MYSQL_READY" -eq 1 ]]; then
    docker compose exec -T mysql mysql -uella -pella_pass ella < scripts/migrate_emotion.sql
    log "Emotion engine tables ready."
else
    warn "MySQL not available — skipping emotion engine migration. Run manually:"
    warn "  docker compose exec -T mysql mysql -uella -pella_pass ella < scripts/migrate_emotion.sql"
fi

# ── Step 8: .env Setup ────────────────────────────────────────────────────────
step "Configuring environment"
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    log "Created .env from .env.example"
    echo ""
    warn "IMPORTANT: Edit .env and set your TELEGRAM_BOT_TOKEN before starting Ella."
    warn "  nano $SCRIPT_DIR/.env"
else
    log ".env already exists — skipping."
fi

# Check if token is set
if grep -q "^TELEGRAM_BOT_TOKEN=your_bot_token_here$" "$SCRIPT_DIR/.env" 2>/dev/null; then
    warn "TELEGRAM_BOT_TOKEN is not configured in .env"
fi

# Create assets directory for speaker WAV
mkdir -p "$SCRIPT_DIR/assets"
if [[ ! -f "$SCRIPT_DIR/assets/speaker.wav" ]]; then
    warn "assets/speaker.wav not found."
    warn "Qwen3-TTS will use its default built-in voice until you provide a reference WAV."
    warn "Place a ≥3 second reference WAV at: $SCRIPT_DIR/assets/speaker.wav"
fi

# ── Step 9: Identity files — ~/Ella/ ─────────────────────────────────────────
step "Setting up Ella's identity files"
ELLA_DIR="$HOME/Ella"
IDENTITY_SRC="$SCRIPT_DIR/identity"
mkdir -p "$ELLA_DIR"

for fname in Identity.md Soul.md User.md Personality.json Personality.md; do
    dest="$ELLA_DIR/$fname"
    src="$IDENTITY_SRC/$fname"
    if [[ -f "$dest" ]]; then
        log "~/Ella/$fname already exists — keeping your version."
    else
        cp "$src" "$dest"
        log "Copied default ~/Ella/$fname"
    fi
done

echo ""
warn "Personalise Ella by editing the files in ~/Ella/:"
warn "  Identity.md      — her name, age, timezone, background, relationship with you"
warn "  Soul.md          — her personality, tone, and values"
warn "  User.md          — who you are (DOB, ethnicity, country, profession)"
warn "  Personality.json — emotion engine trait numbers (resilience, volatility, ECS weights)"
warn "  Personality.md   — narrative description of her emotional character"
warn "Changes take effect immediately (hot-reloaded, no restart needed)."
echo ""

# ── Step 10: Pre-download MLX Models ─────────────────────────────────────────
step "Pre-downloading MLX models (this may take a while)"
log "Models will be cached in ~/.cache/huggingface"

# Read model names from .env (or use defaults)
CHAT_MODEL="$(grep '^MLX_CHAT_MODEL=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'mlx-community/Qwen2.5-7B-Instruct-4bit')"
VL_MODEL="$(grep '^MLX_VL_MODEL=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'mlx-community/Qwen2.5-VL-3B-Instruct-4bit')"
WHISPER_MODEL="$(grep '^MLX_WHISPER_MODEL=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'mlx-community/whisper-small')"
EMBED_MODEL="$(grep '^EMBED_MODEL=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'paraphrase-multilingual-MiniLM-L12-v2')"

# Use CHAT_MODEL default if empty
CHAT_MODEL="${CHAT_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
VL_MODEL="${VL_MODEL:-mlx-community/Qwen2.5-VL-3B-Instruct-4bit}"
WHISPER_MODEL="${WHISPER_MODEL:-mlx-community/whisper-small}"
EMBED_MODEL="${EMBED_MODEL:-paraphrase-multilingual-MiniLM-L12-v2}"

log "Chat LLM:       $CHAT_MODEL"
log "VL model:       $VL_MODEL"
log "Whisper model:  $WHISPER_MODEL"
log "Embed model:    $EMBED_MODEL"

# Download chat LLM
log "Downloading chat LLM..."
python -c "
from mlx_lm import load
print('  Downloading $CHAT_MODEL...')
load('$CHAT_MODEL')
print('  Done.')
" || warn "Chat LLM download failed — will download on first use."

# Download whisper
log "Downloading Whisper STT..."
python -c "
import mlx_whisper
print('  Downloading $WHISPER_MODEL...')
mlx_whisper.transcribe('/dev/null', path_or_hf_repo='$WHISPER_MODEL')
print('  Done.')
" 2>/dev/null || warn "Whisper download failed — will download on first use."

# Download embedding model
log "Downloading embedding model..."
python -c "
from sentence_transformers import SentenceTransformer
print('  Downloading $EMBED_MODEL...')
SentenceTransformer('$EMBED_MODEL')
print('  Done.')
" || warn "Embedding model download failed — will download on first use."

# Download VL model (largest — warn about size)
log "Downloading VL model (may be slow, ~4-5 GB)..."
python -c "
from mlx_vlm import load
print('  Downloading $VL_MODEL...')
load('$VL_MODEL')
print('  Done.')
" || warn "VL model download failed — will download on first use."

# Pre-download Qwen3-TTS model
TTS_MODEL="$(grep '^TTS_MODEL=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit')"
TTS_MODEL="${TTS_MODEL:-mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit}"
log "Downloading Qwen3-TTS model: $TTS_MODEL (~2 GB)..."
python -c "
from mlx_audio.tts.utils import load_model
print('  Downloading $TTS_MODEL...')
load_model('$TTS_MODEL')
print('  Done.')
" || warn "Qwen3-TTS download failed — will download on first use."

# ── Step 10: launchd Services ─────────────────────────────────────────────────
step "Installing launchd services (auto-start on login)"
mkdir -p "$LAUNCHD_DIR"
mkdir -p "$LOG_DIR"

# Main ella process plist
cat > "$LAUNCHD_DIR/$ELLA_PLIST_MAIN" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ella.main</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_DIR/bin/python</string>
        <string>-m</string>
        <string>ella.main</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>VIRTUAL_ENV</key>
        <string>$VENV_DIR</string>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/ella-main.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/ella-main-error.log</string>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
PLIST

# Celery worker plist
cat > "$LAUNCHD_DIR/$ELLA_PLIST_WORKER" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ella.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_DIR/bin/celery</string>
        <string>-A</string>
        <string>ella.tasks.celery_app</string>
        <string>worker</string>
        <string>--loglevel=info</string>
        <string>--concurrency=1</string>
        <string>--pool=solo</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>VIRTUAL_ENV</key>
        <string>$VENV_DIR</string>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/ella-worker.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/ella-worker-error.log</string>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
PLIST

log "launchd plists written to $LAUNCHD_DIR"

# ── Step 11: Start Script ─────────────────────────────────────────────────────
step "Creating start/stop helper scripts"

cat > "$SCRIPT_DIR/start.sh" <<'STARTSH'
#!/usr/bin/env bash
# Start Ella (main process + Celery worker)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

echo "Starting Ella..."
launchctl load -w "$LAUNCHD_DIR/com.ella.worker.plist"
launchctl load -w "$LAUNCHD_DIR/com.ella.main.plist"
echo "Ella started. Logs: ~/Library/Logs/ella/"
echo "  Main:   tail -f ~/Library/Logs/ella/ella-main.log"
echo "  Worker: tail -f ~/Library/Logs/ella/ella-worker.log"
STARTSH

cat > "$SCRIPT_DIR/stop.sh" <<'STOPSH'
#!/usr/bin/env bash
# Stop Ella (main process + Celery worker)
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

echo "Stopping Ella..."
launchctl unload "$LAUNCHD_DIR/com.ella.main.plist"    2>/dev/null || true
launchctl unload "$LAUNCHD_DIR/com.ella.worker.plist"  2>/dev/null || true
echo "Ella stopped."
STOPSH

cat > "$SCRIPT_DIR/logs.sh" <<'LOGSH'
#!/usr/bin/env bash
# Tail all Ella logs
trap 'kill 0' EXIT
tail -f \
    ~/Library/Logs/ella/ella-main.log \
    ~/Library/Logs/ella/ella-main-error.log \
    ~/Library/Logs/ella/ella-worker.log \
    ~/Library/Logs/ella/ella-worker-error.log
LOGSH

chmod +x "$SCRIPT_DIR/start.sh" "$SCRIPT_DIR/stop.sh" "$SCRIPT_DIR/logs.sh"
log "Helper scripts created: start.sh  stop.sh  logs.sh"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  Ella AI Agent — Deployment Complete${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo ""

if grep -q "^TELEGRAM_BOT_TOKEN=your_bot_token_here$" "$SCRIPT_DIR/.env" 2>/dev/null; then
    echo -e "  ${YELLOW}1. Configure your Telegram bot token:${NC}"
    echo -e "     nano $SCRIPT_DIR/.env"
    echo -e "     (Set TELEGRAM_BOT_TOKEN to your token from @BotFather)"
    echo ""
fi

if [[ ! -f "$SCRIPT_DIR/assets/speaker.wav" ]]; then
    echo -e "  ${YELLOW}2. (Optional) Add a reference voice WAV for Qwen3-TTS voice cloning:${NC}"
    echo -e "     Copy a ≥3 second WAV to: $SCRIPT_DIR/assets/speaker.wav"
    echo ""
fi

echo -e "  ${BOLD}Start Ella:${NC}"
echo -e "     ./start.sh"
echo ""
echo -e "  ${BOLD}Stop Ella:${NC}"
echo -e "     ./stop.sh"
echo ""
echo -e "  ${BOLD}View logs:${NC}"
echo -e "     ./logs.sh"
echo -e "     (or: tail -f ~/Library/Logs/ella/ella-main.log)"
echo ""
echo -e "  ${BOLD}Infrastructure:${NC}"
echo -e "     docker compose ps         — check Redis/Qdrant status"
echo -e "     docker compose down       — stop Redis/Qdrant"
echo ""
echo -e "  ${BOLD}Add custom tools (hot-reloaded, no restart):${NC}"
echo -e "     Drop .py files into: $SCRIPT_DIR/ella/tools/custom/"
echo ""
echo -e "  ${BOLD}Personalise Ella (hot-reloaded, no restart):${NC}"
echo -e "     ~/Ella/Identity.md   — name, age, timezone, background, relationship with you"
echo -e "     ~/Ella/Soul.md       — personality, tone, values"
echo -e "     ~/Ella/User.md       — who you are (DOB, ethnicity, country, profession)"
echo ""
