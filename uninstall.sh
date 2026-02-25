#!/usr/bin/env bash
# =============================================================================
# Ella AI Agent — Uninstall Script
#
# Usage:
#   ./uninstall.sh          — remove services, venv, containers, logs
#   ./uninstall.sh --rm     — same + delete all downloaded ML model files
#
# Removes (always):
#   - launchd services (stops + unloads + deletes plists)
#   - Python virtual environment (.venv)
#   - Docker containers and volumes for Redis + Qdrant
#   - Log files (~/Library/Logs/ella/)
#   - Generated helper scripts (start.sh, stop.sh, logs.sh)
#
# Removes (with --rm only):
#   - HuggingFace model cache entries for models used by Ella:
#       mlx-community/Qwen2.5-7B-Instruct-4bit
#       mlx-community/Qwen2.5-VL-3B-Instruct-4bit  (or 7B variant)
#       mlx-community/whisper-small  (or other configured variant)
#       paraphrase-multilingual-MiniLM-L12-v2
#       tts_models/multilingual/multi-dataset/xtts_v2  (TTS cache)
#   - ~/Ella/ identity files (Identity.md, Soul.md, User.md)
#
# Does NOT remove (ever):
#   - Homebrew, Python, pyenv, Docker Desktop
#   - Installed Python packages
#   - Project source code
#   - Your .env file
#   - Any OTHER models in ~/.cache/huggingface not belonging to Ella
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[ella]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
skip() { echo -e "       ${YELLOW}(skipped — not found)${NC}"; }
step() { echo -e "\n${BOLD}${BLUE}── $* ──${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/ella"

# ── Parse flags ───────────────────────────────────────────────────────────────
REMOVE_MODELS=0
for ARG in "$@"; do
    case "$ARG" in
        --rm) REMOVE_MODELS=1 ;;
        *) echo "Unknown flag: $ARG"; echo "Usage: ./uninstall.sh [--rm]"; exit 1 ;;
    esac
done

# ── Read model names from .env (fall back to defaults) ────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"
ENV_SOURCE="$SCRIPT_DIR/.env.example"
[[ -f "$ENV_FILE" ]] && ENV_SOURCE="$ENV_FILE"

_env_val() {
    grep "^$1=" "$ENV_SOURCE" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'"
}

CHAT_MODEL="$(_env_val MLX_CHAT_MODEL)"
VL_MODEL="$(_env_val MLX_VL_MODEL)"
WHISPER_MODEL="$(_env_val MLX_WHISPER_MODEL)"
EMBED_MODEL="$(_env_val EMBED_MODEL)"

CHAT_MODEL="${CHAT_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
VL_MODEL="${VL_MODEL:-mlx-community/Qwen2.5-VL-3B-Instruct-4bit}"
WHISPER_MODEL="${WHISPER_MODEL:-mlx-community/whisper-small}"
EMBED_MODEL="${EMBED_MODEL:-paraphrase-multilingual-MiniLM-L12-v2}"

# HuggingFace cache root
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_CACHE="$HF_CACHE/hub"

# XTTS-v2 is cached by the TTS library in a different location
TTS_CACHE="$HOME/Library/Application Support/tts"

# ── Confirm ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${RED}Ella AI Agent — Uninstall${NC}"
echo ""
echo "This will remove:"
echo "  • launchd services (com.ella.main, com.ella.worker)"
echo "  • Python virtual environment ($SCRIPT_DIR/.venv)"
echo "  • Docker containers + volumes for Redis and Qdrant"
echo "  • Log files ($LOG_DIR)"
echo "  • Helper scripts (start.sh, stop.sh, logs.sh)"

if [[ "$REMOVE_MODELS" -eq 1 ]]; then
    echo ""
    echo -e "  ${RED}• ML model files (--rm flag):${NC}"
    echo "      $CHAT_MODEL"
    echo "      $VL_MODEL"
    echo "      $WHISPER_MODEL"
    echo "      $EMBED_MODEL"
    echo "      tts_models/multilingual/multi-dataset/xtts_v2"
    echo -e "  ${RED}• Identity files (--rm flag): ~/Ella/${NC}"
fi

echo ""
echo "This will NOT remove:"
echo "  • Homebrew, Python, pyenv, Docker Desktop"
if [[ "$REMOVE_MODELS" -eq 0 ]]; then
    echo "  • HuggingFace model cache (~/.cache/huggingface)"
    echo "  • ~/Ella/ identity files (your personalisation)"
fi
echo "  • Project source code"
echo "  • Your .env file"
echo ""
read -r -p "Continue? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ── Step 1: Stop + unload launchd services ────────────────────────────────────
step "Removing launchd services"

for LABEL in com.ella.main com.ella.worker; do
    PLIST="$LAUNCHD_DIR/$LABEL.plist"
    if [[ -f "$PLIST" ]]; then
        # Unload (stops process if running)
        launchctl unload "$PLIST" 2>/dev/null || true
        rm -f "$PLIST"
        log "Removed: $PLIST"
    else
        echo -n "  $LABEL.plist"
        skip
    fi
done

# ── Step 2: Docker services ───────────────────────────────────────────────────
step "Stopping and removing Docker containers + volumes"

if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    cd "$SCRIPT_DIR"
    if [[ -f "docker-compose.yml" ]]; then
        docker compose down --volumes --remove-orphans 2>/dev/null || true
        log "Docker containers and volumes removed."
    else
        warn "docker-compose.yml not found — skipping Docker cleanup."
    fi
else
    warn "Docker not running — skipping Docker cleanup."
fi

# ── Step 3: Virtual environment ───────────────────────────────────────────────
step "Removing Python virtual environment"

VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
    log "Removed: $VENV_DIR"
else
    echo -n "  .venv"
    skip
fi

# ── Step 4: Log files ─────────────────────────────────────────────────────────
step "Removing log files"

if [[ -d "$LOG_DIR" ]]; then
    rm -rf "$LOG_DIR"
    log "Removed: $LOG_DIR"
else
    echo -n "  $LOG_DIR"
    skip
fi

# ── Step 5: Helper scripts ────────────────────────────────────────────────────
step "Removing helper scripts"

for SCRIPT in start.sh stop.sh logs.sh; do
    TARGET="$SCRIPT_DIR/$SCRIPT"
    if [[ -f "$TARGET" ]]; then
        rm -f "$TARGET"
        log "Removed: $TARGET"
    else
        echo -n "  $SCRIPT"
        skip
    fi
done

# ── Step 6: ML model cache (only with --rm) ───────────────────────────────────
if [[ "$REMOVE_MODELS" -eq 1 ]]; then
    step "Removing ML model files"

    # HuggingFace hub cache: each model is stored as
    # ~/.cache/huggingface/hub/models--<org>--<name>/
    # The org/name separator "/" becomes "--" in the directory name.

    _hf_dir() {
        # Convert "org/model-name" → "models--org--model-name"
        echo "models--$(echo "$1" | tr '/' '--')"
    }

    for MODEL in "$CHAT_MODEL" "$VL_MODEL" "$WHISPER_MODEL" "$EMBED_MODEL"; do
        MODEL_DIR="$HF_HUB_CACHE/$(_hf_dir "$MODEL")"
        if [[ -d "$MODEL_DIR" ]]; then
            # Show size before deleting
            SIZE="$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"
            rm -rf "$MODEL_DIR"
            log "Removed model ($SIZE): $MODEL"
        else
            echo -n "  $MODEL"
            skip
        fi
    done

    # XTTS-v2 — stored by the TTS library under ~/Library/Application Support/tts
    XTTS_DIR="$TTS_CACHE/tts_models--multilingual--multi-dataset--xtts_v2"
    if [[ -d "$XTTS_DIR" ]]; then
        SIZE="$(du -sh "$XTTS_DIR" 2>/dev/null | cut -f1)"
        rm -rf "$XTTS_DIR"
        log "Removed XTTS-v2 model ($SIZE): $XTTS_DIR"
    else
        # Also check alternate flat layout used by some TTS versions
        XTTS_ALT="$TTS_CACHE/tts_models"
        if [[ -d "$XTTS_ALT" ]]; then
            SIZE="$(du -sh "$XTTS_ALT" 2>/dev/null | cut -f1)"
            rm -rf "$XTTS_ALT"
            log "Removed TTS model cache ($SIZE): $XTTS_ALT"
        else
            echo -n "  XTTS-v2 TTS cache"
            skip
        fi
    fi

    # Tidy up empty HF hub directory if nothing else is in it
    if [[ -d "$HF_HUB_CACHE" ]] && [[ -z "$(ls -A "$HF_HUB_CACHE" 2>/dev/null)" ]]; then
        rmdir "$HF_HUB_CACHE" 2>/dev/null || true
        log "Removed empty HuggingFace hub cache directory."
    fi

    # ~/Ella/ identity files — only removed with --rm since they contain
    # the user's personal customisations
    step "Removing identity files (~/Ella/)"
    ELLA_DIR="$HOME/Ella"
    if [[ -d "$ELLA_DIR" ]]; then
        rm -rf "$ELLA_DIR"
        log "Removed: $ELLA_DIR"
    else
        echo -n "  ~/Ella/"
        skip
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  Ella uninstalled successfully.${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "Remaining items (not removed):"
echo "  • Source code:    $SCRIPT_DIR"
echo "  • .env file:      $SCRIPT_DIR/.env  (if present)"
if [[ "$REMOVE_MODELS" -eq 0 ]]; then
    echo "  • Model cache:    $HF_HUB_CACHE"
    echo "  • Identity files: ~/Ella/"
    echo "                    (re-run with --rm to also delete models + identity files)"
fi
echo ""
echo "To fully remove the project, delete the source directory:"
echo "  rm -rf \"$SCRIPT_DIR\""
echo ""
