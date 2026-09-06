#!/bin/bash
# setup-mac.sh — one-shot setup for omnilingual live transcription on a Mac.
#
# Installs: Xcode CLT (prompt), Homebrew, blackhole-2ch, ffmpeg,
# switchaudio-osx, uv, git; creates the Aggregate Device "Omnilingual"
# (BlackHole first, mic last — order is load-bearing, the capture downmix
# selects the last channel as the mic) and the stacked Multi-Output Device;
# switches system output to it; optionally clones this repo and runs --check-audio.
#
# Usage:
#   ./scripts/setup-mac.sh [--repo <git-url>] [--branch <name>] [--yes]
# Examples:
#   ./scripts/setup-mac.sh
#   ./scripts/setup-mac.sh --repo git@github.com:you/omnilingual.git --branch feat/live-transcription
#
# Safe to re-run: every step is idempotent. Needs sudo once (coreaudiod
# restart so BlackHole appears immediately); without it, that step is
# skipped and a reboot may be needed on a fresh Mac.

set -euo pipefail

REPO_URL=""
BRANCH="feat/live-transcription"
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_URL="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        -y|--yes) ASSUME_YES=1; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HELPER_SRC="$SCRIPT_DIR/audio-devices.swift"
HELPER_BIN="$(mktemp -t audio-devices).bin"
trap 'rm -f "$HELPER_BIN"' EXIT

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname)" == "Darwin" ]] || die "this script is macOS-only"

# 1. Xcode command-line tools (provides git, swiftc).
if ! xcode-select -p >/dev/null 2>&1; then
    say "installing Xcode command-line tools (follow the Apple prompt, then re-run me)"
    xcode-select --install
    exit 0
fi

# 2. Homebrew (skip if already present).
if ! command -v brew >/dev/null 2>&1; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/null
fi
if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi
command -v brew >/dev/null 2>&1 || die "Homebrew install failed; install it manually, then re-run me"

# 3. Packages (brew install is idempotent).
say "installing packages (blackhole-2ch ffmpeg switchaudio-osx uv git)…"
brew install blackhole-2ch ffmpeg switchaudio-osx uv git

# 4. Fresh HAL drivers need coreaudiod restarted to appear.
if sudo -n true 2>/dev/null; then
    say "restarting coreaudiod so BlackHole appears…"
    sudo killall coreaudiod 2>/dev/null || true
elif [[ -t 0 ]]; then
    say "restarting coreaudiod so BlackHole appears (one sudo prompt)…"
    sudo killall coreaudiod 2>/dev/null || true
else
    say "NOTE: no sudo access, skipping coreaudiod restart."
    say "If BlackHole does not appear below, reboot (or log out/in) and re-run me."
fi
sleep 2

# 5. Build the audio-device helper and wait for BlackHole.
say "building audio-device helper…"
swiftc -o "$HELPER_BIN" "$HELPER_SRC"
say "waiting for BlackHole 2ch…"
for _ in $(seq 1 30); do
    "$HELPER_BIN" list 2>/dev/null | grep -q "BlackHole" && break
    sleep 2
done
"$HELPER_BIN" list 2>/dev/null | grep -q "BlackHole" \
    || die "BlackHole 2ch never appeared; reboot and re-run me"

# 6. Create the virtual devices (idempotent) and verify.
say "creating Aggregate + Multi-Output devices…"
"$HELPER_BIN" ensure || die "device creation failed"
"$HELPER_BIN" check || die "devices missing after creation"
"$HELPER_BIN" list

# 7. Route system audio through the Multi-Output Device so meetings
# reach BlackHole. (Volume keys stop working on it; set volume first.
# Revert anytime with: SwitchAudioSource -s "<your speakers>".)
say "routing system output to Multi-Output Device…"
if [[ $ASSUME_YES == 0 ]]; then
    read -r -p "Switch system output to 'Multi-Output Device'? [Y/n] " ans
    [[ "$ans" == [nN]* ]] && { say "skipped (live mode needs it during meetings)"; }
fi
if [[ $ASSUME_YES == 1 || "$ans" != [nN]* ]]; then
    SwitchAudioSource -s "Multi-Output Device"
fi
say "current output: $(SwitchAudioSource -c)"

# 8. Optional: clone the repo, verify the test suite, probe the mic.
if [[ -n "$REPO_URL" ]]; then
    say "cloning $REPO_URL (branch $BRANCH)…"
    [[ -d omnilingual ]] || git clone --branch "$BRANCH" "$REPO_URL" omnilingual
    cd omnilingual
    say "running test suite…"
    uv run pytest -q
    say "probing capture chain (speak NOW for the voice verdict)…"
    uv run omnilingual live --check-audio || true
    say "setup complete. Next:"
    say "  export SARVAM_API_KEY=<key>"
    say "  uv run omnilingual live --out standup.md   # Ctrl+C stops; file stays valid"
else
    say "setup complete. In your omnilingual checkout, run:"
    say "  uv run pytest -q"
    say "  export SARVAM_API_KEY=<key>"
    say "  uv run omnilingual live --check-audio   # speak during it: expect VOICE LIKELY"
    say "  uv run omnilingual live --out standup.md"
fi
