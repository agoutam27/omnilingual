#!/bin/bash
# setup-mac.sh — install and configure omnilingual on a Mac, re-runnably.
#
# Installs the toolchain (Homebrew, git, uv, ffmpeg), activates the optional
# dependency sets ("extras") you pick — local speech-to-text, speaker labels,
# offline translation, the local UI — stores API keys in .env, optionally wires
# live capture through a BlackHole aggregate device, and pre-downloads the model
# weights so the first transcription does not stall on a 1.5 GB fetch.
#
# Re-run it any time: every step converges the machine toward your current
# answers instead of accumulating state. Deselect an extra and its packages are
# actually uninstalled; deselect live mode and the audio devices are left alone;
# your existing .env keys are never clobbered.
#
# Usage:
#   ./scripts/setup-mac.sh [options]
#
# Options:
#   --repo <git-url>   Clone this repo (only used when this script is not already
#                      sitting inside a checkout).
#   --branch <name>    Branch to clone. Default: the remote's HEAD.
#   -y, --yes          Non-interactive. Take saved answers, never prompt for a
#                      secret. First run falls back to the documented defaults.
#   --dry-run          Print the resolved plan and the detected diff, then exit
#                      without touching anything.
#   --reset            Forget saved answers and re-ask from the defaults.
#   -h, --help         Show this help.
#
# Saved answers live in ${XDG_CONFIG_HOME:-~/.config}/omnilingual/setup-mac.conf,
# outside the repo so a re-clone cannot lose them. That file records which
# capabilities you enabled — never any key material; secrets go only to .env
# (gitignored, chmod 600).
#
# All narration goes to stderr so stdout stays reserved for machine-readable
# values.
#
# Xcode command-line tools are required (they provide git and swiftc). If they
# are missing this script triggers the Apple prompt and exits; just re-run it
# afterwards. The coreaudiod restart needs sudo once, and is skipped with a
# reboot hint if sudo is unavailable.

set -euo pipefail

# Nothing this script writes may be group- or world-readable, even briefly.
umask 077

# ---------------------------------------------------------------- entry flags

REPO_URL=""
BRANCH=""
FLAG_REPO_URL=""
FLAG_BRANCH=""
REPO_URL_SET=0
BRANCH_SET=0
ASSUME_YES=0
DRY_RUN=0
RESET=0

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed -e 's/^# \{0,1\}//' -e '$d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)   [[ $# -ge 2 ]] || { echo "--repo needs a value" >&2; exit 2; }; FLAG_REPO_URL="$2"; REPO_URL_SET=1; shift 2 ;;
        --branch) [[ $# -ge 2 ]] || { echo "--branch needs a value" >&2; exit 2; }; FLAG_BRANCH="$2"; BRANCH_SET=1; shift 2 ;;
        -y|--yes)   ASSUME_YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --reset)   RESET=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
    esac
done

# ------------------------------------------------------------------- messaging
# Everything human-facing goes to stderr. stdout carries only captured values,
# which keeps `EXTRAS=$(prompt_extras)` honest.

say()  { printf '\033[1m==>\033[0m %s\n' "$*" >&2; }
step() { printf '\033[1m  *\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$*" >&2; }
skip() { printf '\033[2m  --\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ primitives

# in_list <needle> <comma-list>
in_list() { case ",$2," in *",$1,"*) return 0 ;; esac; return 1; }

# yes_no <value> -> 0 when the value reads as yes
yes_no() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        y|yes|true|1) return 0 ;;
    esac
    return 1
}

# sanitize <space-separated-allowed> <comma-list> -> deduplicated, ordered,
# filtered comma list. Keeps a hand-edited config file from smuggling in an
# extra that pyproject.toml does not declare.
sanitize() {
    local allowed="$1" list="$2" out="" item oldifs="$IFS"
    IFS=,
    set -f   # unquoted $list must not glob-expand against the current directory
    for item in $list; do
        item="$(printf '%s' "$item" | tr -d '[:space:]')"
        [[ -z "$item" ]] && continue
        case " $allowed " in
            *" $item "*) in_list "$item" "$out" || out="${out:+$out,}$item" ;;
        esac
    done
    set +f
    IFS="$oldifs"
    printf '%s' "$out"
}

interactive() { [[ $ASSUME_YES -eq 0 && -t 0 ]]; }

# confirm <question> <default: y|n>
confirm() {
    local q="$1" def="${2:-y}" hint ans
    if yes_no "$def"; then hint="[Y/n]"; else hint="[y/N]"; fi
    if ! interactive; then yes_no "$def"; return $?; fi
    read -r -p "$q $hint " ans || ans=""
    [[ -z "$ans" ]] && ans="$def"
    yes_no "$ans"
}

# ------------------------------------------------------------------- host facts

[[ "$(uname)" == "Darwin" ]] || die "this script is macOS-only"

# Floor is set by onnxruntime 1.23, whose oldest wheels are macosx_13_0; that is the
# highest requirement among every non-MLX combination on both architectures. MLX needs
# 14 and is reported separately, only when local-stt is actually selected.
MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo 0)"
MACOS_MAJOR="$(printf '%s' "$MACOS_VERSION" | cut -d. -f1)"
[[ "$MACOS_MAJOR" =~ ^[0-9]+$ ]] || MACOS_MAJOR=0
if (( MACOS_MAJOR < 13 )); then
    die "macOS 13 (Ventura) or newer is required; this Mac reports macOS ${MACOS_VERSION}"
fi

# `uname -m` reports x86_64 inside a Rosetta 2 shell even on Apple Silicon, so hardware
# arm64 is read from sysctl instead. Trusting uname alone would tell an M-series Mac
# "MLX does not exist here" and quietly steer it to the slower faster-whisper backend.
ARCH="$(uname -m)"
HARDWARE_ARM64="$(sysctl -n hw.optional.arm64 2>/dev/null || true)"
TRANSLATED=0
if [[ "$(sysctl -n sysctl.proc_translated 2>/dev/null || true)" == "1" ]]; then
    TRANSLATED=1
fi
if [[ "$ARCH" == "arm64" || "$HARDWARE_ARM64" == "1" || $TRANSLATED -eq 1 ]]; then
    ARM64=1
else
    ARM64=0
fi

# Must precede any formula probe: on Apple Silicon brew is off PATH until this runs
brew_env() {
    if [[ -x /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -x /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}
brew_env

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/omnilingual"
CONFIG_FILE="$CONFIG_DIR/setup-mac.conf"

TMP_PATHS=()
cleanup() { local p; for p in ${TMP_PATHS[@]+"${TMP_PATHS[@]}"}; do rm -f "$p"; done; }
trap cleanup EXIT

# ---------------------------------------------------------------- configuration

ALL_EXTRAS="local-stt diarize local-mt ui"
ALL_KEYS="SARVAM_API_KEY GROQ_API_KEY GEMINI_API_KEY"

EXTRAS="local-stt,diarize"
KEYS=""
LIVE_SETUP="no"
ROUTE_OUTPUT="no"
PREFETCH="yes"
RUN_TESTS="yes"

# Parsed with a whitelist loop rather than `source`: this file is read by a
# script that runs sudo, so it must never become a chance to execute code.
load_config() {
    [[ -f "$CONFIG_FILE" ]] || return 0
    local line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" == *=* ]] || continue
        key="$(printf '%s' "${line%%=*}" | tr -d '[:space:]')"
        value="$(printf '%s' "${line#*=}" | tr -d '[:space:]')"
        case "$key" in
            EXTRAS|KEYS|LIVE_SETUP|ROUTE_OUTPUT|PREFETCH|RUN_TESTS|REPO_URL|BRANCH)
                printf -v "$key" '%s' "$value" ;;
            *)
                : ;;  # unknown key: ignored, never executed
        esac
    done < "$CONFIG_FILE"
}

save_config() {
    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"
    {
        printf '# omnilingual setup-mac.sh — saved answers. Re-run the script to change them.\n'
        printf '# Contains no secrets; API keys live in the repo .env instead.\n'
        printf '# Written %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        printf 'EXTRAS=%s\n' "$EXTRAS"
        printf 'KEYS=%s\n' "$KEYS"
        printf 'LIVE_SETUP=%s\n' "$LIVE_SETUP"
        printf 'ROUTE_OUTPUT=%s\n' "$ROUTE_OUTPUT"
        printf 'PREFETCH=%s\n' "$PREFETCH"
        printf 'RUN_TESTS=%s\n' "$RUN_TESTS"
        printf 'REPO_URL=%s\n' "$REPO_URL"
        printf 'BRANCH=%s\n' "$BRANCH"
    } > "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
}

if [[ $RESET -eq 1 ]]; then
    if [[ -f "$CONFIG_FILE" ]]; then
        rm -f "$CONFIG_FILE"
        say "reset: forgotten saved answers at $CONFIG_FILE"
    fi
elif [[ -f "$CONFIG_FILE" ]]; then
    load_config
fi

# An explicit flag on this run outranks whatever was saved last time.
if [[ $REPO_URL_SET -eq 1 ]]; then REPO_URL="$FLAG_REPO_URL"; fi
if [[ $BRANCH_SET -eq 1 ]]; then BRANCH="$FLAG_BRANCH"; fi

EXTRAS="$(sanitize "$ALL_EXTRAS" "$EXTRAS")"
KEYS="$(sanitize "$ALL_KEYS" "$KEYS")"
# Reached with a saved config too, where the extras prompt above never runs.
if [[ $ARM64 -eq 1 ]] && (( MACOS_MAJOR < 14 )) && in_list local-stt "$EXTRAS"; then
    warn "local-stt will install, but MLX needs macOS 14+ and this Mac is macOS $MACOS_VERSION."
    warn "Offline transcription will fail; use --stt faster-whisper (CPU) or update macOS."
fi
# Routing the system output only means something once the devices exist.
if ! yes_no "$LIVE_SETUP"; then ROUTE_OUTPUT="no"; fi

# ------------------------------------------------------------- locate the repo

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    # The script was fetched on its own; find or ask for the repository.
    if [[ -z "$REPO_URL" && -d "$REPO_ROOT/.git" ]]; then
        REPO_URL="$(git -C "$REPO_ROOT" config --get remote.origin.url || true)"
    fi
    [[ -n "$REPO_URL" ]] || die "not inside a checkout and no --repo given; pass --repo <git-url>"
    say "cloning $REPO_URL${BRANCH:+ (branch $BRANCH)}…"
    TARGET="$PWD/omnilingual"
    if [[ -d "$TARGET/.git" ]]; then
        step "reusing existing clone at $TARGET"
    elif [[ $DRY_RUN -eq 1 ]]; then
        step "would clone into $TARGET"
        exit 0
    else
        if [[ -n "$BRANCH" ]]; then
            git clone --branch "$BRANCH" -- "$REPO_URL" "$TARGET"
        else
            git clone -- "$REPO_URL" "$TARGET"
        fi
    fi
    REPO_ROOT="$TARGET"
    SCRIPT_DIR="$REPO_ROOT/scripts"
fi

cd "$REPO_ROOT"
ENV_FILE="$REPO_ROOT/.env"

# Run a command with the repo as the working directory.
in_repo() { ( cd "$REPO_ROOT" && "$@" ); }

# ------------------------------------------------------------------- prompting

prompt_extras() {
    local cur="$1" ans="$1"
    if interactive; then
        {
            printf '\n'
            say "Capabilities to install"
            printf '    %s\n' \
                "local-stt  offline speech-to-text (mlx-whisper here; faster-whisper on Intel)" \
                "diarize    speaker labels via sherpa-onnx   (+86 MB of models)" \
                "local-mt   offline Indic translation via CTranslate2 (+850 MB of models)" \
                "ui         local browser UI window (FastAPI + pywebview), no models to fetch"
            if [[ $ARM64 -eq 1 ]]; then
                printf '    this Mac is Apple Silicon, so local-stt uses mlx-whisper.\n'
                if (( MACOS_MAJOR < 14 )); then
                    printf '    note: MLX needs macOS 14+ but this Mac is macOS %s, so pass\n' "$MACOS_VERSION"
                    printf '    --stt faster-whisper to transcribe offline on CPU instead.\n'
                fi
                if [[ $TRANSLATED -eq 1 ]]; then
                    printf '    note: this shell is translated by Rosetta 2; re-run from a native\n'
                    printf '    Terminal to get native arm64 packages.\n'
                fi
            else
                printf '    this Mac is Intel (x86_64): MLX does not exist here, so local-stt uses\n'
                printf '    faster-whisper on CPU instead.\n'
            fi
            printf '    comma-separated, or none. Enter keeps [%s]\n' "$cur"
        } >&2
        read -r -p "    capabilities: " ans || ans=""
        [[ -z "$ans" ]] && ans="$cur"
    fi
    sanitize "$ALL_EXTRAS" "$ans"
}

prompt_keys() {
    local cur="$1" ans="$1"
    if interactive; then
        {
            printf '\n'
            say "API keys to keep in .env"
            printf '    %s\n' \
                "SARVAM_API_KEY  Sarvam speech-to-text (saaras:v4) and Mayura translation" \
                "GROQ_API_KEY    Groq Whisper speech-to-text, a cheap cloud fallback" \
                "GEMINI_API_KEY  Gemini translation"
            [[ -z "$cur" ]] && printf '    With local-stt + local-mt + diarize you need no keys at all.\n'
            printf '    comma-separated names, or none. Enter keeps [%s]\n' "${cur:-none}"
        } >&2
        read -r -p "    keys: " ans || ans=""
        [[ -z "$ans" ]] && ans="$cur"
    fi
    sanitize "$ALL_KEYS" "$ans"
}

# ask_yes_no_setting <var-name> <question>
ask_yes_no_setting() {
    local name="$1" q="$2"
    if confirm "$q" "${!name}"; then printf -v "$name" '%s' yes; else printf -v "$name" '%s' no; fi
}

# ------------------------------------------------------------------- env file

# env_get <key> -> value, empty when absent or blank
env_get() {
    local key="$1"
    [[ -f "$ENV_FILE" ]] || return 0
    sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "$ENV_FILE" | tail -n 1
}

# env_set <key> <value> — rewrite in place or append, preserving comments and
# every other line. Pure bash builtins: the value never reaches a subprocess's
# argv, so it cannot show up in `ps`.
env_set() {
    local key="$1" value="$2" tmp line found=0
    tmp="$(mktemp "${TMPDIR:-/tmp}/omnilingual-env.XXXXXX")"
    TMP_PATHS+=("$tmp")
    if [[ -f "$ENV_FILE" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" =~ ^[[:space:]]*$key[[:space:]]*= ]]; then
                printf '%s=%s\n' "$key" "$value" >> "$tmp"
                found=1
            else
                printf '%s\n' "$line" >> "$tmp"
            fi
        done < "$ENV_FILE"
    fi
    [[ $found -eq 1 ]] || printf '%s=%s\n' "$key" "$value" >> "$tmp"
    mv "$tmp" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
}

# ask_key_value <key> — sets NEW_KEY_VALUE, or leaves it empty to keep current
ask_key_value() {
    local key="$1" current ans
    current="$(env_get "$key")"
    NEW_KEY_VALUE=""
    if [[ -n "$current" ]]; then
        interactive || return 0
        if ! confirm "    $key is already set in .env — replace it?" n; then
            skip "$key already set in .env, leaving it alone"
            return 0
        fi
    fi
    interactive || return 0
    printf '    value for %s (input hidden, blank to skip): ' "$key" >&2
    if ! read -r -s ans; then ans=""; fi
    printf '\n' >&2
    [[ -z "$ans" ]] || NEW_KEY_VALUE="$ans"
}

# --------------------------------------------------------------- state probing

# No `| grep -q` here: grep exits on first match, the producer takes SIGPIPE, and
# pipefail then reports the whole pipeline as failed even though the match hit.
brew_has() {
    local installed
    installed="$(brew list --formula 2>/dev/null || true)"
    grep -qx "$1" <<<"$installed"
}

# extra_active <extra> — is its marker package importable right now? --no-sync so
# that merely looking never mutates the venv.
extra_active() {
    local pkg
    case "$1" in
        local-stt) if [[ $ARM64 -eq 1 ]]; then pkg=mlx_whisper; else pkg=faster_whisper; fi ;;
        diarize)   pkg=sherpa_onnx ;;
        local-mt)  pkg=ctranslate2 ;;
        # fastapi is the marker: declared first by the ui extra, pulled in by no
        # other, and spelled the same as the distribution (unlike pywebview/webview).
        ui)         pkg=fastapi ;;
        *)         return 1 ;;
    esac
    in_repo uv run --no-sync python -c \
        'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' \
        "$pkg" >/dev/null 2>&1
}

# prefetchable <extra> -> 0 when the extra ships weights worth pre-downloading. Kept
# as its own predicate, rather than inferred from an empty prefetch_label, so that
# an extra with no prefetch branch is skipped outright: the prefetch_models default
# arm returns failure, and reporting "could not pre-download the ui models" for a
# capability that has none would be a lie.
prefetchable() {
    case "$1" in
        local-stt|diarize|local-mt) return 0 ;;
        *) return 1 ;;
    esac
}

prefetch_label() {
    case "$1" in
        diarize)
            printf 'sherpa diarization models (~86 MB)'
            ;;
        local-stt)
            if [[ $ARM64 -eq 1 ]]; then printf 'mlx-whisper large-v3-turbo (~1.5 GB)'
            else printf 'faster-whisper small (~500 MB)'; fi
            ;;
        local-mt)
            printf 'IndicTrans2 int8 (~850 MB)'
            ;;
    esac
}

prefetch_models() {
    case "$1" in
        diarize)
            in_repo uv run python - <<'PY'
from omnilingual.config import Settings
from omnilingual.diarize.sherpa import SherpaDiarizer
for p in SherpaDiarizer(Settings(api_key=None)).ensure_models():
    print(p)
PY
            ;;
        local-stt)
            if [[ $ARM64 -eq 1 ]]; then
                in_repo uv run python - <<'PY'
import mlx.core as mx
from mlx_whisper.transcribe import ModelHolder
from omnilingual.config import DEFAULT_STT_MODELS
print(ModelHolder.get_model(DEFAULT_STT_MODELS["mlx-whisper"], mx.float16))
PY
            else
                in_repo uv run python - <<'PY'
from faster_whisper import WhisperModel
from omnilingual.config import DEFAULT_STT_MODELS
WhisperModel(DEFAULT_STT_MODELS["faster-whisper"], device="cpu", compute_type="int8")
print("ok")
PY
            fi
            ;;
        local-mt)
            in_repo uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from omnilingual.config import DEFAULT_MT_MODELS
print(snapshot_download(
    DEFAULT_MT_MODELS["indictrans2"],
    local_dir=Path.home() / ".cache" / "omnilingual" / "models" / "indictrans2",
    allow_patterns=["*/ctranslate2_model/*"],
))
PY
            ;;
        *)
            return 1
            ;;
    esac
}

# ============================================================== 1. ask

EXTRAS="$(prompt_extras "$EXTRAS")"
KEYS="$(prompt_keys "$KEYS")"

ask_yes_no_setting LIVE_SETUP "Set up live capture (install BlackHole, create the 'Omnilingual' device)?"
if yes_no "$LIVE_SETUP"; then
    say "Live capture needs system output routed through BlackHole so meeting audio"
    say "reaches the recorder. Volume keys stop working on that device, and you can"
    say 'revert any time with: SwitchAudioSource -s "<your speakers>"'
    ask_yes_no_setting ROUTE_OUTPUT "Switch system output to the Multi-Output Device now?"
fi

PENDING_KEYS=""
NEW_KEY_VALUE=""
if interactive; then
    for k in $ALL_KEYS; do
        in_list "$k" "$KEYS" || continue
        ask_key_value "$k"
        if [[ -n "$NEW_KEY_VALUE" ]]; then
            PENDING_KEYS="${PENDING_KEYS:+$PENDING_KEYS }$k"
            # Written here, not batched later: each key must keep its own value.
            if [[ $DRY_RUN -eq 1 ]]; then
                step "would write $k to .env"
            else
                env_set "$k" "$NEW_KEY_VALUE"
                ok "wrote $k to .env (chmod 600)"
            fi
        fi
    done
else
    step "not interactive: leaving API keys in .env untouched"
fi

# Ask about the weights only when a selected capability ships some, so selecting
# `ui` alone does not promise a 2.4 GB download that never happens.
PREFETCH_EXTRAS=""
for e in $ALL_EXTRAS; do
    in_list "$e" "$EXTRAS" || continue
    prefetchable "$e" || continue
    PREFETCH_EXTRAS="${PREFETCH_EXTRAS:+$PREFETCH_EXTRAS }$e"
done
if [[ -n "$PREFETCH_EXTRAS" ]]; then
    ask_yes_no_setting PREFETCH "Download model weights now (~2.4 GB) instead of on first use?"
else
    # True whether nothing at all was selected or only a weightless `ui` was.
    skip "no selected capability ships model weights, nothing to pre-download"
    PREFETCH="no"
fi
ask_yes_no_setting RUN_TESTS "Run the test suite (uv run pytest -q) when done?"

# ============================================================== 2. plan

WANT_FORMULAS="git uv ffmpeg"
if yes_no "$LIVE_SETUP"; then WANT_FORMULAS="$WANT_FORMULAS blackhole-2ch switchaudio-osx"; fi

MISSING_FORMULAS=""
for f in $WANT_FORMULAS; do
    brew_has "$f" || MISSING_FORMULAS="${MISSING_FORMULAS:+$MISSING_FORMULAS }$f"
done

ADD_EXTRAS=""
DROP_EXTRAS=""
for e in $ALL_EXTRAS; do
    if in_list "$e" "$EXTRAS"; then
        extra_active "$e" || ADD_EXTRAS="${ADD_EXTRAS:+$ADD_EXTRAS }$e"
    elif extra_active "$e"; then
        DROP_EXTRAS="${DROP_EXTRAS:+$DROP_EXTRAS }$e"
    fi
done

{
    printf '\n'
    say "Plan"
    if [[ $ARM64 -eq 1 ]]; then
        printf '    detected                : Apple Silicon (arm64)%s\n' \
            "$( [[ $TRANSLATED -eq 1 ]] && printf ', via a translated shell' )"
        printf '    local-stt backend       : mlx-whisper%s\n' \
            "$( (( MACOS_MAJOR < 14 )) && printf '  WARNING: needs macOS 14+, use --stt faster-whisper' )"
    else
        printf '    detected                : Intel (x86_64), macOS %s\n' "$MACOS_VERSION"
        printf '    local-stt backend       : faster-whisper (MLX has no Intel build)\n'
    fi
    if [[ -n "$MISSING_FORMULAS" ]]; then
        printf '    brew formulas to install : %s\n' "$MISSING_FORMULAS"
    else
        printf '    brew formulas            : all present (%s)\n' "$WANT_FORMULAS"
    fi
    printf '    extras enabled           : %s\n' "${EXTRAS:-none}"
    [[ -n "$ADD_EXTRAS" ]]  && printf '    extras to install        : %s\n' "$ADD_EXTRAS"
    [[ -n "$DROP_EXTRAS" ]] && printf '    extras to uninstall      : %s\n' "$DROP_EXTRAS"
    printf '    keys tracked in .env     : %s\n' "${KEYS:-none}"
    [[ -n "$PENDING_KEYS" ]] && printf '    keys to write now        : %s\n' "$PENDING_KEYS"
    printf '    live capture devices     : %s\n' "$LIVE_SETUP"
    printf '    system output routed     : %s\n' "$ROUTE_OUTPUT"
    printf '    pre-download weights     : %s\n' "$PREFETCH"
    printf '    run test suite           : %s\n' "$RUN_TESTS"
    printf '\n'
} >&2

if [[ $DRY_RUN -eq 1 ]]; then
    say "dry run: nothing above was changed and no answers were saved."
    exit 0
fi

save_config
ok "saved answers to $CONFIG_FILE"

# ============================================================== 3. apply

# --- Xcode command-line tools (git + swiftc) --------------------------------
if ! xcode-select -p >/dev/null 2>&1; then
    say "installing Xcode command-line tools — follow the Apple prompt, then re-run me"
    xcode-select --install || true
    exit 0
fi

# --- Homebrew ---------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
    say "installing Homebrew…"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/null
    brew_env
fi
command -v brew >/dev/null 2>&1 || die "Homebrew install failed; install it manually, then re-run me"

# --- brew delta -------------------------------------------------------------
if [[ -n "$MISSING_FORMULAS" ]]; then
    say "installing: $MISSING_FORMULAS"
    # shellcheck disable=SC2086  # MISSING_FORMULAS is a space-separated list we built
    brew install $MISSING_FORMULAS
    ok "installed $MISSING_FORMULAS"
else
    skip "brew formulas already satisfied ($WANT_FORMULAS)"
fi
command -v ffmpeg >/dev/null 2>&1 || warn "ffmpeg is still not on PATH — omnilingual needs ffmpeg and ffprobe"

# --- extras convergence -----------------------------------------------------
# `uv sync --extra X` only ever adds. `--no-extra` is honoured only alongside
# --all-extras, so the convergent form is: add everything, subtract what you did
# not choose. The result then depends on your answers, not on run history.
NO_EXTRA_ARGS=()
for e in $ALL_EXTRAS; do
    in_list "$e" "$EXTRAS" || NO_EXTRA_ARGS+=("--no-extra" "$e")
done

if [[ -n "$ADD_EXTRAS$DROP_EXTRAS" ]]; then
    say "syncing dependencies (extras: ${EXTRAS:-none})…"
    in_repo uv sync --all-extras ${NO_EXTRA_ARGS[@]+"${NO_EXTRA_ARGS[@]}"}
    ok "extras converged"
else
    step "extras already match (${EXTRAS:-none}); skipping uv sync"
fi

# --- API keys ---------------------------------------------------------------
if [[ -n "$KEYS" ]]; then
    for k in $ALL_KEYS; do
        in_list "$k" "$KEYS" || continue
        if [[ -n "$(env_get "$k")" ]]; then
            ok "$k present in .env"
        else
            warn "$k is tracked but not set in .env — add it yourself"
        fi
    done
else
    skip "no API keys tracked"
fi

# --- live capture -----------------------------------------------------------
if yes_no "$LIVE_SETUP"; then
    if [[ ! -f "$SCRIPT_DIR/audio-devices.swift" ]]; then
        warn "scripts/audio-devices.swift not found — skipping live capture setup"
    else
        # Fresh HAL plugins need coreaudiod restarted before they enumerate.
        if sudo -n true 2>/dev/null; then
            step "restarting coreaudiod so BlackHole appears…"
            sudo killall coreaudiod 2>/dev/null || true
        elif [[ -t 0 ]]; then
            step "restarting coreaudiod so BlackHole appears (one sudo prompt)…"
            sudo killall coreaudiod 2>/dev/null || true
        else
            warn "no sudo access, so the coreaudiod restart is skipped."
            warn "if BlackHole does not appear below, reboot (or log out and back in) and re-run me."
        fi
        sleep 2

        HELPER_BIN="$(mktemp -t audio-devices)"
        TMP_PATHS+=("$HELPER_BIN")
        step "building the audio-device helper…"
        swiftc -o "$HELPER_BIN" "$SCRIPT_DIR/audio-devices.swift" \
            || die "swiftc failed on $SCRIPT_DIR/audio-devices.swift (is the Xcode command-line tools installed?)"

        step "waiting for BlackHole 2ch to register…"
        appeared=0
        for _ in $(seq 1 30); do
            if grep -q "BlackHole" <<<"$("$HELPER_BIN" list 2>/dev/null || true)"; then
                appeared=1; break
            fi
            sleep 2
        done
        [[ $appeared -eq 1 ]] || die "BlackHole 2ch never appeared; reboot and re-run me"

        say "creating the Aggregate + Multi-Output devices (idempotent)…"
        "$HELPER_BIN" ensure || die "device creation failed"
        "$HELPER_BIN" check || die "devices missing after creation"
        "$HELPER_BIN" list
        ok "live capture devices ready"

        if yes_no "$ROUTE_OUTPUT"; then
            say "switching system output to the Multi-Output Device…"
            SwitchAudioSource -s "Multi-Output Device"
            step "current output: $(SwitchAudioSource -c)"
            step 'revert any time with: SwitchAudioSource -s "<your speakers>"'
        else
            step "leaving system output alone — switch it yourself before a meeting:"
            step '  SwitchAudioSource -s "Multi-Output Device"'
        fi
    fi
else
    skip "live capture not configured (re-run and answer yes to enable it)"
fi

# --- model prefetch ---------------------------------------------------------
if yes_no "$PREFETCH"; then
    for e in $ALL_EXTRAS; do
        in_list "$e" "$EXTRAS" || continue
        prefetchable "$e" || continue
        label="$(prefetch_label "$e")"
        step "pre-downloading $label…"
        if prefetch_models "$e"; then
            ok "$label ready"
        else
            warn "could not pre-download the $e models; they will fetch on first use instead"
        fi
    done
else
    skip "model weights will download on first use"
fi

# --- verify -----------------------------------------------------------------
if yes_no "$RUN_TESTS"; then
    say "running the test suite…"
    if in_repo uv run pytest -q; then
        ok "test suite passed"
    else
        warn "test suite failed — see the output above"
    fi
else
    skip "test suite not run"
fi

if yes_no "$LIVE_SETUP"; then
    say "probing the capture chain — speak NOW when the verdict appears…"
    in_repo uv run omnilingual live --check-audio || true
fi

# --- handoff ----------------------------------------------------------------
{
    printf '\n'
    say "Setup complete. Re-run this script any time to change your answers."
    printf '\n'
    printf '    cd %s\n' "$REPO_ROOT"
    printf '    uv run omnilingual live --ask              # backends, models, speakers, output\n'
    printf '    uv run omnilingual transcribe <recording> --ask\n'
    if [[ -n "$KEYS" ]]; then
        printf '    uv run --env-file .env omnilingual transcribe <recording> --out notes.md\n'
    fi
    if yes_no "$LIVE_SETUP"; then
        printf '    uv run omnilingual live --out standup.md   # Ctrl+C stops; the file stays valid\n'
    fi
    printf '\n'
} >&2