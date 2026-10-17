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
#   --extras <list>    Comma-separated extras to enable, from:
#                      local-stt (offline speech-to-text), indic-stt (Indic-
#                      Transcribe-core: Apple Silicon, gated model), diarize
#                      (speaker labels), local-mt (offline translation), ui (the
#                      local app). Overrides the saved answers for this run.
#   --keys <list>      Comma-separated API keys to prompt for and store in .env,
#                      from: SARVAM_API_KEY, GROQ_API_KEY, GEMINI_API_KEY.
#   --live-setup <yn>  Create the BlackHole capture devices. Default: no.
#   --route-output <yn> Switch system output to the Multi-Output Device. Only
#                      meaningful with --live-setup yes; forced to no without it.
#   --prefetch <yn>    Pre-download model weights. Default: yes.
#   --run-tests <yn>   Run the test suite at the end. Default: yes.
#   Each of these overrides the saved answers for this run and the resolved
#   value is written back, so re-running converges instead of re-asking.
#   --check            Host due-diligence only: report whether this machine can run
#                      the selected extras (OS, CPU, RAM, disk, network, Hugging
#                      Face access), change nothing, exit 1 if anything blocks.
#   --skip-checks      Run the host check but do not stop on blocking problems.
#   -h, --help         Show this help.
#
# Before changing anything the script checks the host (macOS version, Apple
# Silicon vs Intel, Rosetta, RAM, free disk against what the selected extras
# need, network reachability, Hugging Face licence/token for gated models) and
# stops with a reason if the selection cannot work here. Pick what suits the
# machine with --extras; the prompt marks which extras this Mac can run.
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
# Explicit flags for this run. Empty means "not given", which is what lets the
# saved config win; *_SET distinguishes "given as empty" from "not given".
# These six *_SET lines must stay ABOVE the parse loop below — declared after it,
# they would reset what the parser just set and every flag would be discarded.
FLAG_EXTRAS=""
FLAG_KEYS=""
FLAG_LIVE_SETUP=""
FLAG_ROUTE_OUTPUT=""
FLAG_PREFETCH=""
FLAG_RUN_TESTS=""
EXTRAS_SET=0
KEYS_SET=0
LIVE_SETUP_SET=0
ROUTE_OUTPUT_SET=0
PREFETCH_SET=0
RUN_TESTS_SET=0
ASSUME_YES=0
DRY_RUN=0
RESET=0
CHECK_ONLY=0
SKIP_CHECKS=0
# Test/CI escape hatch: lets a suite run the script without the answer depending on
# the free disk or network of whoever runs it.
case "${OMNILINGUAL_SETUP_SKIP_CHECKS:-}" in 1|yes|true) SKIP_CHECKS=1 ;; esac

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed -e 's/^# \{0,1\}//' -e '$d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)   [[ $# -ge 2 ]] || { echo "--repo needs a value" >&2; exit 2; }; FLAG_REPO_URL="$2"; REPO_URL_SET=1; shift 2 ;;
        --branch) [[ $# -ge 2 ]] || { echo "--branch needs a value" >&2; exit 2; }; FLAG_BRANCH="$2"; BRANCH_SET=1; shift 2 ;;
        --extras)       [[ $# -ge 2 ]] || { echo "--extras needs a value" >&2; exit 2; }; FLAG_EXTRAS="$2"; EXTRAS_SET=1; shift 2 ;;
        --keys)         [[ $# -ge 2 ]] || { echo "--keys needs a value" >&2; exit 2; }; FLAG_KEYS="$2"; KEYS_SET=1; shift 2 ;;
        --live-setup)   [[ $# -ge 2 ]] || { echo "--live-setup needs a value" >&2; exit 2; }; FLAG_LIVE_SETUP="$2"; LIVE_SETUP_SET=1; shift 2 ;;
        --route-output) [[ $# -ge 2 ]] || { echo "--route-output needs a value" >&2; exit 2; }; FLAG_ROUTE_OUTPUT="$2"; ROUTE_OUTPUT_SET=1; shift 2 ;;
        --prefetch)     [[ $# -ge 2 ]] || { echo "--prefetch needs a value" >&2; exit 2; }; FLAG_PREFETCH="$2"; PREFETCH_SET=1; shift 2 ;;
        --run-tests)    [[ $# -ge 2 ]] || { echo "--run-tests needs a value" >&2; exit 2; }; FLAG_RUN_TESTS="$2"; RUN_TESTS_SET=1; shift 2 ;;
        -y|--yes)   ASSUME_YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --reset)   RESET=1; shift ;;
        --check)   CHECK_ONLY=1; DRY_RUN=1; ASSUME_YES=1; shift ;;
        --skip-checks) SKIP_CHECKS=1; shift ;;
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

RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
[[ -n "$CPU_BRAND" ]] || CPU_BRAND="$ARCH"

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

ALL_EXTRAS="local-stt indic-stt diarize local-mt ui"
ALL_KEYS="SARVAM_API_KEY GROQ_API_KEY GEMINI_API_KEY"

# The id the setup script probes for Hugging Face access; must equal
# config.DEFAULT_STT_MODELS["indic-transcribe"] (a test pins this).
INDIC_STT_MODEL="bodhan-ai/indic-transcribe-core"

# extra_blocker <extra> -> prints why this host cannot run it; prints nothing when
# it can. One predicate shared by the prompt (to mark choices) and the host check
# (to refuse them), so what is offered and what is allowed cannot drift apart.
extra_blocker() {
    case "$1" in
        indic-stt)
            if [[ $ARM64 -ne 1 ]]; then
                printf 'needs Apple Silicon: PyTorch publishes no wheels for Intel Macs'
            elif [[ $TRANSLATED -eq 1 ]]; then
                printf 'this shell runs under Rosetta, so uv would fetch an x86_64 Python that has no PyTorch wheel; re-run from a native Terminal'
            elif (( MACOS_MAJOR < 14 )); then
                printf 'PyTorch wheels need macOS 14+ (this Mac is macOS %s)' "$MACOS_VERSION"
            elif (( RAM_GB > 0 && RAM_GB < 8 )); then
                printf 'the 1.2B model needs at least 8 GB of RAM (this Mac has %s GB)' "$RAM_GB"
            fi
            ;;
    esac
}

# Rough sizes in tenths of a GB, so the arithmetic stays integer in bash 3.2.
# install = venv + wheels; weights = what the model download adds.
install_tenths() {
    case "$1" in
        local-stt) echo 5 ;; indic-stt) echo 25 ;; diarize) echo 2 ;; local-mt) echo 3 ;; ui) echo 3 ;; *) echo 0 ;;
    esac
}
weights_tenths() {
    case "$1" in
        local-stt) if [[ $ARM64 -eq 1 ]]; then echo 15; else echo 5; fi ;;
        indic-stt) echo 49 ;; diarize) echo 1 ;; local-mt) echo 9 ;; *) echo 0 ;;
    esac
}
tenths_str() { printf '%s.%s' $(( $1 / 10 )) $(( $1 % 10 )); }

EXTRAS="local-stt,diarize,ui"
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
if [[ $EXTRAS_SET -eq 1 ]]; then EXTRAS="$FLAG_EXTRAS"; fi
if [[ $KEYS_SET -eq 1 ]]; then KEYS="$FLAG_KEYS"; fi
if [[ $LIVE_SETUP_SET -eq 1 ]]; then LIVE_SETUP="$FLAG_LIVE_SETUP"; fi
if [[ $ROUTE_OUTPUT_SET -eq 1 ]]; then ROUTE_OUTPUT="$FLAG_ROUTE_OUTPUT"; fi
if [[ $PREFETCH_SET -eq 1 ]]; then PREFETCH="$FLAG_PREFETCH"; fi
if [[ $RUN_TESTS_SET -eq 1 ]]; then RUN_TESTS="$FLAG_RUN_TESTS"; fi

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
                "indic-stt  offline Indic speech-to-text, best accuracy (Indic-Transcribe-core, ~4.9 GB, gated)" \
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
            local e why
            for e in $ALL_EXTRAS; do
                why="$(extra_blocker "$e")"
                [[ -n "$why" ]] && printf '    %s is NOT available here: %s\n' "$e" "$why"
            done
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
        indic-stt) pkg=torch ;;
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
        local-stt|indic-stt|diarize|local-mt) return 0 ;;
        # ui is not here and does not need an arm: it ships wheels, not weights.
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
        indic-stt)
            printf 'Indic-Transcribe-core (~4.9 GB, gated)'
            ;;
        # Failing is the honest answer: ui ships wheels, not weights, so there is
        # no label for it. The caller skips on failure, so this arm is what makes
        # the function correct by construction rather than only because the
        # prefetchable gate happens to sit between the check and the call.
        ui) return 1 ;;
        *) return 1 ;;
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
        indic-stt)
            # Gated: needs `hf auth login` and the licence accepted on the model page.
            # snapshot_download resumes, so a re-run continues a half-finished fetch.
            in_repo uv run python - <<'PY'
from huggingface_hub import snapshot_download
from omnilingual.config import DEFAULT_STT_MODELS
print(snapshot_download(DEFAULT_STT_MODELS["indic-transcribe"]))
PY
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


# ------------------------------------------------------------- host due diligence

PF_FAILS=0
PF_WARNS=0
pf_ok()   { printf '\033[32m  ok\033[0m %s\n' "$*" >&2; }
pf_info() { printf '\033[2m  ..\033[0m %s\n' "$*" >&2; }
pf_warn() { printf '\033[33m  !!\033[0m %s\n' "$*" >&2; PF_WARNS=$(( PF_WARNS + 1 )); }
pf_fail() { printf '\033[31m  xx\033[0m %s\n' "$*" >&2; PF_FAILS=$(( PF_FAILS + 1 )); }

HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_DIR="${HF_HUB_CACHE:-$HF_HOME_DIR/hub}"

# free_tenths <path> -> free space of the volume holding path (nearest existing
# ancestor), in tenths of a GB; empty when df cannot tell.
free_tenths() {
    local p="$1" kb
    while [[ ! -e "$p" && "$p" != "/" ]]; do p="$(dirname "$p")"; done
    kb="$(df -k "$p" 2>/dev/null | awk 'NR==2{print $4}')"
    [[ "$kb" =~ ^[0-9]+$ ]] || return 0
    printf '%s' $(( kb * 10 / 1048576 ))
}
# volume_id <path> -> the device df reports, so two paths on one disk are summed.
volume_id() {
    local p="$1"
    while [[ ! -e "$p" && "$p" != "/" ]]; do p="$(dirname "$p")"; done
    df -k "$p" 2>/dev/null | awk 'NR==2{print $1}'
}

# http_status <url> [token] -> status code, 000 when unreachable. The token goes to
# curl on stdin as a config line: in argv it would show up in `ps`.
http_status() {
    local url="$1" token="${2:-}"
    if [[ -n "$token" ]]; then
        printf 'header = "Authorization: Bearer %s"\n' "$token" \
            | curl -s -o /dev/null -w '%{http_code}' --max-time 8 -K - "$url" 2>/dev/null || true
    else
        curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$url" 2>/dev/null || true
    fi
}
reachable() { local c; c="$(http_status "$1")"; [[ -n "$c" && "$c" != "000" ]]; }

# hf_token -> the token huggingface_hub would use (env first, then the login file).
hf_token() {
    local t="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" f="${HF_TOKEN_PATH:-$HF_HOME_DIR/token}"
    if [[ -z "$t" && -f "$f" ]]; then t="$(tr -d '[:space:]' < "$f")"; fi
    printf '%s' "$t"
}

indic_weights_cached() {
    local d="$HF_HUB_DIR/models--${INDIC_STT_MODEL//\//--}/snapshots" f
    [[ -d "$d" ]] || return 1
    for f in "$d"/*/model.safetensors; do [[ -e "$f" ]] && return 0; done
    return 1
}

# preflight — read-only. Every check ends in ok / warning (!!) / blocker (xx); a blocker
# means the selected capabilities cannot work on this machine as asked.
preflight() {
    PF_FAILS=0; PF_WARNS=0
    local e why code tok need_repo free_repo free_home net_ok h hosts=""

    say "Host check"
    pf_info "macOS $MACOS_VERSION, $CPU_BRAND, $NCPU cores, ${RAM_GB} GB RAM"
    if [[ $ARM64 -eq 1 ]]; then
        pf_ok "Apple Silicon: MLX and Metal (MPS) acceleration are available"
        [[ $TRANSLATED -eq 1 ]] && pf_warn "this shell is translated by Rosetta 2; re-run from a native Terminal for arm64 packages"
    else
        pf_info "Intel Mac: local-stt uses faster-whisper on CPU; indic-stt is unavailable"
    fi
    command -v xcode-select >/dev/null 2>&1 && xcode-select -p >/dev/null 2>&1 \
        && pf_ok "Xcode command-line tools present" \
        || pf_warn "Xcode command-line tools missing: the apply step will trigger Apple's installer and ask you to re-run"

    # Can this machine run what was asked for?
    for e in $ALL_EXTRAS; do
        in_list "$e" "$EXTRAS" || continue
        why="$(extra_blocker "$e")"
        if [[ -n "$why" ]]; then
            pf_fail "$e cannot run here: $why. Re-run with --extras listing the others (without $e)"
        fi
    done
    if in_list local-stt "$EXTRAS" && [[ $ARM64 -eq 1 ]] && (( MACOS_MAJOR < 14 )); then
        pf_warn "local-stt: MLX needs macOS 14+ (this is $MACOS_VERSION); pass --stt faster-whisper to transcribe on CPU"
    fi
    if in_list indic-stt "$EXTRAS" && [[ -z "$(extra_blocker indic-stt)" ]]; then
        if (( RAM_GB < 16 )); then
            pf_warn "indic-stt: ${RAM_GB} GB RAM is tight for a 4.9 GB model plus the OS; expect swapping, close other apps"
        else
            pf_ok "indic-stt: ${RAM_GB} GB RAM is enough for the 1.2B model on the GPU (~12-28x realtime measured on an M4 Max)"
        fi
    fi
    if in_list local-mt "$EXTRAS" && (( RAM_GB > 0 && RAM_GB < 8 )); then
        pf_warn "local-mt: ${RAM_GB} GB RAM is low for IndicTrans2; translation may be slow"
    fi

    # Disk: what the apply step will add, against what is free where it lands.
    local repo_t=0 home_t=0
    [[ -n "$MISSING_FORMULAS" ]] && repo_t=$(( repo_t + 5 ))
    for e in $ADD_EXTRAS; do repo_t=$(( repo_t + $(install_tenths "$e") )); done
    if yes_no "$PREFETCH"; then
        for e in $EXTRAS_LIST; do
            prefetchable "$e" || continue
            if [[ "$e" == indic-stt ]] && indic_weights_cached; then continue; fi
            home_t=$(( home_t + $(weights_tenths "$e") ))
        done
    fi
    # +20 % margin: wheels are unpacked next to uv's cache copy.
    repo_t=$(( repo_t * 12 / 10 )); home_t=$(( home_t * 12 / 10 ))
    free_repo="$(free_tenths "$REPO_ROOT")"; free_home="$(free_tenths "$HF_HOME_DIR")"
    if [[ -z "$free_repo" || -z "$free_home" ]]; then
        pf_warn "could not read free disk space; skipping the disk check"
    elif [[ "$(volume_id "$REPO_ROOT")" == "$(volume_id "$HF_HOME_DIR")" ]]; then
        need_repo=$(( repo_t + home_t ))
        if (( free_repo < need_repo )); then
            pf_fail "disk: need ~$(tenths_str "$need_repo") GB free, only $(tenths_str "$free_repo") GB available"
        elif (( free_repo < need_repo + 50 )); then
            pf_warn "disk: ~$(tenths_str "$need_repo") GB needed, $(tenths_str "$free_repo") GB free (under 5 GB headroom)"
        else
            pf_ok "disk: ~$(tenths_str "$need_repo") GB needed, $(tenths_str "$free_repo") GB free"
        fi
    else
        if (( free_repo < repo_t )); then pf_fail "disk (repo volume): need ~$(tenths_str "$repo_t") GB, only $(tenths_str "$free_repo") GB free"
        else pf_ok "disk (repo volume): ~$(tenths_str "$repo_t") GB needed, $(tenths_str "$free_repo") GB free"; fi
        if (( free_home < home_t )); then pf_fail "disk (model cache $HF_HOME_DIR): need ~$(tenths_str "$home_t") GB, only $(tenths_str "$free_home") GB free"
        else pf_ok "disk (model cache): ~$(tenths_str "$home_t") GB needed, $(tenths_str "$free_home") GB free"; fi
    fi

    # Network: only the hosts this run will actually talk to.
    case "${UV_OFFLINE:-}" in
        1|true|yes) pf_info "UV_OFFLINE is set: network checks skipped" ; net_ok=0 ;;
        *) net_ok=1 ;;
    esac
    if [[ $net_ok -eq 1 ]]; then
        [[ -n "$ADD_EXTRAS" ]] && hosts="$hosts https://pypi.org/simple/"
        [[ -n "$MISSING_FORMULAS" ]] && hosts="$hosts https://formulae.brew.sh/"
        if yes_no "$PREFETCH" || in_list indic-stt "$ADD_EXTRAS"; then
            for e in $EXTRAS_LIST; do prefetchable "$e" && { hosts="$hosts https://huggingface.co/"; break; }; done
        fi
        for h in $hosts; do
            case " $SEEN_HOSTS " in *" $h "*) continue ;; esac
            SEEN_HOSTS="$SEEN_HOSTS $h"
            if reachable "$h"; then pf_ok "network: $h reachable"; else pf_fail "network: cannot reach $h (offline, proxy or firewall?)"; fi
        done
        [[ -z "$hosts" ]] && pf_info "nothing to download: network checks not needed"
    fi

    # Hugging Face: the gated model fails late and obscurely, so test it now.
    if in_list indic-stt "$EXTRAS" && [[ -z "$(extra_blocker indic-stt)" ]]; then
        if indic_weights_cached; then
            pf_ok "indic-stt: model already in the Hugging Face cache, no access needed"
        else
            tok="$(hf_token)"
            if [[ -z "$tok" ]]; then
                pf_warn "indic-stt: no Hugging Face token found. Accept the licence at https://huggingface.co/$INDIC_STT_MODEL then log in (the apply step offers 'hf auth login')"
            elif [[ $net_ok -eq 1 ]]; then
                code="$(http_status "https://huggingface.co/$INDIC_STT_MODEL/resolve/main/config.json" "$tok")"
                case "$code" in
                    200|302) pf_ok "indic-stt: Hugging Face token has access to $INDIC_STT_MODEL" ;;
                    401)     pf_fail "indic-stt: Hugging Face rejected the token (401); run 'hf auth login' with a valid read token" ;;
                    403)     pf_fail "indic-stt: licence not accepted for this token (403); accept it at https://huggingface.co/$INDIC_STT_MODEL and, for a fine-grained token, tick 'public gated repos'" ;;
                    000)     pf_warn "indic-stt: could not reach huggingface.co to verify access" ;;
                    *)       pf_warn "indic-stt: unexpected HTTP $code while verifying Hugging Face access" ;;
                esac
            else
                pf_info "indic-stt: Hugging Face token found (access not verified offline)"
            fi
        fi
    fi

    if (( PF_FAILS > 0 )); then
        printf '\n' >&2
        if [[ $SKIP_CHECKS -eq 1 ]]; then
            warn "$PF_FAILS blocking problem(s) above ignored (--skip-checks)."
        else
            warn "$PF_FAILS blocking problem(s) found; $PF_WARNS warning(s)."
        fi
    else
        pf_ok "host check passed ($PF_WARNS warning(s))"
    fi
    printf '\n' >&2
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
    PREFETCH_T=0
    for e in $PREFETCH_EXTRAS; do PREFETCH_T=$(( PREFETCH_T + $(weights_tenths "$e") )); done
    ask_yes_no_setting PREFETCH "Download model weights now (~$(tenths_str "$PREFETCH_T") GB) instead of on first use?"
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

SEEN_HOSTS=""
EXTRAS_LIST="${EXTRAS//,/ }"
preflight

if [[ $CHECK_ONLY -eq 1 ]]; then
    exit $(( PF_FAILS > 0 ? 1 : 0 ))
fi
if (( PF_FAILS > 0 )) && [[ $SKIP_CHECKS -eq 0 && $DRY_RUN -eq 0 ]]; then
    die "this machine cannot satisfy the selected capabilities; fix the items marked xx, or re-run with --extras to choose differently (--skip-checks overrides)"
fi

{
    printf '\n'
    say "Plan"
    if [[ $ARM64 -eq 1 ]]; then
        printf '    detected                : Apple Silicon (arm64)%s\n' \
            "$( [[ $TRANSLATED -eq 1 ]] && printf ', via a translated shell' )"
        in_list indic-stt "$EXTRAS" && printf '    indic-stt backend       : Indic-Transcribe-core on Metal (MPS)\n'
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

# --- Hugging Face login (gated models) --------------------------------------
if in_list indic-stt "$EXTRAS" && ! indic_weights_cached && [[ -z "$(hf_token)" ]]; then
    step "indic-stt downloads a gated model: accept the licence at https://huggingface.co/$INDIC_STT_MODEL, then log in"
    if interactive && confirm "    Run 'hf auth login' now?" y; then
        in_repo uv run hf auth login || warn "hf auth login did not finish; the model download below will be skipped until you log in"
    else
        warn "no Hugging Face login: run 'cd $REPO_ROOT && uv run hf auth login', then re-run me to fetch the model"
    fi
fi

# --- model prefetch ---------------------------------------------------------
if yes_no "$PREFETCH"; then
    for e in $ALL_EXTRAS; do
        in_list "$e" "$EXTRAS" || continue
        prefetchable "$e" || continue
        label="$(prefetch_label "$e")"
        # Braces, not "$label…": bash 3.2 reads the byte after $label as part of
        # the name, so the UTF-8 ellipsis becomes "label\xef" and `set -u` aborts
        # the whole script at the first prefetch. It has been failing this way
        # since the ellipsis landed, on every macOS that still ships bash 3.2.
        step "pre-downloading ${label}…"
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
    # The two lines below are the whole of the ui handoff, and both are
    # conditional: the EXTRAS default now carries ui, but load_config lets a
    # saved setup-mac.conf override it, so a user who ran this script before the
    # default changed sees no ui at all. Telling them the line to edit beats
    # editing their saved answers, which would silently answer a question they
    # were never asked.
    if in_list indic-stt "$EXTRAS"; then
        printf '    uv run omnilingual REC.m4a --stt indic-transcribe --langs hi-IN,en-IN --mt gemini   # free STT; --langs narrows language ID\n'
    fi
    if in_list ui "$EXTRAS"; then
        printf '    uv run omnilingual-ui                     # the windowed app\n'
    else
        say "ui is not installed: your saved EXTRAS= overrides the default that now includes it."
        say "add ui to EXTRAS= in $CONFIG_FILE (or re-run and answer yes), then re-run this script."
    fi
    printf '\n'
} >&2
