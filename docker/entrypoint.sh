#!/bin/bash
set -euo pipefail

# The data folder (the image's volume): REMI_DATA_DIR, set by the Dockerfile
: "${REMI_DATA_DIR:=/home/remi/data}"

# TCMalloc reduces memory fragmentation under large model workloads
TCMALLOC="$(find /usr/lib -name 'libtcmalloc_minimal.so.4' -print -quit 2>/dev/null || true)"
if [[ -n "$TCMALLOC" ]]; then
    export LD_PRELOAD="$TCMALLOC"
fi

# Read COMMANDLINE_ARGS then unset it (the launcher would read it again). It is split as a
# shell splits words, as the launcher itself does (shlex.split): quotes keep a value with
# spaces together.
RAW_ARGS="${COMMANDLINE_ARGS:-}"
unset COMMANDLINE_ARGS

EXTRA_ARGS=()
if [[ -n "$RAW_ARGS" ]]; then
    # xargs refuses an unbalanced quote; stop rather than start without some of the options
    # (the value itself is not printed: it can hold a password)
    if ! xargs -r printf '%s\0' <<<"$RAW_ARGS" >/dev/null 2>&1; then
        echo "COMMANDLINE_ARGS cannot be split into arguments: check its quotes" >&2
        exit 2
    fi
    while IFS= read -r -d '' arg; do
        EXTRA_ARGS+=("$arg")
    done < <(xargs -r printf '%s\0' <<<"$RAW_ARGS")
fi
# (an empty array is "unbound" under set -u before bash 4.4: ${a[@]+"${a[@]}"} expands it safely)
ALL_ARGS=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} "$@")

# The folder the launcher will use: the last --data-dir given (argparse takes the last one,
# so one in the launch options wins over the default below), else REMI_DATA_DIR
DATA_DIR="$REMI_DATA_DIR"
for ((i = 0; i < ${#ALL_ARGS[@]}; i++)); do
    case "${ALL_ARGS[$i]}" in
        --data-dir=*) DATA_DIR="${ALL_ARGS[$i]#--data-dir=}" ;;
        --data-dir) if ((i + 1 < ${#ALL_ARGS[@]})); then DATA_DIR="${ALL_ARGS[$((i + 1))]}"; fi ;;
    esac
done

# Every piece of user data (config.json, styles.csv, user.css, remi-data/, models/,
# output/, extensions/) is in the data folder: one volume keeps it across containers.
mkdir -p "$DATA_DIR"/models/{Stable-diffusion,VAE,text_encoder,Lora,ControlNet,ESRGAN,embeddings} \
    "$DATA_DIR"/{output,extensions}

exec python /home/remi/app/launch.py \
    --listen \
    --data-dir "$REMI_DATA_DIR" \
    ${ALL_ARGS[@]+"${ALL_ARGS[@]}"}
