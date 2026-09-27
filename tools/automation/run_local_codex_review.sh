#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="${GOOSEOMNI_REMOTE_ROOT:-/public/home/xty/workdir/omni_goose}"
RUN_ROOT=""
KIND=""
MODEL="${GOOSEOMNI_CODEX_MODEL:-gpt-5.6-sol}"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_ROOT="$2"; shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ROOT" || ! "$KIND" =~ ^(trajectory|tom)$ ]]; then
  echo "usage: $0 --run RUN_ROOT --kind trajectory|tom [--model MODEL] [--force]" >&2
  exit 2
fi
if [[ ! "$RUN_ROOT" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "unsafe run path: $RUN_ROOT" >&2
  exit 2
fi
if [[ "$RUN_ROOT" == /* || "$RUN_ROOT" == *".."* ]]; then
  echo "run path must be repository-relative without '..': $RUN_ROOT" >&2
  exit 2
fi

BUNDLE="$RUN_ROOT/local_codex/automation/$KIND"
hpc run "cd \"$REMOTE_ROOT\" && .venv/bin/python -m gooseomni data review --run \"$RUN_ROOT\" --kind \"$KIND\" --mode prepare"

LOCAL_BUNDLE="$PROJECT_ROOT/$BUNDLE"
PROMPT="$LOCAL_BUNDLE/prompt.txt"
SCHEMA="$LOCAL_BUNDLE/response_schema.json"
RESPONSE="$LOCAL_BUNDLE/response.json"

# FUSE-T can expose files shortly after the remote prepare command returns.
for _ in {1..30}; do
  if [[ -s "$PROMPT" && -s "$SCHEMA" && -f "$LOCAL_BUNDLE/image_paths.txt" ]]; then
    break
  fi
  sleep 1
done
if [[ ! -s "$PROMPT" || ! -s "$SCHEMA" || ! -f "$LOCAL_BUNDLE/image_paths.txt" ]]; then
  echo "local review bundle is not visible through FUSE-T: $LOCAL_BUNDLE" >&2
  exit 1
fi

IMAGE_ARGS=()
while IFS= read -r remote_path; do
  [[ -z "$remote_path" ]] && continue
  local_path="${remote_path/#$REMOTE_ROOT/$PROJECT_ROOT}"
  if [[ ! -f "$local_path" ]]; then
    echo "review image is not visible through FUSE-T: $local_path" >&2
    exit 1
  fi
  IMAGE_ARGS+=(--image "$local_path")
done < "$LOCAL_BUNDLE/image_paths.txt"

CONFIG_ARGS=()
CODEX_CONFIG="${CODEX_HOME:-$HOME/.codex}/config.toml"
if [[ -f "$CODEX_CONFIG" ]]; then
  PROVIDER="$(awk -F= '/^model_provider[[:space:]]*=/{gsub(/[ "\047]/, "", $2); print $2; exit}' "$CODEX_CONFIG")"
  BASE_URL="$(awk -v section="[model_providers.$PROVIDER]" '
    $0 == section { found=1; next }
    /^\[/ { found=0 }
    found && /^base_url[[:space:]]*=/ {
      sub(/^[^=]*=[[:space:]]*/, ""); gsub(/["\047]/, ""); print; exit
    }
  ' "$CODEX_CONFIG")"
  if [[ -n "$PROVIDER" && -n "$BASE_URL" ]]; then
    CONFIG_ARGS+=(
      --ignore-user-config
      -c "model_provider=\"$PROVIDER\""
      -c "model_providers.$PROVIDER={name=\"$PROVIDER\",base_url=\"$BASE_URL\",wire_api=\"responses\",requires_openai_auth=true}"
    )
  fi
fi

codex exec \
  "${CONFIG_ARGS[@]}" \
  --model "$MODEL" \
  --ephemeral \
  --sandbox read-only \
  --skip-git-repo-check \
  --cd "$PROJECT_ROOT" \
  --output-schema "$SCHEMA" \
  --output-last-message "$RESPONSE" \
  "${IMAGE_ARGS[@]}" \
  - < "$PROMPT"

APPLY_ARGS=""
if [[ "$FORCE" == "1" ]]; then
  APPLY_ARGS="--force"
fi
hpc run "cd \"$REMOTE_ROOT\" && .venv/bin/python -m gooseomni data review --run \"$RUN_ROOT\" --kind \"$KIND\" --mode apply --response \"$BUNDLE/response.json\" $APPLY_ARGS"
