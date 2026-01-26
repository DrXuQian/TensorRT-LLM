#!/usr/bin/env bash
set -u

BIN=""
NUM_TOKENS=3823
HIDDEN_SIZE=2048
INTER_SIZE=768
NUM_EXPERTS=128
EXPERTS_PER_TOKEN=8
OP="fc1"
WARMUP=10
ITERS=100
NCU=0
DEBUG=0
OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--bin <test_moe_fused_gemm>] [--op fc1|fc2] \\
  [--num_tokens N] [--hidden_size N] [--inter_size N] [--num_experts N] [--experts_per_token N] \\
  [--warmup N] [--iters N] [--ncu] [--debug] [--out FILE]

Notes:
  - This forces each config via --config=..., so the binary does not run its selector.
  - For --ncu runs, consider using ncu's --launch-skip/--launch-count to profile a single steady-state launch.

Example:
  $0 --op fc1 --num_tokens 3823 --hidden_size 2048 --inter_size 768 \\
     --num_experts 128 --experts_per_token 8 --warmup 10 --iters 200 --out results_fc1.txt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2;;
    --num_tokens) NUM_TOKENS="$2"; shift 2;;
    --hidden_size) HIDDEN_SIZE="$2"; shift 2;;
    --inter_size) INTER_SIZE="$2"; shift 2;;
    --num_experts) NUM_EXPERTS="$2"; shift 2;;
    --experts_per_token) EXPERTS_PER_TOKEN="$2"; shift 2;;
    --op) OP="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --ncu) NCU=1; shift 1;;
    --debug) DEBUG=1; shift 1;;
    --out) OUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$BIN" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  BIN="${SCRIPT_DIR}/../build/test_moe_fused_gemm"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN" >&2
  echo "Hint: build it with:" >&2
  echo "  cmake -S moe_fused_gemm_standalone -B moe_fused_gemm_standalone/build" >&2
  echo "  cmake --build moe_fused_gemm_standalone/build -j8" >&2
  exit 1
fi

if [[ "$OP" != "fc1" && "$OP" != "fc2" ]]; then
  echo "--op must be fc1 or fc2 (use separate runs for both)." >&2
  exit 1
fi

if [[ -n "$OUT" ]]; then
  : > "$OUT"
fi

configs=$("$BIN" --list_configs | sed -n 's/^ *[0-9]\\+: tile_m=\\([0-9]\\+\\) tile_n=\\([0-9]\\+\\) tile_k=\\([0-9]\\+\\) stages=\\([0-9]\\+\\)$/\\1,\\2,\\3,\\4/p')
if [[ -z "$configs" ]]; then
  echo "Failed to parse configs from: $BIN --list_configs" >&2
  exit 1
fi

fail=0
idx=0
while IFS= read -r cfg; do
  if [[ -z "$cfg" ]]; then
    continue
  fi

  cmd=("$BIN"
    "--num_tokens=$NUM_TOKENS" "--hidden_size=$HIDDEN_SIZE" "--inter_size=$INTER_SIZE"
    "--num_experts=$NUM_EXPERTS" "--experts_per_token=$EXPERTS_PER_TOKEN"
    "--op=$OP" "--warmup=$WARMUP" "--iters=$ITERS"
    "--config=$cfg")

  if [[ "$NCU" -eq 1 ]]; then
    cmd+=("--ncu")
  fi
  if [[ "$DEBUG" -eq 1 ]]; then
    cmd+=("--debug")
  fi

  echo "[${idx}] ${cmd[*]}"
  if [[ -n "$OUT" ]]; then
    {
      echo "[${idx}] ${cmd[*]}"
      "${cmd[@]}"
      echo
    } >> "$OUT" 2>&1 || fail=$((fail+1))
  else
    "${cmd[@]}" || fail=$((fail+1))
  fi

  idx=$((idx+1))
done <<< "$configs"

if [[ $fail -ne 0 ]]; then
  echo "Done with $fail failures." >&2
  exit 1
fi

echo "Done."

