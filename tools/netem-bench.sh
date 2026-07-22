#!/bin/sh
# tools/netem-bench.sh — M3 simulated-WAN harness (issue #5).
#
# Emulates a WAN round-trip on loopback inside an UNPRIVILEGED network namespace
# (`unshare -rn`) with `tc netem`, then runs the netns-gated dispatch bench so the
# MANIFESTO §4.4 per-layer RTT barrier becomes measurable without root, a second
# box, or a new transport. netem on `lo` delays the loopback TCP between the
# spine's main thread and the node's background thread (tests/dispatch.rs), so the
# existing thread harness runs UNCHANGED under emulated RTT.
#
# Honest labelling: this is a SIMULATED WAN on a single host, NOT a real
# second-box LAN — the real-LAN validation stays issue #4.
#
# Usage:
#   tools/netem-bench.sh [--rtt MS] [--loss PCT] [--jitter MS]
#
# Fixture arm (model-free, ~10 min): amortization B-sweep at the given RTT.
#   tools/netem-bench.sh --rtt 30
# Real-model anchor (KENNY_MODEL_DIR set, ~5 min): B in {1,8}, run at 0 ms AND
# RTT ms in the SAME netns so BENCH reports Δt_step = t_step(RTT) − t_step(0),
# isolating the RTT term from any per-namespace overhead.
#   KENNY_MODEL_DIR=<model_dir> tools/netem-bench.sh --rtt 30
#
# If unprivileged netns is unavailable (e.g. CI), prints a skip and exits 0 — a
# plain `cargo test` never touches netem (the Rust arm is KENNY_NETEM_RTT_MS-gated).
set -eu

rtt=30
loss=
jitter=

while [ $# -gt 0 ]; do
  case "$1" in
    --rtt) rtt=$2; shift 2 ;;
    --loss) loss=$2; shift 2 ;;
    --jitter) jitter=$2; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "netem-bench: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# Clean skip when the host cannot create an unprivileged net namespace.
if ! unshare -rn true 2>/dev/null; then
  echo "netns unavailable — skipping M3 netem harness"
  exit 0
fi

cd "$(dirname "$0")/.."

# Pre-build the test binary OUTSIDE the namespace: compilation needs no network,
# and a build failure should surface without the netns/netem noise.
cargo test --release --test dispatch --no-run >/dev/null

# run_one RTT_MS: bring lo up, install a netem qdisc emulating an RTT_MS round
# trip (half each direction on lo egress — a loopback packet is delayed once per
# direction), export the gate + scenario, and run the bench inside the namespace.
# Loss/jitter attach only when RTT_MS > 0 (jitter needs a base delay; the 0 ms
# control measures the bare netns overhead so Δt_step isolates the RTT term).
run_one() {
  _rtt=$1
  _half=$((_rtt / 2))
  _netem="delay ${_half}ms"
  if [ "$_rtt" -gt 0 ]; then
    [ -n "$jitter" ] && _netem="$_netem ${jitter}ms"
    [ -n "$loss" ] && _netem="$_netem loss ${loss}%"
  fi
  echo "=== netem: lo ${_netem} (RTT ${_rtt} ms) ==="
  KENNY_NETEM_RTT_MS=$_rtt \
  KENNY_NETEM_LOSS_PCT=${loss:-0} \
  KENNY_NETEM_JITTER_MS=${jitter:-0} \
  NETEM_QDISC="$_netem" \
  unshare -rn sh -c '
    set -e
    ip link set lo up
    tc qdisc add dev lo root netem $NETEM_QDISC
    exec cargo test --release --test dispatch netem_amortization -- --nocapture --test-threads=1
  '
}

if [ -n "${KENNY_MODEL_DIR:-}" ]; then
  # Real-model anchor: control (0 ms) then the RTT run, one netns each.
  run_one 0
  run_one "$rtt"
else
  # Fixture arm: a single RTT run carries the amortization B-sweep (the M2
  # loopback line is the RTT≈0 contrast — BENCH.md).
  run_one "$rtt"
fi
