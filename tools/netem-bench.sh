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
#   tools/netem-bench.sh [--rtt MS] [--loss PCT] [--jitter MS] [--loss-hol]
#
# Fixture arm (model-free, ~10 min): amortization B-sweep at the given RTT.
#   tools/netem-bench.sh --rtt 30
# Real-model anchor (KENNY_MODEL_DIR set, ~5 min): B in {1,8}, run at 0 ms AND
# RTT ms in the SAME netns so BENCH reports Δt_step = t_step(RTT) − t_step(0),
# isolating the RTT term from any per-namespace overhead.
#   KENNY_MODEL_DIR=<model_dir> tools/netem-bench.sh --rtt 30
# Loss / head-of-line (HOL) matrix (model-free): the per-layer timeout OFF vs ON
# under a loss sweep L in {0, 0.5, 1, 2}%, one netem qdisc + test run per L.
#   tools/netem-bench.sh --rtt 30 --loss-hol
#
# If unprivileged netns is unavailable (e.g. CI), prints a skip and exits 0 — a
# plain `cargo test` never touches netem (the Rust arm is KENNY_NETEM_RTT_MS-gated).
set -eu

rtt=30
loss=
jitter=
loss_hol=

while [ $# -gt 0 ]; do
  case "$1" in
    --rtt) rtt=$2; shift 2 ;;
    --loss) loss=$2; shift 2 ;;
    --jitter) jitter=$2; shift 2 ;;
    --loss-hol) loss_hol=1; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
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

# run_one RTT_MS TEST LOSS_PCT: bring lo up, install a netem qdisc emulating an
# RTT_MS round trip (half each direction on lo egress — a loopback packet is
# delayed once per direction), export the gate + scenario, and run TEST inside the
# namespace. Loss/jitter attach only when RTT_MS > 0 (jitter needs a base delay;
# the 0 ms control measures the bare netns overhead so Δt_step isolates the RTT).
run_one() {
  _rtt=$1
  _test=$2
  _loss=$3
  _half=$((_rtt / 2))
  _netem="delay ${_half}ms"
  if [ "$_rtt" -gt 0 ]; then
    [ -n "$jitter" ] && _netem="$_netem ${jitter}ms"
    [ -n "$_loss" ] && [ "$_loss" != "0" ] && _netem="$_netem loss ${_loss}%"
  fi
  echo "=== netem: lo ${_netem} (RTT ${_rtt} ms, test ${_test}) ==="
  KENNY_NETEM_RTT_MS=$_rtt \
  KENNY_NETEM_LOSS_PCT=${_loss:-0} \
  KENNY_NETEM_JITTER_MS=${jitter:-0} \
  NETEM_QDISC="$_netem" \
  NETEM_TEST="$_test" \
  unshare -rn sh -c '
    set -e
    ip link set lo up
    tc qdisc add dev lo root netem $NETEM_QDISC
    exec cargo test --release --test dispatch $NETEM_TEST -- --nocapture --test-threads=1
  '
}

if [ -n "$loss_hol" ]; then
  # Loss / HOL matrix: one netem qdisc + netem_loss_hol run per loss value; the
  # test sweeps B in {16,64} and the per-layer timeout OFF/ON at each L.
  for L in 0 0.5 1 2; do
    run_one "$rtt" netem_loss_hol "$L"
  done
elif [ -n "${KENNY_MODEL_DIR:-}" ]; then
  # Real-model anchor: control (0 ms) then the RTT run, one netns each.
  run_one 0 netem_amortization "$loss"
  run_one "$rtt" netem_amortization "$loss"
else
  # Fixture arm: a single RTT run carries the amortization B-sweep (the M2
  # loopback line is the RTT≈0 contrast — BENCH.md).
  run_one "$rtt" netem_amortization "$loss"
fi
