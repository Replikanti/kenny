#!/bin/sh
# tools/netem-bench.sh — simulated-WAN harness (issues #5, #6).
#
# Emulates a WAN round-trip on loopback inside an UNPRIVILEGED network namespace
# (`unshare -rn`) with `tc netem`, then runs the netns-gated dispatch bench so the
# MANIFESTO §4.4 per-layer RTT barrier becomes measurable without root, a second
# box, or a new transport. netem on `lo` delays the loopback TCP between the
# spine's main thread and the node's background thread (tests/dispatch.rs), so the
# existing thread harness runs UNCHANGED under emulated RTT.
#
# Honest labelling: this is a SIMULATED WAN on a single host, NOT a real
# second-box LAN — the real-LAN validation stays issue #4 and the real multi-node
# party stays issue #6.
#
# Usage:
#   tools/netem-bench.sh [--rtt MS] [--loss PCT] [--jitter MS] [--loss-hol] [--hedge]
#   tools/netem-bench.sh --nodes N --placement
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
# Tail-latency hedge (model-free): the hedge OFF vs ON at a fixed loss (1% by
# default) — the ADR-0010 hedge-rate-vs-p99 number.
#   tools/netem-bench.sh --rtt 30 --hedge
# Multi-node PLACEMENT sim (M4, issue #6): N `kenny` nodes bound to distinct
# loopback IPs (127.0.0.2 ..), each behind its OWN `tc netem delay+rate` band via
# a per-destination `prio` filter, so a `PlacedDispatch` fans routed experts
# across HETEROGENEOUS shaped uplinks and the ADR-0009 per-node step p99 spread
# becomes measurable (a single shared `lo` qdisc cannot produce it). Fixture arm
# is model-free; set KENNY_MODEL_DIR for the real Qwen3-30B-A3B anchor.
#   tools/netem-bench.sh --nodes 3 --placement
#   KENNY_MODEL_DIR=<model_dir> tools/netem-bench.sh --nodes 3 --placement
# The per-node shaping profile (delay ms / rate Mbit) is the fixed heterogeneous
# class list below (node 0 = fat-and-near, ascending delay + descending rate); the
# spine->node egress is shaped, the node->spine return is left unshaped (every
# return packet has dst 127.0.0.1), so the measured per-node RTT ~= the one-way delay.
#
# If unprivileged netns is unavailable (e.g. CI), prints a skip and exits 0 — a
# plain `cargo test` never touches netem (the Rust arms gate on KENNY_NETEM_RTT_MS
# / KENNY_NETEM_NODES).
set -eu

rtt=30
loss=
jitter=
nodes=3
placement=
loss_hol=
hedge=

while [ $# -gt 0 ]; do
  case "$1" in
    --rtt) rtt=$2; shift 2 ;;
    --loss) loss=$2; shift 2 ;;
    --jitter) jitter=$2; shift 2 ;;
    --nodes) nodes=$2; shift 2 ;;
    --placement) placement=1; shift ;;
    --loss-hol) loss_hol=1; shift ;;
    --hedge) hedge=1; shift ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
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

# Fixed heterogeneous shaping classes (delay ms : rate Mbit), one per node index,
# ascending delay + descending rate — node 0 is the fat-and-near uplink the
# placement engine should hand the hot experts, node N-1 the thin far one. Extend
# the list to raise the --nodes ceiling.
node_classes="20:1000 60:100 100:50 150:20 250:10"

# run_placement N: bind node j to 127.0.0.(j+2) behind its own `tc netem` band
# selected by a per-destination `prio` u32 filter, export the node table as
# KENNY_NETEM_NODES, and run the netem_placement measurement inside the namespace.
# A dedicated pfifo band (1:1) carries everything unmatched (incl. the node->spine
# return traffic to 127.0.0.1) unshaped, so a node's measured RTT ~= its one-way
# delay. Real-model anchor when KENNY_MODEL_DIR is set (fixture arm otherwise).
run_placement() {
  _n=$1
  _bands=$((_n + 1))
  # Build the KENNY_NETEM_NODES table (ip:delay:rate) and the tc spec list.
  _nodes=""
  _specs=""
  _j=0
  for _class in $node_classes; do
    [ "$_j" -ge "$_n" ] && break
    _delay=$(echo "$_class" | cut -d: -f1)
    _rate=$(echo "$_class" | cut -d: -f2)
    _ip="127.0.0.$((_j + 2))"
    _nodes="${_nodes:+$_nodes,}${_ip}:${_delay}:${_rate}"
    _specs="${_specs:+$_specs }${_ip}:${_delay}:${_rate}"
    _j=$((_j + 1))
  done
  if [ "$_j" -lt "$_n" ]; then
    echo "netem-bench: --nodes $_n exceeds the $_j shaping classes available" >&2
    exit 2
  fi
  echo "=== netem placement: $_n shaped nodes [$_nodes] ==="
  KENNY_NETEM_NODES="$_nodes" \
  NETEM_BANDS="$_bands" \
  NETEM_SPECS="$_specs" \
  unshare -rn sh -c '
    set -e
    ip link set lo up
    ip addr add 127.0.0.1/8 dev lo 2>/dev/null || true
    tc qdisc add dev lo root handle 1: prio bands "$NETEM_BANDS" \
      priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    tc qdisc add dev lo parent 1:1 handle 11: pfifo
    b=2
    for spec in $NETEM_SPECS; do
      ip=$(echo "$spec" | cut -d: -f1)
      delay=$(echo "$spec" | cut -d: -f2)
      rate=$(echo "$spec" | cut -d: -f3)
      tc qdisc add dev lo parent 1:$b handle ${b}0: netem delay ${delay}ms rate ${rate}mbit
      tc filter add dev lo protocol ip parent 1: prio 1 u32 match ip dst ${ip}/32 flowid 1:$b
      b=$((b + 1))
    done
    exec cargo test --release --test dispatch netem_placement -- --nocapture --test-threads=1
  '
}

if [ -n "$placement" ]; then
  # Multi-node placement sim (M4, #6): heterogeneous per-node shaped uplinks.
  run_placement "$nodes"
elif [ -n "$hedge" ]; then
  # Tail-latency hedge: one netem qdisc + netem_hedge run at a fixed loss (1% by
  # default) — the test contrasts the hedge OFF vs ON at a fixed B.
  run_one "$rtt" netem_hedge "${loss:-1}"
elif [ -n "$loss_hol" ]; then
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
