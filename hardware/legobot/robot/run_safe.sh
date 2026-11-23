#!/bin/bash
# Run a robot module and coast both motors when it ends for any reason, including kill -9 of
# the Python process (which skips Python's own cleanup). Usage, from ~/legobot:
#     robot/run_safe.sh robot.bench_log --seconds 5 --pwm 0.15
# SIGINT/SIGTERM sent to this wrapper are passed on to Python, which coasts itself; the wrapper
# then sends the coast command again. Killing the wrapper itself with -9 leaves no guard.
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-$HOME/legobot-venv/bin/python}"

coast() {
    stty -F /dev/serial0 115200 raw -echo 2>/dev/null
    printf 'port 0 ; coast ; port 1 ; coast\r' > /dev/serial0
    echo "run_safe: sent coast to both motors" >&2
}
trap coast EXIT

"$PY" -m "$@" 0<&0 &   # explicit: a background job would otherwise get /dev/null as stdin (prompts)
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM HUP
trap 'kill -INT $child 2>/dev/null' INT
wait $child
status=$?
while kill -0 $child 2>/dev/null; do   # wait returned early because a signal arrived
    wait $child
    status=$?
done
exit $status
