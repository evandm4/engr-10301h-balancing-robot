#!/bin/bash
# Does the Build HAT keep driving after the Python process is killed with -9? (C1; DATASHEET 3.3)
# WHEELS OFF THE GROUND. Usage from ~/legobot:  robot/kill9_test.sh [duty, default 0.15]
#   Part 1: bench_log run directly, kill -9 after 3 s, then read the HAT's stream for speed.
#   Part 2: the same under run_safe.sh, which should coast the motors after the kill.
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/legobot-venv/bin/python"
DUTY="${1:-0.15}"

peek() {   # print the newest speed on each port from 0.5 s of stream, without sending anything
    "$PY" - <<'EOF'
import serial, time
s = serial.Serial("/dev/serial0", 115200, timeout=0.1)
s.reset_input_buffer()
end, last = time.time() + 0.5, {}
while time.time() < end:
    line = s.readline().decode("ascii", "replace").strip()
    if line[:1] == "P" and line[2:5] == "C0:":
        last[line[1]] = line
print("   stream:", last if last else "no data lines (stream stopped)")
EOF
}
coast() {
    stty -F /dev/serial0 115200 raw -echo
    printf 'port 0 ; coast ; port 1 ; coast ; port 0 ; select ; port 1 ; select\r' > /dev/serial0
}

echo "Part 1: plain python, kill -9"
"$PY" -m robot.bench_log --seconds 20 --pwm "$DUTY" --no-imu > /tmp/kill9_part1.log 2>&1 &
pid=$!
sleep 3
kill -9 $pid; wait $pid 2>/dev/null
sleep 1.5
echo "  1.5 s after kill -9:"; peek
sleep 1.5
echo "  3 s after kill -9:"; peek
coast; sleep 0.3
echo "  after coast:"; peek

echo "Part 2: under run_safe.sh, kill -9 of the python child"
robot/run_safe.sh robot.bench_log --seconds 20 --pwm "$DUTY" --no-imu > /tmp/kill9_part2.log 2>&1 &
wrapper=$!
sleep 3
child=$(pgrep -P $wrapper -f robot.bench_log)
kill -9 "$child"; wait $wrapper 2>/dev/null
sleep 1.5
echo "  1.5 s after kill -9 (wrapper should have coasted):"; peek
coast
tail -n 2 /tmp/kill9_part2.log
