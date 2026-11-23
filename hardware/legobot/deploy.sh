#!/bin/bash
# Copy the robot code, and the simulator code the balancing loop imports, to the Pi's ~/legobot:
#     bash hardware/legobot/deploy.sh [host]          (default host: legopi, from ~/.ssh/config)
# Layout on the Pi: ~/legobot/{robot/, config.py, legobot_config.py, simulation/balance_sim/, DEPLOYED}.
# DEPLOYED records the commit (with -dirty for uncommitted changes to these files) and the time; balance.py
# copies it into every log. Tests and caches are left out; logs/ and calibration.json on the Pi are untouched.
set -euo pipefail
HOST="${1:-legopi}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
FILES=(config.py legobot_config.py simulation/balance_sim)
DIRTY=""
git diff --quiet HEAD -- "${FILES[@]}" hardware/legobot/robot || DIRTY="-dirty"
[ -z "$(git ls-files --others --exclude-standard -- hardware/legobot/robot simulation/balance_sim)" ] || DIRTY="-dirty"
VERSION="$(git rev-parse --short HEAD)$DIRTY $(date '+%Y-%m-%d %H:%M')"
tar --exclude=__pycache__ --exclude='test_*.py' -cf - "${FILES[@]}" -C hardware/legobot robot \
    | ssh "$HOST" "mkdir -p ~/legobot && tar -xf - -C ~/legobot && chmod +x ~/legobot/robot/*.sh \
                   && echo '$VERSION' > ~/legobot/DEPLOYED"
echo "deployed $VERSION to $HOST:~/legobot"
