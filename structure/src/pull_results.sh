#!/bin/bash
# Local watcher: polls the instance for the overnight pipeline's RESULTS_READY
# marker, then downloads and unpacks RESULTS.tgz.
#
# Downloads land in the project's tmp/ directory, never the project root.
# Runs detached (setsid nohup) so it survives the session that started it.
# Also snapshots the remote log each cycle, so progress is visible locally even
# before the run finishes.

PORT=${REMOTE_PORT:?set REMOTE_PORT}
HOST=root@${REMOTE_HOST:?set REMOTE_HOST}
SSHO="-p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25"
DEST=/home/ileven/Projects/cyp-challenge/tmp/instance_20261103
LOG=$DEST/watcher.log
mkdir -p "$DEST"

say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
say "watcher started, polling every 5 min"

for i in $(seq 1 240); do   # up to 20 hours
  # snapshot progress (best effort; a dropped connection must not kill the loop)
  scp $SSHO -q "$HOST:/workspace/cyp/overnight.log" "$DEST/overnight.log" 2>/dev/null
  scp $SSHO -q "$HOST:/workspace/cyp/runs/progress.txt" "$DEST/progress.txt" 2>/dev/null

  if ssh $SSHO "$HOST" 'test -f /workspace/cyp/RESULTS_READY' 2>/dev/null; then
    say "RESULTS_READY seen - downloading"
    for attempt in 1 2 3 4 5; do
      if scp $SSHO -q "$HOST:/workspace/cyp/RESULTS.tgz" "$DEST/RESULTS.tgz" 2>>"$LOG"; then
        tar xzf "$DEST/RESULTS.tgz" -C "$DEST" && say "unpacked to $DEST/pack"
        ls -la "$DEST/pack/zips" >> "$LOG" 2>&1
        say "DONE"
        exit 0
      fi
      say "download attempt $attempt failed, retrying in 60s"
      sleep 60
    done
    say "GAVE UP after 5 attempts - tarball still on instance at /workspace/cyp/RESULTS.tgz"
    exit 1
  fi
  sleep 300
done
say "timed out after 20 h without RESULTS_READY"
exit 1
