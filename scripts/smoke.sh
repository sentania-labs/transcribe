#!/usr/bin/env bash
# Boot the image the way the cluster does, minus the model volume, and check
# what can be checked without the gated weights: the server starts, /health
# names the missing files (503), the API refuses a missing key (401) and,
# with the key, says the models are unavailable (503). The real transcription
# proof needs the model volume and is done in the lab (see README.md).
set -euo pipefail
image=${1:?usage: smoke.sh IMAGE}
name=transcribe-smoke-$$
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$name" -p 127.0.0.1::8000 -e TRANSCRIBE_API_KEY=smoke-key \
  --read-only --tmpfs /tmp "$image" >/dev/null
port=$(docker port "$name" 8000/tcp | head -1 | cut -d: -f2)
for _ in $(seq 1 60); do
  curl -s "http://127.0.0.1:$port/health" -o /dev/null && break
  sleep 1
done
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
health=$(curl -s "http://127.0.0.1:$port/health")
echo "health: $health"
grep -q 'model files missing' <<<"$health"
[ "$(code "http://127.0.0.1:$port/health")" = 503 ]
printf 'RIFF' > /tmp/smoke-$$.wav
[ "$(code -F file=@/tmp/smoke-$$.wav "http://127.0.0.1:$port/v1/audio/transcriptions")" = 401 ]
[ "$(code -H 'Authorization: Bearer wrong' -F file=@/tmp/smoke-$$.wav "http://127.0.0.1:$port/v1/audio/transcriptions")" = 401 ]
[ "$(code -H 'Authorization: Bearer smoke-key' -F file=@/tmp/smoke-$$.wav "http://127.0.0.1:$port/v1/audio/transcriptions")" = 503 ]
rm -f /tmp/smoke-$$.wav
echo "smoke: ok"
