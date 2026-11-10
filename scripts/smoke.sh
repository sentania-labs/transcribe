#!/usr/bin/env bash
# Boot the image the way the cluster does, minus the model volume, and check
# what can be checked without the gated weights: the server starts, /health is
# 503, /v1/status needs the key and names the missing files, and every error
# has the {"error": {"reason", "message"}} body with the right reason. The
# real transcription proof needs the model volume and is done in the lab.
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
base="http://127.0.0.1:$port"
key='Authorization: Bearer smoke-key'
[ "$(code "$base/health")" = 503 ]
# Status: bearer-protected, names the missing model files.
[ "$(code "$base/v1/status")" = 401 ]
status=$(curl -s -H "$key" "$base/v1/status")
echo "status: $status"
grep -q 'model files missing' <<<"$status"
grep -q '"max_upload_bytes"' <<<"$status"
printf 'RIFF' > /tmp/smoke-$$.wav
# Every error: {"error": {"reason", "message"}} with a stable reason.
body=$(curl -s -F file=@/tmp/smoke-$$.wav "$base/v1/audio/transcriptions")
grep -q '"reason":"unauthorized"' <<<"$body"
[ "$(code -H 'Authorization: Bearer wrong' -F file=@/tmp/smoke-$$.wav "$base/v1/audio/transcriptions")" = 401 ]
body=$(curl -s -H "$key" -F file=@/tmp/smoke-$$.wav "$base/v1/audio/transcriptions")
grep -q '"reason":"internal"' <<<"$body"
[ "$(code -H "$key" -F file=@/tmp/smoke-$$.wav "$base/v1/audio/transcriptions/jobs")" = 503 ]
body=$(curl -s -H "$key" -H 'Transfer-Encoding: chunked' -F file=@/tmp/smoke-$$.wav "$base/v1/audio/transcriptions")
grep -q '"reason":"length_required"' <<<"$body"
body=$(curl -s -H "$key" "$base/v1/audio/transcriptions/jobs/nope")
grep -q '"reason":"not_found"' <<<"$body"
rm -f /tmp/smoke-$$.wav
echo "smoke: ok"
