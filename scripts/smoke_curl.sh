#!/usr/bin/env bash
# Smoke-test the running TTS API with curl.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
OUT_DIR="${OUT_DIR:-./output-smoke}"
mkdir -p "${OUT_DIR}"

echo "==> GET ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health" | tee "${OUT_DIR}/health.json"
echo

echo "==> POST ${BASE_URL}/tts (yue)"
curl -fsS -X POST "${BASE_URL}/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，我係測試。",
    "language": "yue",
    "speaker": "default",
    "speed": 1.0,
    "format": "wav"
  }' \
  --output "${OUT_DIR}/tts_yue.wav"
ls -la "${OUT_DIR}/tts_yue.wav"

echo "==> POST ${BASE_URL}/tts-file"
printf '%s\n' '你好，我係檔案測試。' > "${OUT_DIR}/sample.txt"
curl -fsS -X POST "${BASE_URL}/tts-file" \
  -F "file=@${OUT_DIR}/sample.txt" \
  -F "language=yue" \
  -F "format=wav" \
  --output "${OUT_DIR}/tts_file.wav"
ls -la "${OUT_DIR}/tts_file.wav"

echo "==> POST ${BASE_URL}/tts-long"
curl -fsS -X POST "${BASE_URL}/tts-long" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "第一段。這是長文測試。\n\n第二段。我們會自動分句與合併音訊。",
    "language": "zh",
    "speed": 1.0
  }' \
  --output "${OUT_DIR}/tts_long.mp3"
ls -la "${OUT_DIR}/tts_long.mp3"

echo "Smoke tests wrote files under ${OUT_DIR}"
