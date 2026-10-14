#!/usr/bin/env bash
#
# worker.sh — RAY PI 3.14 music-video job worker
# Runs as CMD inside a RunPod GPU pod. Fully env-var driven.
# Pipeline per job: download inputs -> per scene (Wan2.2 clip -> audio slice ->
# LatentSync lip-sync) -> concat -> mux full song -> upload -> DONE.
# ALWAYS self-terminates the pod at the end (success or failure) — billing safety.
#
# Required env:
#   SONG_URL      direct URL to the song audio (mp3)
#   IMAGE_URL     direct URL to the character reference image
#   SCENES_JSON   JSON array: [{"prompt":"...","duration":5}, ...] (seconds per scene)
# Optional env:
#   LIPSYNC=true|false        default: true
#   JOB_ID                    default: job-<epoch>
#   MODELS_DIR                default: /workspace/models        (network volume)
#   WORK_BASE                 default: /workspace/jobs          (network volume)
#   WAN_DIR                   default: /opt/Wan2.2              (baked in image)
#   LATENTSYNC_DIR            default: /opt/LatentSync         (baked in image)
#   RESULT_TOPIC              ntfy.sh topic, e.g. raypi-<uuid> (progress + DONE)
#   MAX_RUNTIME_HRS           default: 4 (watchdog self-terminates)
#   ALLOW_MODEL_DOWNLOAD      default: false (fail fast if models missing)
#   VIDEO_SIZE                default: 704*1280  (9:16 vertical, 720p-class for TI2V-5B)
# Auto-injected by RunPod — do NOT set manually:
#   RUNPOD_POD_ID, RUNPOD_API_KEY (pod-scoped; used only for self-terminate)

set -euo pipefail
# Evita la fragmentación de VRAM que causaba OOM en el decode VAE de Wan2.2.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------- config ---
: "${SONG_URL:?SONG_URL env var is required}"
: "${IMAGE_URL:?IMAGE_URL env var is required}"
: "${SCENES_JSON:?SCENES_JSON env var is required}"

LIPSYNC="${LIPSYNC:-true}"
JOB_ID="${JOB_ID:-job-$(date +%s)}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
WORK_BASE="${WORK_BASE:-/workspace/jobs}"
WAN_DIR="${WAN_DIR:-/opt/Wan2.2}"
LATENTSYNC_DIR="${LATENTSYNC_DIR:-/opt/LatentSync}"
RESULT_TOPIC="${RESULT_TOPIC:-}"
MAX_RUNTIME_HRS="${MAX_RUNTIME_HRS:-4}"
ALLOW_MODEL_DOWNLOAD="${ALLOW_MODEL_DOWNLOAD:-false}"
VIDEO_SIZE="${VIDEO_SIZE:-704*1280}"
WAN_CKPT="$MODELS_DIR/Wan2.2-TI2V-5B"
LS_CKPT="$MODELS_DIR/latentsync/checkpoints/latentsync_unet.pt"

WORK_DIR="$WORK_BASE/$JOB_ID"
mkdir -p "$WORK_DIR"/{clips,audio,lipsync,norm}

# ---------------------------------------------------------------- logging ---
# El log va a stdout Y a work.log (el trap finish() lo sube si hay error)
log()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$WORK_DIR/work.log"; }

notify() {
  # Best-effort progress to the phone via ntfy.sh. Never fails the job.
  # ntfy.sh a veces se degrada (respuestas de 20s o 522): se reintenta.
  log "NOTIFY: $*"
  if [ -n "$RESULT_TOPIC" ]; then
    local attempt=0
    while [ $attempt -lt 3 ]; do
      if curl -s -m 30 -d "$*" "https://ntfy.sh/$RESULT_TOPIC" >/dev/null 2>&1; then
        return 0
      fi
      attempt=$((attempt + 1))
      sleep 3
    done
  fi
  return 0
}

# ------------------------------------------------------- self-termination ---
TERMINATED=0
WATCHDOG_PID=""
_rp_env() { # _rp_env VAR — print VAR from env or /etc/rp_environment
  local v="${1}" val=""
  val="${!v:-}"
  if [ -z "$val" ] && [ -f /etc/rp_environment ]; then
    val="$(grep -E "^${v}=" /etc/rp_environment 2>/dev/null | cut -d= -f2- || true)"
  fi
  printf '%s' "$val"
}

self_terminate() {
  [ "$TERMINATED" = "1" ] && return 0
  TERMINATED=1
  local pod_id api_key
  pod_id="$(_rp_env RUNPOD_POD_ID)"
  api_key="$(_rp_env RUNPOD_API_KEY)"
  if [ -n "$pod_id" ] && [ -n "$api_key" ]; then
    log "Terminating pod $pod_id ..."
    curl -s -m 25 -o /dev/null -X POST https://api.runpod.io/graphql \
      -H "Authorization: Bearer $api_key" \
      -H "Content-Type: application/json" \
      -H "User-Agent: Mozilla/5.0" \
      -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\"$pod_id\\\" }) }\"}" || true
    log "Terminate request sent."
  else
    log "WARNING: cannot self-terminate (RUNPOD_POD_ID/RUNPOD_API_KEY unavailable)."
  fi
}

finish() {
  local code=$?
  # stop the watchdog; it must not outlive us
  [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null || true
  if [ "$code" -ne 0 ]; then
    # Subir work.log y mandar URL + cola (mensajes cortos para ntfy)
    local logurl="upload-failed"
    logurl=$(curl -s -m 60 -F "file=@$WORK_DIR/work.log" https://tmpfiles.org/api/v1/upload 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data']['url'].replace('tmpfiles.org/','tmpfiles.org/dl/'))" 2>/dev/null || echo "upload-failed")
    local tail15=$(tail -15 "$WORK_DIR/work.log" 2>/dev/null | tr '\n' '~' | cut -c1-1200)
    notify "ERROR job $JOB_ID failed (exit $code). LOG=$logurl TAIL:$tail15 Terminating pod."
    sleep 3
  fi
  self_terminate
}
trap finish EXIT
trap 'exit 130' TERM INT

# Max-runtime watchdog: a hung job must never burn credit overnight.
(
  sleep $(( MAX_RUNTIME_HRS * 3600 ))
  log "WATCHDOG: max runtime (${MAX_RUNTIME_HRS}h) reached — terminating."
  kill -TERM $$
) &
WATCHDOG_PID=$!

# ---------------------------------------------------------------- checks ---
for bin in python3 ffmpeg ffprobe curl; do
  command -v "$bin" >/dev/null 2>&1 || { log "FATAL: required tool '$bin' missing in image."; exit 10; }
done
[ -d "$WAN_DIR" ] || { log "FATAL: WAN_DIR $WAN_DIR not found (image build issue)."; exit 10; }
[ -d "$LATENTSYNC_DIR" ] || { log "FATAL: LATENTSYNC_DIR $LATENTSYNC_DIR not found (image build issue)."; exit 10; }

check_models() {
  local missing=""
  [ -d "$WAN_CKPT" ] || missing="${missing} Wan2.2-TI2V-5B"
  [ -f "$LS_CKPT" ] || missing="${missing} latentsync_unet.pt"
  [ -f "$MODELS_DIR/latentsync/checkpoints/whisper/tiny.pt" ] || missing="${missing} whisper/tiny.pt"
  if [ -n "$missing" ]; then
    if [ "$ALLOW_MODEL_DOWNLOAD" = "true" ]; then
      log "Downloading missing models:$missing (slow — prefer a pre-seeded volume)"
      mkdir -p "$MODELS_DIR"
      # Descarga con reintentos; el stderr va al log (no se traga con tail).
      # Se desactiva pipefail temporalmente para controlar el error manualmente.
      local attempt=0 dl_ok=0
      set +o pipefail
      while [ $attempt -lt 3 ] && [ $dl_ok -eq 0 ]; do
        attempt=$((attempt + 1))
        log "Wan2.2 download attempt $attempt/3..."
        if huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN_CKPT" 2>&1 | tee -a "$WORK_DIR/work.log" | tail -5; then
          [ -d "$WAN_CKPT" ] && dl_ok=1
        fi
        [ $dl_ok -eq 0 ] && sleep 10
      done
      set -o pipefail
      [ $dl_ok -eq 1 ] || { log "FATAL: Wan2.2 model download failed after 3 attempts."; exit 22; }
      mkdir -p "$MODELS_DIR/latentsync/checkpoints"
      attempt=0 dl_ok=0
      set +o pipefail
      while [ $attempt -lt 3 ] && [ $dl_ok -eq 0 ]; do
        attempt=$((attempt + 1))
        log "LatentSync download attempt $attempt/3..."
        if huggingface-cli download ByteDance/LatentSync-1.6 --local-dir "$MODELS_DIR/latentsync/checkpoints" 2>&1 | tee -a "$WORK_DIR/work.log" | tail -5; then
          [ -f "$MODELS_DIR/latentsync/checkpoints/latentsync_unet.pt" ] && dl_ok=1
        fi
        [ $dl_ok -eq 0 ] && sleep 10
      done
      set -o pipefail
      [ $dl_ok -eq 1 ] || { log "FATAL: LatentSync download failed after 3 attempts."; exit 22; }
    else
      log "FATAL: missing models on volume:$missing"
      log "Seed the network volume once (see runpod-api-notes.md §11), then re-run."
      exit 11
    fi
  fi
  log "Models OK."
}

# ---------------------------------------------------------------- inputs --
dl() { # dl URL DEST — descarga con reintentos; re-resuelve tmpfiles en cada intento
  local url="$1" dest="$2"
  # Un /dl/ SIN token numérico (tmpfiles.org/dl/<id>/<file>) no es link directo:
  # es la página con /dl/ insertado. Se revierte a URL de página para resolverlo.
  if echo "$url" | grep -q "tmpfiles.org/dl/" && ! echo "$url" | grep -qE "tmpfiles.org/dl/[0-9]+\.[a-zA-Z0-9]+/"; then
    url="$(echo "$url" | sed 's|tmpfiles.org/dl/|tmpfiles.org/|')"
    log "URL /dl/ sin token convertida a página de tmpfiles."
  fi
  local is_tmpfiles=0
  echo "$url" | grep -q "tmpfiles.org/" && ! echo "$url" | grep -q "/dl/" && is_tmpfiles=1
  local attempt=1
  while [ $attempt -le 4 ]; do
    local u="$url"
    if [ "$is_tmpfiles" = "1" ]; then
      # Los tokens /dl/ de tmpfiles caducan en minutos y son de un solo uso:
      # se obtiene un link fresco en CADA intento.
      local direct
      direct=$(curl -sL -A "Mozilla/5.0" --max-time 30 "$url" | grep -o 'https://tmpfiles.org/dl/[^"]*' | head -1)
      if [ -n "$direct" ]; then u="$direct"; log "Link directo fresco (intento $attempt)."; fi
    fi
    if curl -sSL --max-time 300 -o "$dest" "$u" 2>/dev/null && [ -s "$dest" ]; then
      # Si lo descargado es una página HTML, es un enlace muerto: reintentar.
      if head -c 15 "$dest" | grep -qi "<!DOCTYPE\|<html"; then
        log "WARN: descarga devolvió HTML (enlace caducado), reintentando..."
      else
        return 0
      fi
    else
      log "WARN: descarga falló (intento $attempt), reintentando..."
    fi
    attempt=$((attempt+1))
    sleep 5
  done
  return 1
}

log "=== RAY PI 3.14 video job $JOB_ID starting ==="
notify "START job $JOB_ID"
notify "PHASE models"
check_models
notify "PHASE models-ok"

log "Downloading song..."
notify "PHASE inputs"
dl "$SONG_URL" "$WORK_DIR/song.mp3"
# Validar que no sea una página de error HTML (tmpfiles expira a los 60 min)
head -c 4 "$WORK_DIR/song.mp3" | grep -q "^ID3\|^RIFF\|\xff\xfb\|\xff\xf3\|\xff\xf2" || {
  log "FATAL: song.mp3 no es audio válido (¿URL expirada?). Primeros bytes:"
  head -c 200 "$WORK_DIR/song.mp3" | tr '\n' ' ' | cut -c1-200 | tee -a "$WORK_DIR/work.log"
  exit 20
}
log "Downloading character image..."
dl "$IMAGE_URL" "$WORK_DIR/input_raw.jpg"
head -c 2 "$WORK_DIR/input_raw.jpg" | grep -q $'\xff\xd8' || {
  log "FATAL: input_raw.jpg no es JPEG válido (¿URL expirada?)."
  head -c 200 "$WORK_DIR/input_raw.jpg" | tr '\n' ' ' | cut -c1-200 | tee -a "$WORK_DIR/work.log"
  exit 21
}
notify "PHASE inputs-ok"

# Wan I2V follows the INPUT image aspect ratio: force 9:16 vertical for reels.
log "Preparing 9:16 reference frame..."
ffmpeg -y -v error -i "$WORK_DIR/input_raw.jpg" \
  -vf "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280" \
  "$WORK_DIR/input.jpg"

SONG_DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK_DIR/song.mp3")"
log "Song duration: ${SONG_DUR}s"

# Parse SCENES_JSON -> scenes.tsv : idx \t duration \t base64(prompt)
python3 - <<'PYEOF' > "$WORK_DIR/scenes.tsv"
import json, os, base64, sys
try:
    scenes = json.loads(os.environ["SCENES_JSON"])
except Exception as e:
    print(f"SCENES_JSON parse error: {e}", file=sys.stderr); sys.exit(2)
if not isinstance(scenes, list) or not scenes:
    print("SCENES_JSON must be a non-empty array", file=sys.stderr); sys.exit(2)
for i, s in enumerate(scenes):
    prompt = s.get("prompt", "") or ""
    try: duration = float(s.get("duration", 5))
    except Exception: duration = 5.0
    duration = min(max(duration, 1.0), 12.0)
    b64 = base64.b64encode(prompt.encode("utf-8")).decode()
    print(f"{i}\t{duration}\t{b64}")
PYEOF
N_SCENES="$(wc -l < "$WORK_DIR/scenes.tsv")"
log "Scenes: $N_SCENES"

# ------------------------------------------------------------- primitives ---
valid_video() { # valid_video FILE — true if it exists and has >1s of video
  [ -f "$1" ] || return 1
  local d
  d="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null || echo 0)"
  awk -v d="$d" 'BEGIN{exit !(d>1)}'
}

# Wan2.2 TI2V-5B image-to-video. Writes clips/scene_<i>.mp4 (checkpointed).
gen_scene() {
  local idx="$1" prompt="$2" duration="$3"
  local out="$WORK_DIR/clips/scene_${idx}.mp4"
  if valid_video "$out"; then log "scene $idx: clip cached, skipping generation"; return 0; fi

  # Wan's VAE compresses time 4x -> frames must be 4n+1. 24fps native for TI2V-5B.
  local frames
  frames="$(awk -v d="$duration" 'BEGIN{ f=int(d*24)+1; n=int((f-1+3)/4); print n*4+1 }')"
  log "scene $idx: generating ${duration}s (${frames}f) with Wan2.2-TI2V-5B ..."

  local gdir="$WORK_DIR/gen_$idx"
  rm -rf "$gdir"; mkdir -p "$gdir"
  # NOTE: "--size" uses '*' (an AREA spec); quote it so the shell does not glob.
  # NOTE: --frame_num flag name + generate.py output filename: VERIFY on first run.
  if ! ( cd "$gdir" && python3 "$WAN_DIR/generate.py" \
      --task ti2v-5B \
      --size "$VIDEO_SIZE" \
      --ckpt_dir "$WAN_CKPT" \
      --offload_model True --convert_model_dtype --t5_cpu \
      --image "$WORK_DIR/input.jpg" \
      --prompt "$prompt" \
      --frame_num "$frames" 2>&1 | tee -a "$WORK_DIR/work.log" ); then
    log "scene $idx: generation FAILED"
    return 1
  fi
  local produced
  produced="$(ls -t "$gdir"/*.mp4 2>/dev/null | head -n1 || true)"
  if [ -z "$produced" ] || ! valid_video "$produced"; then
    log "scene $idx: no valid mp4 produced in $gdir"
    return 1
  fi
  mv "$produced" "$out"
  rm -rf "$gdir"
  log "scene $idx: clip ready -> $out"
  return 0
}

# Slice the song segment matching this scene -> 16kHz mono wav for LatentSync.
slice_audio() {
  local idx="$1" start="$2" duration="$3"
  local out="$WORK_DIR/audio/scene_${idx}.wav"
  [ -f "$out" ] && return 0
  ffmpeg -y -v error -ss "$start" -t "$duration" -i "$WORK_DIR/song.mp3" \
    -ar 16000 -ac 1 "$out"
}

# LatentSync lip-sync for one scene. Falls back to the raw clip on failure
# (e.g. no detectable face) so one bad scene never kills the job.
lipsync_scene() {
  local idx="$1"
  local clip="$WORK_DIR/clips/scene_${idx}.mp4"
  local wav="$WORK_DIR/audio/scene_${idx}.wav"
  local out="$WORK_DIR/lipsync/scene_${idx}.mp4"
  if [ "$LIPSYNC" != "true" ]; then return 0; fi
  if valid_video "$out"; then log "scene $idx: lipsync cached, skipping"; return 0; fi
  log "scene $idx: lip-syncing with LatentSync ..."
  # LatentSync busca whisper en la ruta relativa checkpoints/whisper/tiny.pt
  # desde su propio dir; enlazamos los checkpoints descargados ahí.
  ln -sfn "$MODELS_DIR/latentsync/checkpoints" "$LATENTSYNC_DIR/checkpoints" 2>/dev/null || true
  # NOTE: config path (stage2.yaml) is for LatentSync 1.6 — VERIFY on first run.
  if ( cd "$LATENTSYNC_DIR" && python3 -m scripts.inference \
      --unet_config_path "configs/unet/stage2.yaml" \
      --inference_ckpt_path "$LS_CKPT" \
      --video_path "$clip" \
      --audio_path "$wav" \
      --video_out_path "$out" \
      --inference_steps 20 --guidance_scale 1.5 --seed 1247 2>&1 | tee -a "$WORK_DIR/work.log" ); then
    valid_video "$out" && { log "scene $idx: lip-sync OK"; return 0; }
  fi
  log "scene $idx: WARNING lip-sync failed — using raw clip"
  rm -f "$out"
  return 0
}

# Normalize to 24fps h264/yuv420p so concat never chokes on mixed fps/codecs.
normalize_scene() {
  local idx="$1"
  local src="$WORK_DIR/lipsync/scene_${idx}.mp4"
  valid_video "$src" || src="$WORK_DIR/clips/scene_${idx}.mp4"
  local out="$WORK_DIR/norm/scene_${idx}.mp4"
  if valid_video "$out"; then return 0; fi
  ffmpeg -y -v error -i "$src" -r 24 -c:v libx264 -preset veryfast -crf 20 \
    -pix_fmt yuv420p -c:a aac "$out"
}

upload_file() { # upload_file FILE — prints URL; tries catbox -> litterbox -> uguu -> tmpfiles
  local file="$1" url=""
  log "Uploading $(du -h "$file" | cut -f1) to catbox.moe ..."
  url="$(curl -s -m 180 -F "reqtype=fileupload" -F "fileToUpload=@$file" \
    https://catbox.moe/user/api.php || true)"
  if [[ "$url" == https://files.catbox.moe/* ]]; then echo "$url"; return 0; fi
  log "catbox failed, trying litterbox (72h) ..."
  url="$(curl -s -m 240 -F "reqtype=fileupload" -F "time=72h" \
    -F "fileToUpload=@$file" https://litterbox.catbox.moe/resources/internals/api.php || true)"
  if [[ "$url" == https://litterbox.catbox.moe/* ]]; then echo "$url"; return 0; fi
  log "litterbox failed, trying uguu.se ..."
  url="$(curl -s -m 240 -F "files[]=@$file" https://uguu.se/upload || true)"
  if [[ "$url" == https://* ]]; then echo "$url"; return 0; fi
  log "uguu failed, trying tmpfiles.org ..."
  url="$(curl -s -m 240 -F "file=@$file" https://tmpfiles.org/api/v1/upload 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['url'])" 2>/dev/null || true)"
  if [[ "$url" == https://tmpfiles.org/* ]]; then echo "$url"; return 0; fi
  return 1
}

# ------------------------------------------------------------- main loop ---
FAILED=""
t=0
while IFS=$'\t' read -r idx duration b64; do
  prompt="$(printf '%s' "$b64" | base64 -d)"
  log "--- scene $idx/$((N_SCENES-1)): ${duration}s ---"
  notify "SCENE $((idx+1))/$N_SCENES"
  if ! gen_scene "$idx" "$prompt" "$duration"; then
    log "scene $idx: retrying once ..."
    sleep 5
    gen_scene "$idx" "$prompt" "$duration" || {
      log "scene $idx: FAILED twice — skipping (job continues)"
      FAILED="${FAILED} ${idx}"
      t="$(awk -v t="$t" -v d="$duration" 'BEGIN{print t+d}')"
      continue
    }
  fi
  slice_audio "$idx" "$t" "$duration"
  lipsync_scene "$idx"
  normalize_scene "$idx"
  t="$(awk -v t="$t" -v d="$duration" 'BEGIN{print t+d}')"
done < "$WORK_DIR/scenes.tsv"

# ------------------------------------------------- concat / mux / upload ---
[ -n "$(ls "$WORK_DIR"/norm/scene_*.mp4 2>/dev/null)" ] || {
  log "FATAL: no usable scenes produced."; exit 20
}
: > "$WORK_DIR/concat.txt"
for f in "$WORK_DIR"/norm/scene_*.mp4; do
  valid_video "$f" && echo "file '$f'" >> "$WORK_DIR/concat.txt"
done

log "Concatenating scenes..."
ffmpeg -y -v error -f concat -safe 0 -i "$WORK_DIR/concat.txt" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
  "$WORK_DIR/assembled.mp4"

log "Muxing full song audio..."
ffmpeg -y -v error -i "$WORK_DIR/assembled.mp4" -i "$WORK_DIR/song.mp3" \
  -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart \
  "$WORK_DIR/final.mp4"
valid_video "$WORK_DIR/final.mp4" || { log "FATAL: final.mp4 invalid."; exit 21; }

FINAL_URL=""
if FINAL_URL="$(upload_file "$WORK_DIR/final.mp4")"; then
  log "Uploaded: $FINAL_URL"
else
  log "FATAL: all upload hosts failed. final.mp4 kept at $WORK_DIR/final.mp4 on the volume."
  notify "ERROR upload failed; video is on the volume at $WORK_DIR/final.mp4"
  exit 22
fi

[ -n "$FAILED" ] && notify "WARN scenes skipped:$FAILED"
notify "DONE $FINAL_URL"
echo "DONE $FINAL_URL"
log "=== job $JOB_ID complete ==="
exit 0
