from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

# ----------------------------- Configuration -----------------------------

MOVIE_DIR = Path(os.getenv("MOVIE_DIR", r"E:\movie")).expanduser().resolve()
CACHE_DIR = Path(os.getenv("MOVIE_CACHE_DIR", str(MOVIE_DIR / ".movie_server_cache"))).resolve()
HLS_DIR = CACHE_DIR / "hls"
THUMB_DIR = CACHE_DIR / "thumbnails"
DB_PATH = CACHE_DIR / "media.db"

APP_DIR = Path(__file__).resolve().parent


def resolve_media_tool(env_name: str, bundled_name: str, command: str) -> str:
    configured = os.getenv(env_name)
    if configured:
        return configured

    bundled = APP_DIR / bundled_name
    if bundled.is_file():
        return str(bundled)
    return shutil.which(command) or command


FFMPEG = resolve_media_tool("FFMPEG_PATH", "ffmpeg.exe", "ffmpeg")
FFPROBE = resolve_media_tool("FFPROBE_PATH", "ffprobe.exe", "ffprobe")

# CPU is the safe default. Override with H264_ENCODER=h264_nvenc, h264_qsv,
# or h264_amf only if your GPU, drivers, and FFmpeg build support it.
H264_ENCODER = os.getenv("H264_ENCODER", "libx264")
X264_PRESET = os.getenv("X264_PRESET", "ultrafast")
HLS_SEGMENT_SECONDS = int(os.getenv("HLS_SEGMENT_SECONDS", "4"))
SUPPORTED_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".wmv", ".webm", ".ts", ".m2ts"}

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("movie-server")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
HLS_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

media_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-worker")
thumbnail_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thumbnail-worker")
jobs_lock = threading.RLock()
jobs: dict[str, dict[str, Any]] = {}
movie_index: dict[str, dict[str, Any]] = {}
probe_cache: dict[str, dict[str, Any]] = {}
thumb_jobs: set[str] = set()


# ----------------------------- Database -----------------------------------

def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db_connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS playback_progress (
                movie_id TEXT PRIMARY KEY,
                position REAL NOT NULL DEFAULT 0,
                duration REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)
        con.commit()


init_db()


# ----------------------------- Utilities ---------------------------------

def ffmpeg_available() -> bool:
    return all(
        Path(tool).is_file() or shutil.which(tool) is not None
        for tool in (FFMPEG, FFPROBE)
    )


def safe_movie_path(relative_name: str) -> Path:
    if not relative_name or "\x00" in relative_name:
        abort(400, "Invalid movie path")
    candidate = (MOVIE_DIR / relative_name).resolve()
    try:
        candidate.relative_to(MOVIE_DIR)
    except ValueError:
        abort(403, "Path is outside the movie directory")
    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS or not candidate.is_file():
        abort(404, "Movie not found")
    return candidate


def make_movie_id(path: Path) -> str:
    stat = path.stat()
    relative = path.relative_to(MOVIE_DIR).as_posix()
    raw = f"{relative}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def probe_media(path: Path) -> dict[str, Any]:
    cache_key = f"{path}|{path.stat().st_mtime_ns}|{path.stat().st_size}"
    if cache_key in probe_cache:
        return probe_cache[cache_key]

    if not ffmpeg_available():
        raise RuntimeError(
            "FFmpeg/FFprobe not found. Install FFmpeg and add its bin folder to PATH, "
            "or set the FFMPEG_PATH and FFPROBE_PATH environment variables."
        )

    cmd = [
        str(FFPROBE), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFprobe could not read this file")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})
    info = {
        "video_codec": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
        "audio_codec": audio.get("codec_name"),
        "container": (fmt.get("format_name") or "").lower(),
        "duration": float(fmt.get("duration") or 0),
        "width": video.get("width"),
        "height": video.get("height"),
    }
    probe_cache[cache_key] = info
    return info


def direct_play_compatible(path: Path, info: dict[str, Any]) -> bool:
    # Direct MP4/M4V/MOV avoids any preparation delay. MKV still goes through HLS.
    return (
        path.suffix.lower() in {".mp4", ".m4v", ".mov"}
        and info.get("video_codec") == "h264"
        and info.get("pixel_format") in {"yuv420p", "yuvj420p"}
        and info.get("audio_codec") in {None, "aac"}
    )


def list_movies() -> list[dict[str, Any]]:
    result = []
    movie_index.clear()
    for path in MOVIE_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # Do not index files created inside the server cache.
        try:
            relative = path.relative_to(MOVIE_DIR)
        except ValueError:
            continue
        if CACHE_DIR == path or CACHE_DIR in path.parents:
            continue
        try:
            movie_id = make_movie_id(path)
            item = {
                "id": movie_id,
                "name": path.stem,
                "filename": relative.as_posix(),
                "size": path.stat().st_size,
            }
            movie_index[movie_id] = item
            result.append(item)
        except OSError:
            continue
    result.sort(key=lambda x: x["name"].casefold())
    return result


def placeholder_svg() -> Response:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">
    <rect width="320" height="180" fill="#20242b"/>
    <path d="M132 55 L132 125 L195 90 Z" fill="#8b95a5"/>
    <text x="160" y="155" text-anchor="middle" fill="#aab2bf" font-family="Arial" font-size="12">Preparing thumbnail</text>
    </svg>"""
    return Response(svg, mimetype="image/svg+xml", headers={"Cache-Control": "no-store"})


# ----------------------------- Background work ---------------------------

def create_thumbnail(movie_id: str, source: Path):
    output = THUMB_DIR / f"{movie_id}.jpg"
    if output.exists() or not ffmpeg_available():
        return
    temp = THUMB_DIR / f"{movie_id}.tmp.jpg"
    cmd = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "00:00:05", "-i", str(source),
        "-frames:v", "1", "-vf", "scale=480:-2",
        "-q:v", "5", str(temp)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
        if temp.exists() and temp.stat().st_size > 0:
            temp.replace(output)
    except Exception as exc:
        log.warning("Thumbnail failed for %s: %s", source.name, exc)
        temp.unlink(missing_ok=True)
    finally:
        with jobs_lock:
            thumb_jobs.discard(movie_id)


def queue_thumbnail(movie_id: str, source: Path):
    with jobs_lock:
        if movie_id in thumb_jobs or (THUMB_DIR / f"{movie_id}.jpg").exists():
            return
        thumb_jobs.add(movie_id)
    thumbnail_executor.submit(create_thumbnail, movie_id, source)


def build_hls(movie_id: str, source: Path, info: dict[str, Any]):
    output_dir = HLS_DIR / movie_id
    output_dir.mkdir(parents=True, exist_ok=True)
    playlist = output_dir / "index.m3u8"
    log_file = output_dir / "ffmpeg.log"

    video_copy = (
        info.get("video_codec") == "h264"
        and info.get("pixel_format") in {"yuv420p", "yuvj420p"}
    )
    audio_copy = info.get("audio_codec") in {None, "aac"}

    cmd = [
        str(FFMPEG), "-hide_banner", "-y",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-sn", "-dn",
    ]

    if video_copy:
        cmd += ["-c:v", "copy"]
    else:
        cmd += [
            "-c:v", H264_ENCODER,
            "-pix_fmt", "yuv420p",
        ]
        if H264_ENCODER == "libx264":
            cmd += ["-preset", X264_PRESET, "-crf", "23"]
        elif H264_ENCODER == "h264_nvenc":
            cmd += ["-preset", "p4", "-cq", "23"]
        elif H264_ENCODER == "h264_qsv":
            cmd += ["-preset", "veryfast", "-global_quality", "23"]
        elif H264_ENCODER == "h264_amf":
            cmd += ["-quality", "speed", "-qp_i", "23", "-qp_p", "23"]

        # Short GOPs let HLS publish segments promptly and predictably.
        cmd += ["-g", str(max(24, HLS_SEGMENT_SECONDS * 30)), "-sc_threshold", "0"]

    if audio_copy:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ac", "2"]

    cmd += [
        "-f", "hls",
        "-hls_time", str(HLS_SEGMENT_SECONDS),
        "-hls_playlist_type", "event",
        "-hls_flags", "independent_segments+temp_file",
        "-hls_segment_filename", str(output_dir / "segment_%06d.ts"),
        str(playlist),
    ]

    with jobs_lock:
        jobs[movie_id] = {
            "state": "processing",
            "error": None,
            "started_at": time.time(),
            "mode": "hls",
            "log_file": str(log_file),
        }

    log.info("Starting HLS preparation: %s (video_copy=%s, audio_copy=%s)", source.name, video_copy, audio_copy)
    try:
        with open(log_file, "w", encoding="utf-8", errors="replace") as log_stream:
            process = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=log_stream, text=True
            )
        if process.returncode != 0:
            tail = ""
            try:
                tail = log_file.read_text(encoding="utf-8", errors="replace")[-5000:]
            except OSError:
                pass
            raise RuntimeError(tail or f"FFmpeg exited with code {process.returncode}")

        with jobs_lock:
            jobs[movie_id]["state"] = "complete"
    except Exception as exc:
        log.exception("HLS preparation failed for %s", source.name)
        with jobs_lock:
            jobs[movie_id]["state"] = "error"
            jobs[movie_id]["error"] = str(exc)[-5000:]


def start_preparation(movie_id: str, force_hls: bool = False) -> dict[str, Any]:
    item = movie_index.get(movie_id)
    if not item:
        # Refresh the index in case the browser page has been open for a while.
        list_movies()
        item = movie_index.get(movie_id)
    if not item:
        raise FileNotFoundError("Movie not found. Refresh the movie library.")

    source = safe_movie_path(item["filename"])
    info = probe_media(source)
    queue_thumbnail(movie_id, source)

    is_mkv = source.suffix.lower() == ".mkv"
    if direct_play_compatible(source, info) or (is_mkv and not force_hls):
        return {
            "mode": "direct",
            "url": f"/direct/{movie_id}",
            "duration": info.get("duration", 0),
        }

    output_dir = HLS_DIR / movie_id
    playlist = output_dir / "index.m3u8"
    first_segment_exists = any(output_dir.glob("segment_*.ts")) if output_dir.exists() else False

    with jobs_lock:
        current = jobs.get(movie_id)
        if current and current.get("state") in {"processing", "complete"}:
            pass
        elif playlist.exists() and first_segment_exists:
            jobs[movie_id] = {"state": "complete", "mode": "hls", "error": None}
        else:
            jobs[movie_id] = {"state": "queued", "mode": "hls", "error": None}
            media_executor.submit(build_hls, movie_id, source, info)

    return {"mode": "hls", "duration": info.get("duration", 0)}


# ----------------------------- API routes ---------------------------------

@app.get("/")
def index():
    return PAGE


@app.get("/api/movies")
def api_movies():
    if not MOVIE_DIR.exists():
        return jsonify({"error": f"Movie directory does not exist: {MOVIE_DIR}"}), 500
    movies = list_movies()
    with db_connect() as con:
        rows = con.execute("SELECT movie_id, position, duration FROM playback_progress").fetchall()
    progress = {r["movie_id"]: {"position": r["position"], "duration": r["duration"]} for r in rows}
    for movie in movies:
        movie["progress"] = progress.get(movie["id"], {"position": 0, "duration": 0})
    return jsonify({"movies": movies})


@app.post("/api/prepare")
def api_prepare():
    data = request.get_json(silent=True) or {}
    movie_id = str(data.get("movie_id", ""))
    try:
        result = start_preparation(movie_id, force_hls=data.get("force_hls") is True)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        log.exception("Could not prepare movie")
        return jsonify({"error": str(exc)}), 500


@app.get("/api/status/<movie_id>")
def api_status(movie_id: str):
    output_dir = HLS_DIR / movie_id
    playlist = output_dir / "index.m3u8"
    has_segment = output_dir.exists() and any(output_dir.glob("segment_*.ts"))
    with jobs_lock:
        job = dict(jobs.get(movie_id, {}))
    if has_segment and playlist.exists():
        return jsonify({
            "state": job.get("state", "processing"),
            "ready": True,
            "complete": job.get("state") == "complete",
            "url": f"/hls/{movie_id}/index.m3u8",
            "error": job.get("error"),
        })
    if job.get("state") == "error":
        return jsonify({"state": "error", "ready": False, "error": job.get("error")})
    return jsonify({"state": job.get("state", "queued"), "ready": False})


@app.get("/direct/<movie_id>")
def direct_stream(movie_id: str):
    item = movie_index.get(movie_id)
    if not item:
        list_movies()
        item = movie_index.get(movie_id)
    if not item:
        abort(404)
    path = safe_movie_path(item["filename"])
    mimetype = "video/x-matroska" if path.suffix.lower() == ".mkv" else "video/mp4"
    return send_file(path, conditional=True, mimetype=mimetype, max_age=0)


@app.get("/hls/<movie_id>/index.m3u8")
def hls_playlist(movie_id: str):
    path = HLS_DIR / movie_id / "index.m3u8"
    if not path.is_file():
        abort(404)
    response = send_file(path, mimetype="application/vnd.apple.mpegurl", max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/hls/<movie_id>/<segment_name>")
def hls_segment(movie_id: str, segment_name: str):
    if not segment_name.endswith(".ts") or Path(segment_name).name != segment_name:
        abort(404)
    path = HLS_DIR / movie_id / segment_name
    if not path.is_file():
        abort(404)
    response = send_file(path, mimetype="video/mp2t", conditional=True, max_age=3600)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/thumbnail/<movie_id>.jpg")
def thumbnail(movie_id: str):
    item = movie_index.get(movie_id)
    if not item:
        list_movies()
        item = movie_index.get(movie_id)
    if not item:
        abort(404)
    path = THUMB_DIR / f"{movie_id}.jpg"
    if path.exists():
        return send_file(path, mimetype="image/jpeg", max_age=86400)
    try:
        queue_thumbnail(movie_id, safe_movie_path(item["filename"]))
    except Exception:
        pass
    return placeholder_svg()


@app.get("/api/progress/<movie_id>")
def get_progress(movie_id: str):
    with db_connect() as con:
        row = con.execute(
            "SELECT position, duration FROM playback_progress WHERE movie_id = ?",
            (movie_id,),
        ).fetchone()
    if not row:
        return jsonify({"position": 0, "duration": 0})
    return jsonify({"position": row["position"], "duration": row["duration"]})


@app.post("/api/progress/<movie_id>")
def save_progress(movie_id: str):
    data = request.get_json(silent=True) or {}
    try:
        position = max(0.0, float(data.get("position", 0)))
        duration = max(0.0, float(data.get("duration", 0)))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid progress values"}), 400

    # If playback is essentially complete, clear the resume point.
    if duration > 0 and position / duration > 0.95:
        position = 0.0

    with db_connect() as con:
        con.execute("""
            INSERT INTO playback_progress(movie_id, position, duration, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(movie_id) DO UPDATE SET
                position=excluded.position,
                duration=excluded.duration,
                updated_at=excluded.updated_at
        """, (movie_id, position, duration, time.time()))
        con.commit()
    return jsonify({"ok": True})


# ----------------------------- Frontend -----------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>My Movie Library</title>
<style>
:root{color-scheme:dark;--bg:#101318;--panel:#1b2029;--muted:#9ca7b7;--accent:#78a9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f4f6fa;font:16px system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:#101318ed;backdrop-filter:blur(12px);padding:16px max(16px,env(safe-area-inset-left));border-bottom:1px solid #2b3340}
h1{font-size:20px;margin:0 0 12px}.toolbar{display:flex;gap:10px}
input{width:100%;padding:12px 14px;border-radius:10px;border:1px solid #394354;background:#1b2029;color:white;font-size:16px}
main{padding:18px;max-width:1500px;margin:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:16px}
.movie{background:var(--panel);border:1px solid #2a3340;border-radius:12px;overflow:hidden;cursor:pointer;min-width:0}
.movie:active{transform:scale(.99)}.movie img{width:100%;aspect-ratio:16/9;object-fit:cover;background:#242a34;display:block}
.movie-name{padding:10px 11px 4px;font-size:14px;line-height:1.35;min-height:48px;overflow-wrap:anywhere}
.meta{padding:0 11px 11px;color:var(--muted);font-size:12px}
.progress{height:3px;background:#343d4b;margin:0 11px 10px}.progress span{display:block;height:100%;background:var(--accent)}
#playerPanel{display:none;position:fixed;inset:0;z-index:20;background:#080a0e;overflow:auto;padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom)}
.playerTop{display:flex;align-items:center;gap:12px;padding:12px 16px}.playerTop button,.close{border:0;border-radius:9px;background:#252d39;color:white;padding:10px 14px;font-size:15px}
#playingTitle{font-size:15px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
video{display:block;width:100%;max-height:calc(100vh - 100px);background:#000;object-fit:contain}
#status{padding:12px 16px;color:var(--muted);font-size:14px}
.empty{color:var(--muted);padding:30px 0}
@media(min-width:900px){#playerPanel video{max-height:calc(100vh - 120px);height:calc(100vh - 120px)}}
</style>
</head>
<body>
<header><h1>My Movie Library</h1><div class="toolbar"><input id="search" type="search" placeholder="Search movies..." autocomplete="off"></div></header>
<main><div id="count" class="meta"></div><div id="grid" class="grid"></div></main>
<section id="playerPanel">
  <div class="playerTop"><button id="back">← Library</button><div id="playingTitle"></div></div>
  <video id="video" controls playsinline webkit-playsinline preload="metadata"></video>
  <div id="status">Preparing video…</div>
</section>
<script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js"></script>
<script>
const grid=document.getElementById('grid'), search=document.getElementById('search');
const panel=document.getElementById('playerPanel'), video=document.getElementById('video');
const statusEl=document.getElementById('status'), titleEl=document.getElementById('playingTitle');
let movies=[], currentMovie=null, hls=null, pollTimer=null, progressTimer=null, thumbTimer=null;
let lastProgressSave=0, directMkvMovieId=null, mkvFallbackStarted=false;
const isIOS=/iPad|iPhone|iPod/.test(navigator.userAgent)||(navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1);

function humanSize(bytes){if(!bytes)return '';const units=['B','KB','MB','GB','TB'];let i=0,n=bytes;while(n>=1024&&i<units.length-1){n/=1024;i++;}return n.toFixed(i?1:0)+' '+units[i];}
function render(){
 const q=search.value.trim().toLowerCase();
 const shown=movies.filter(m=>m.name.toLowerCase().includes(q)||m.filename.toLowerCase().includes(q));
 document.getElementById('count').textContent=`${shown.length} movie${shown.length===1?'':'s'}`;
 grid.innerHTML='';
 if(!shown.length){grid.innerHTML='<div class="empty">No movies found.</div>';return;}
 for(const m of shown){
  const card=document.createElement('article');card.className='movie';
  const img=document.createElement('img');img.loading='lazy';img.alt='';img.src=`/thumbnail/${m.id}.jpg`;img.dataset.thumb=m.id;
  const name=document.createElement('div');name.className='movie-name';name.textContent=m.name;
  const meta=document.createElement('div');meta.className='meta';meta.textContent=humanSize(m.size);
  const bar=document.createElement('div');bar.className='progress';const fill=document.createElement('span');
  const pct=m.progress&&m.progress.duration>0?Math.min(100,m.progress.position/m.progress.duration*100):0;fill.style.width=pct+'%';bar.append(fill);
  card.append(img,name,meta,bar);card.addEventListener('click',()=>playMovie(m));grid.append(card);
 }
}
async function loadMovies(){
 try{const r=await fetch('/api/movies');const data=await r.json();if(!r.ok)throw new Error(data.error||'Could not load library');movies=data.movies||[];render();}
 catch(e){grid.innerHTML=`<div class="empty">${e.message}</div>`;}
}
search.addEventListener('input',render);

function stopTimers(){clearInterval(pollTimer);clearInterval(progressTimer);clearInterval(thumbTimer);pollTimer=progressTimer=thumbTimer=null;}
function destroyPlayer(){
 stopTimers();if(hls){hls.destroy();hls=null;}video.pause();video.removeAttribute('src');video.load();
}
function closePlayer(){
 if(currentMovie)saveProgress(true);destroyPlayer();panel.style.display='none';currentMovie=null;
}
document.getElementById('back').addEventListener('click',closePlayer);
function showStatus(text){statusEl.textContent=text;}
async function saveProgress(force=false){
 if(!currentMovie||!Number.isFinite(video.currentTime))return;
 const now=Date.now();if(!force&&now-lastProgressSave<10000)return;lastProgressSave=now;
 try{await fetch(`/api/progress/${currentMovie.id}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({position:video.currentTime,duration:video.duration||0})});}catch(e){}
}
function attachSource(url, movie, allowMkvFallback=false){
 directMkvMovieId=allowMkvFallback?movie.id:null;
 if(video.canPlayType('application/vnd.apple.mpegurl')){
   video.src=url;video.load();
 }else if(window.Hls&&Hls.isSupported()){
   hls=new Hls({enableWorker:true,lowLatencyMode:false});
   hls.loadSource(url);hls.attachMedia(video);
   hls.on(Hls.Events.ERROR,(_,data)=>{if(data.fatal)showStatus('Playback error: '+(data.details||data.type));});
 }else{
   showStatus('This browser does not support HLS playback. Try Safari on iPhone/iPad or a current desktop browser.');
 }
 video.addEventListener('loadedmetadata',async function restore(){
   video.removeEventListener('loadedmetadata',restore);
   try{const r=await fetch(`/api/progress/${currentMovie.id}`);const p=await r.json();if(p.position>5&&p.duration>0&&p.position<p.duration-10)video.currentTime=p.position;}catch(e){}
   showStatus('Ready — tap Play to start.');
 });
 video.addEventListener('playing',()=>showStatus('Playing'));
 video.addEventListener('waiting',()=>showStatus('Buffering…'));
}
async function prepareAndPlay(movie, forceHls=false){
 try{
   const response=await fetch('/api/prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({movie_id:movie.id,force_hls:forceHls})});
   const prep=await response.json();if(!response.ok)throw new Error(prep.error||'Could not prepare video');
   if(!currentMovie||currentMovie.id!==movie.id)return;
   if(prep.mode==='direct'){attachSource(prep.url,movie,!forceHls&&/\.mkv$/i.test(movie.filename));return;}
   showStatus('Preparing the first playable segments… You can start watching before the full movie is processed.');
   pollTimer=setInterval(async()=>{
     if(!currentMovie||currentMovie.id!==movie.id){clearInterval(pollTimer);pollTimer=null;return;}
     try{
       const r=await fetch(`/api/status/${movie.id}`);const s=await r.json();
       if(s.ready){clearInterval(pollTimer);pollTimer=null;attachSource(s.url,movie);if(!s.complete)showStatus('Streaming while the remaining video is prepared…');}
       else if(s.state==='error'){clearInterval(pollTimer);pollTimer=null;showStatus('Preparation failed: '+(s.error||'Check the FFmpeg log in the cache folder.'));}
       else showStatus(s.state==='queued'?'Waiting for a media worker…':'Preparing first segments…');
     }catch(e){showStatus('Waiting for server…');}
   },1000);
   progressTimer=setInterval(()=>saveProgress(false),5000);
 }catch(e){if(currentMovie&&currentMovie.id===movie.id)showStatus(e.message);}
}
async function playMovie(movie){
 destroyPlayer();currentMovie=movie;lastProgressSave=0;
 directMkvMovieId=null;mkvFallbackStarted=false;
 panel.style.display='block';titleEl.textContent=movie.name;showStatus('Checking video…');
 const isMkv=/\.mkv$/i.test(movie.filename);
 await prepareAndPlay(movie,isIOS&&isMkv);
}
video.addEventListener('error',()=>{
 if(video.error&&directMkvMovieId&&currentMovie&&currentMovie.id===directMkvMovieId&&!mkvFallbackStarted){
   mkvFallbackStarted=true;directMkvMovieId=null;
   showStatus('MKV playback is unsupported here. Preparing HLS…');
   prepareAndPlay(currentMovie,true);
 }else if(video.error){showStatus('Video error. The source may contain a codec this browser cannot play.');}
});
video.addEventListener('timeupdate',()=>{if(currentMovie&&video.currentTime>0)saveProgress(false);});
video.addEventListener('pause',()=>saveProgress(true));
video.addEventListener('ended',()=>saveProgress(true));
document.addEventListener('visibilitychange',()=>{if(document.hidden)saveProgress(true);});

// Retry thumbnails in the background without blocking the movie grid.
thumbTimer=setInterval(()=>{
 document.querySelectorAll('img[data-thumb]').forEach(img=>{
   if(!img.dataset.ready){const id=img.dataset.thumb;img.src=`/thumbnail/${id}.jpg?t=${Date.now()}`;img.onload=()=>{img.dataset.ready='1';};}
 });
},12000);
window.addEventListener('pagehide',()=>saveProgress(true));
loadMovies();
</script>
</body>
</html>"""


if __name__ == "__main__":
    if not MOVIE_DIR.exists():
        raise SystemExit(f"Movie folder does not exist: {MOVIE_DIR}")
    if not ffmpeg_available():
        log.warning("FFmpeg/FFprobe not found. Set FFMPEG_PATH and FFPROBE_PATH or add FFmpeg to PATH.")
    log.info("Movie folder: %s", MOVIE_DIR)
    log.info("Cache folder: %s", CACHE_DIR)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True, debug=False)
