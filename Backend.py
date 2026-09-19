import os
import sys
import json
import shutil
import logging
import subprocess
import re
import sqlite3
import argparse
import io
from pathlib import Path
from datetime import datetime

# ==============================================================================
# DYNAMIC ENVIRONMENT & DLL SETUP (PORTABLE, NO HARDCODED USERNAME)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
USER_HOME = Path.home()

# Locate venv dynamically: check script directory first, then user home
VENV_DIR = None
for candidate in [SCRIPT_DIR / ".venv", SCRIPT_DIR / "venv", USER_HOME / ".venv"]:
    if candidate.exists() and (candidate / "Scripts").exists():
        VENV_DIR = candidate
        break

# Dynamic CUDA Toolkit detection
CUDA_DIR = None
cuda_env = os.environ.get("CUDA_PATH")
if cuda_env and Path(cuda_env).exists():
    CUDA_DIR = Path(cuda_env)
else:
    cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_base.exists():
        versions = sorted(list(cuda_base.glob("v*")), reverse=True)
        if versions:
            CUDA_DIR = versions[0]

if sys.platform == "win32":
    # Ensure standard I/O handles utf-8 cleanly
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    
    if VENV_DIR:
        torch_lib = VENV_DIR / "Lib" / "site-packages" / "torch" / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib.resolve()))
            
    if CUDA_DIR and (CUDA_DIR / "bin").exists():
        os.add_dll_directory(str((CUDA_DIR / "bin").resolve()))

# ==============================================================================
# 1. LOGGING SETUP & CRASH RECOVERY INSPECTION
# ==============================================================================
LOG_FILE = Path("reconstruct_pipeline.log")
log_file_existed = LOG_FILE.exists()

def check_previous_session_crash():
    """Checks if the previous run terminated prematurely and returns session info."""
    if not LOG_FILE.exists():
        return None
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None

    if not lines:
        return None

    start_indices = [i for i, line in enumerate(lines) if "=== Reconstruction Program Started ===" in line]
    if not start_indices:
        return None

    last_session_lines = lines[start_indices[-1]:]
    is_completed = any("=== Entire Pipeline Complete." in line for line in last_session_lines)
    if is_completed:
        return None

    video_path = None
    frames_extracted = None
    for line in last_session_lines:
        vid_match = re.search(r"User selected video path:\s*(.*)", line)
        if vid_match:
            video_path = vid_match.group(1).strip()
        frame_match = re.search(r"Successfully extracted (\d+) frames\.", line)
        if frame_match:
            frames_extracted = int(frame_match.group(1))

    return {
        "video_path": video_path,
        "frames_extracted": frames_extracted,
        "raw_lines": last_session_lines
    }

crash_state = check_previous_session_crash()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

if not log_file_existed:
    logging.info("Log file was missing. Created a new log file.")
logging.info("=== Reconstruction Program Started ===")

# ==============================================================================
# CONFIGURATION MANAGEMENT (SAFE RUNTIME CONFIG)
# ==============================================================================
CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG = {
    "extraction_mode": "all",  # "all", "fps", "every_nth"
    "fps_value": 2.0,
    "nth_frame_value": 5,
    "sequential_overlap": 15,
    "max_num_features": 8192,
    "max_image_size": 4096,
    "dense_max_image_size": 2000,
    "generate_gsplat": True,
    "gsplat_iterations": 30000,
    "export_gsplat_mesh": True,
    "generate_dense": False,
    "generate_mesh": False,
    "output_scenes_dir": "SCENES",
    "output_frames_dir": "VIDEOS",
    "camera_model": "SIMPLE_RADIAL",
    "vis": "viewer",
    "viewer_quit_on_train_completion": True,
    "downscale_factor": 1
}

def load_or_create_config() -> dict:
    if not CONFIG_FILE.exists():
        logging.warning("Config file not found. Generating default 'config.json'.")
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:  # <-- Changed to utf-8-sig
            user_config = json.load(f)
        logging.info(f"Loaded existing configuration from {CONFIG_FILE.name}")
        return {**DEFAULT_CONFIG, **user_config}
    except Exception as e:
        logging.error(f"Failed to parse config file: {e}. Falling back to default settings.")
        return DEFAULT_CONFIG

config = load_or_create_config()

# CLI Options for server/interactive usage
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="Full path to target video")
parser.add_argument("--resume", action="store_true", help="Force resume from previous crash")
parser.add_argument("--non-interactive", action="store_true", help="Do not prompt stdin")
args = parser.parse_args()

# ==============================================================================
# HARDWARE & DEPENDENCY CHECKS
# ==============================================================================
def check_cuda() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            logging.info("CUDA GPU detected via nvidia-smi.")
            return True
    except FileNotFoundError:
        pass
    logging.warning("CUDA/NVIDIA GPU not detected. Falling back to CPU.")
    return False

def check_dependencies():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logging.critical("FFMPEG was not found in the system PATH. Terminating program.")
        sys.exit(1)
    logging.info(f"FFmpeg found at: {ffmpeg_path}")

    colmap_path = shutil.which("colmap") or shutil.which("colmap.bat")
    if not colmap_path:
        logging.critical("COLMAP was not found in the system PATH. Terminating program.")
        sys.exit(1)
    logging.info(f"COLMAP found at: {colmap_path}")
    return ffmpeg_path, colmap_path

has_cuda = check_cuda()
ffmpeg_bin, colmap_bin = check_dependencies()

# ==============================================================================
# RECOVERY & VIDEO SELECTION
# ==============================================================================
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm", ".ts", ".mts"}
selected_video_path = None
resuming_session = False

# 1. Check CLI force resume
if args.resume and crash_state and crash_state.get("video_path"):
    candidate = Path(crash_state["video_path"])
    if candidate.is_file():
        selected_video_path = candidate
        resuming_session = True
        logging.info(f"CLI forced resume for: {selected_video_path}")

# 2. Check CLI video arg
if not selected_video_path and args.video:
    candidate = Path(args.video.strip(' "\''))
    if candidate.is_file():
        selected_video_path = candidate
        logging.info(f"User specified video via CLI: {selected_video_path}")

# 3. Interactive crash detection fallback
if not selected_video_path and crash_state and crash_state.get("video_path") and not args.non_interactive:
    prev_path = Path(crash_state["video_path"])
    if prev_path.is_file():
        logging.warning(f"Unfinished pipeline run detected for: {prev_path.resolve()}")
        choice = input(f"\n[!] Previous run for '{prev_path.name}' did not finish.\n    Resume where you left off? (Y/N): ").strip().lower()
        if choice in ("y", "yes"):
            selected_video_path = prev_path
            resuming_session = True
            logging.info(f"User elected to resume previous session for: {selected_video_path}")

# 4. Standard prompt
if not selected_video_path:
    if args.non_interactive:
        logging.critical("No video provided in non-interactive mode. Exiting.")
        sys.exit(1)
    user_input = input("\nEnter the full file path to your video: ").strip(' "\'')
    selected_video_path = Path(user_input)

if not selected_video_path.is_file():
    logging.critical(f"File not found: {selected_video_path}")
    sys.exit(1)

if selected_video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
    logging.critical(f"Unsupported file format '{selected_video_path.suffix}'. Terminating.")
    sys.exit(1)

logging.info(f"User selected video path: {selected_video_path}")

# ==============================================================================
# DIRECTORY STRUCTURE SETUP & MANIFEST PERSISTENCE
# ==============================================================================
video_name = selected_video_path.stem
frames_base_dir = Path(config.get("output_frames_dir", "VIDEOS")) / video_name
scenes_base_dir = Path(config.get("output_scenes_dir", "SCENES")) / video_name
sparse_dir = scenes_base_dir / "sparse"
primary_sparse_model = sparse_dir / "0"
dense_dir = scenes_base_dir / "dense"
database_path = scenes_base_dir / "database.db"
manifest_path = Path(config.get("output_scenes_dir", "SCENES")) / "scenes_manifest.json"

frames_base_dir.mkdir(parents=True, exist_ok=True)
sparse_dir.mkdir(parents=True, exist_ok=True)
dense_dir.mkdir(parents=True, exist_ok=True)

def update_manifest(status: str, extra: dict = None):
    """Safely records pipeline state to scenes_manifest.json for the server API."""
    try:
        data = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        entry = data.get(video_name, {
            "name": video_name,
            "video_path": str(selected_video_path.resolve()),
            "frames_dir": str(frames_base_dir.resolve()),
            "scene_dir": str(scenes_base_dir.resolve()),
            "created_at": datetime.now().isoformat()
        })
        entry["last_updated"] = datetime.now().isoformat()
        entry["status"] = status
        
        mesh_dir = scenes_base_dir / "mesh"
        entry["has_colored_mesh"] = (mesh_dir / "colored.obj").exists() or (mesh_dir / "colored.ply").exists()
        entry["has_gsplat_ply"] = any(mesh_dir.glob("*.ply")) if mesh_dir.exists() else False
        entry["sparse_reconstructed"] = (primary_sparse_model / "cameras.bin").exists() or (primary_sparse_model / "cameras.txt").exists()
        
        sample_frame = next(frames_base_dir.glob("*.jpg"), None)
        if sample_frame:
            entry["thumbnail"] = str(sample_frame.resolve())

        if extra:
            entry.update(extra)

        data[video_name] = entry
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.warning(f"Manifest update ignored: {e}")

update_manifest("RUNNING")

# ==============================================================================
# PIPELINE EXECUTION HELPER (ORIGINAL LIVE CHUNK STREAMING)
# ==============================================================================
def get_msvc_env():
    """Locate vcvars64.bat and extract full MSVC compiler environment variables."""
    vs_candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    
    vcvars_path = next((p for p in vs_candidates if Path(p).exists()), None)
    if not vcvars_path:
        return os.environ.copy()

    cmd = f'"{vcvars_path}" && set'
    out = subprocess.check_output(cmd, shell=True, text=True, errors="replace")
    
    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.upper()] = v
    return env

import threading

def run_logged_process(cmd_args: list, step_name: str, exit_on_fail: bool = True):
    logging.info(f"Running step: {step_name}")
    
    sub_env = get_msvc_env()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    sub_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sub_env["CFLAGS"] = f"{sub_env.get('CFLAGS', '')} /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING /Zc:preprocessor"
    sub_env["CXXFLAGS"] = f"{sub_env.get('CXXFLAGS', '')} /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING /Zc:preprocessor"
    sub_env["TORCH_CUDA_ARCH_LIST"] = "8.9"
    sub_env["NERFSTUDIO_INTERACTIVE"] = "0"
    sub_env["TYPER_INTERACTIVE"] = "0"
    
    extra_paths = []
    if VENV_DIR:
        extra_paths.append(str((VENV_DIR / "Scripts").resolve()))
        torch_lib = VENV_DIR / "Lib" / "site-packages" / "torch" / "lib"
        if torch_lib.exists():
            extra_paths.append(str(torch_lib.resolve()))
    if CUDA_DIR:
        extra_paths.append(str((CUDA_DIR / "bin").resolve()))
        
    sub_env["PATH"] = ";".join(extra_paths) + ";" + sub_env.get("PATH", "")

    # 1. Spawn process with standard input enabled
    proc = subprocess.Popen(
        cmd_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=sub_env
    )

    # 2. Feeder thread to automatically reply 'y' to interactive prompts (floor rounding, etc.)
    def feed_stdin(target_proc):
        import time
        for _ in range(5):
            time.sleep(1.0)
            if target_proc.poll() is not None:
                break
            try:
                if target_proc.stdin and not target_proc.stdin.closed:
                    target_proc.stdin.write("y\n")
                    target_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                break

    feeder_thread = threading.Thread(target=feed_stdin, args=(proc,), daemon=True)
    feeder_thread.start()

    # 3. Stream output live
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        while True:
            chunk = proc.stdout.read(64)
            if not chunk and proc.poll() is not None:
                break
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                f.write(chunk)
                f.flush()
            
    proc.wait()
    if proc.returncode != 0:
        error_msg = f"{step_name} failed with exit code {proc.returncode}."
        logging.error(error_msg)
        update_manifest("FAILED", {"error_stage": step_name})
        if exit_on_fail:
            sys.exit(1)
        else:
            raise RuntimeError(error_msg)

# ==============================================================================
# STAGE 1: FRAME EXTRACTION
# ==============================================================================
skip_extraction = False

if resuming_session:
    existing_frames = list(frames_base_dir.glob("*.jpg"))
    logged_count = crash_state.get("frames_extracted") if crash_state else None
    
    if logged_count and len(existing_frames) == logged_count and logged_count > 0:
        logging.info(f"Resumption Verified: All {logged_count} frames present. Skipping extraction.")
        skip_extraction = True
    elif len(existing_frames) > 0 and not logged_count:
        logging.info(f"Found {len(existing_frames)} existing frames in folder. Skipping extraction.")
        skip_extraction = True

if not skip_extraction:
    mode = config.get("extraction_mode", "all")
    ffmpeg_filter = []

    if mode == "fps":
        fps = float(config.get("fps_value", 2.0))
        ffmpeg_filter = ["-vf", f"fps={fps}"]
        logging.info(f"Extraction Mode: FPS ({fps} frames/sec) to '{frames_base_dir}'")
    elif mode == "every_nth":
        n = int(config.get("nth_frame_value", 5))
        ffmpeg_filter = ["-vf", f"select=not(mod(n\\,{n}))", "-vsync", "vfr"]
        logging.info(f"Extraction Mode: N-th Frame (every {n}th frame) to '{frames_base_dir}'")
    else:
        logging.info(f"Extraction Mode: ALL frames to '{frames_base_dir}'")

    output_pattern = str(frames_base_dir / "frame_%06d.jpg")
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-stats",
        "-i", str(selected_video_path),
        *ffmpeg_filter,
        "-qscale:v", "2",
        output_pattern
    ]

    logging.info(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
    ff_proc = subprocess.run(ffmpeg_cmd)
    if ff_proc.returncode != 0:
        logging.critical("FFmpeg frame extraction failed. Terminating.")
        sys.exit(1)

    extracted_images = list(frames_base_dir.glob("*.jpg"))
    if not extracted_images:
        logging.critical("No frames were extracted. Terminating.")
        sys.exit(1)
    logging.info(f"Successfully extracted {len(extracted_images)} frames.")
    update_manifest("FRAMES_EXTRACTED", {"total_frames": len(extracted_images)})

# ==============================================================================
# STAGE 2: SIFT FEATURE EXTRACTION
# ==============================================================================
cuda_val = "1" if has_cuda else "0"
cpu_threads = str(os.cpu_count() or 4)

skip_sift = False
if database_path.exists():
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM keypoints;")
        keypoints_count = cursor.fetchone()[0]
        conn.close()
        if keypoints_count > 0:
            logging.info(f"Existing keypoints ({keypoints_count}) found in database.db. Skipping SIFT extraction.")
            skip_sift = True
    except Exception:
        pass

if not skip_sift:
    camera_model = str(config.get("camera_model", "SIMPLE_RADIAL"))
    max_img_size = str(int(config.get("max_image_size", 4096)))
    max_features = str(int(config.get("max_num_features", 8192)))

    run_logged_process([
        colmap_bin, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(frames_base_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
        "--FeatureExtraction.use_gpu", cuda_val,
        "--FeatureExtraction.max_image_size", max_img_size,
        "--SiftExtraction.max_num_features", max_features
    ], "feature_extractor")
    update_manifest("SIFT_COMPLETE")

# ==============================================================================
# STAGE 3: FEATURE MATCHING
# ==============================================================================
skip_matching = False
if database_path.exists() and skip_sift:
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM two_view_geometries WHERE rows > 0;")
        matches_count = cursor.fetchone()[0]
        conn.close()
        if matches_count > 0:
            logging.info(f"Existing verified matches ({matches_count}) found in database.db. Skipping matching.")
            skip_matching = True
    except Exception:
        pass

if not skip_matching:
    overlap = str(int(config.get("sequential_overlap", 15)))
    run_logged_process([
        colmap_bin, "sequential_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", cuda_val,
        "--SequentialMatching.overlap", overlap
    ], "sequential_matcher")
    update_manifest("MATCHING_COMPLETE")

# ==============================================================================
# STAGE 4: SPARSE RECONSTRUCTION (3-TIER CASCADE: CASPAR GPU -> CERES GPU -> CPU)
# ==============================================================================
has_sparse_model = (
    primary_sparse_model.exists() and 
    (
        (primary_sparse_model / "cameras.bin").exists() or 
        (primary_sparse_model / "cameras.txt").exists()
    )
)

if has_sparse_model:
    logging.info("Existing sparse reconstruction ('sparse/0') detected. Skipping mapper.")
else:
    mapper_base_cmd = [
        colmap_bin, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(frames_base_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.num_threads", cpu_threads
    ]

    if has_cuda:
        logging.info("Attempting Mapper with Caspar GPU backend...")
        caspar_cmd = mapper_base_cmd + [
            "--Mapper.ba_use_gpu", "1",
            "--Mapper.ba_local_backend", "CASPAR",
            "--Mapper.ba_global_backend", "CASPAR"
        ]
        
        try:
            run_logged_process(caspar_cmd, "mapper_caspar", exit_on_fail=False)
        except RuntimeError:
            logging.warning("Caspar backend failed. Retrying with Ceres CUDA GPU backend...")
            ceres_gpu_cmd = mapper_base_cmd + [
                "--Mapper.ba_use_gpu", "1",
                "--Mapper.ba_local_backend", "CERES",
                "--Mapper.ba_global_backend", "CERES"
            ]
            
            try:
                run_logged_process(ceres_gpu_cmd, "mapper_ceres_gpu", exit_on_fail=False)
            except RuntimeError:
                logging.warning("Ceres GPU solver failed. Falling back to strict CPU default...")
                logging.info("Running Incremental Mapper on CPU (Ceres solver with multi-threading)...")
                mapper_cmd = [
                    colmap_bin, "mapper",
                    "--database_path", str(database_path),
                    "--image_path", str(frames_base_dir),
                    "--output_path", str(sparse_dir),
                    "--Mapper.num_threads", cpu_threads,
                    "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "25000"
                ]
                run_logged_process(mapper_cmd, "mapper")
    else:
        logging.warning("No CUDA device detected. Running mapper strictly on CPU...")
        run_logged_process(mapper_base_cmd, "mapper_cpu")

    update_manifest("SPARSE_COMPLETE")

# ==============================================================================
# STAGE 5: 3D GAUSSIAN SPLATTING -> PLY -> 3D MESH (OBJ + PLY)
# ==============================================================================
gsplat_output_dir = scenes_base_dir / "gsplat"

if config.get("generate_gsplat", True) and primary_sparse_model.exists():
    if not has_cuda:
        logging.warning("GSPLAT requires a CUDA GPU. Skipping Stage 5.")
    else:
        logging.info("Starting Stage 5: 3D Gaussian Splatting (Nerfstudio splatfacto)...")
        gsplat_output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_images_dir = scenes_base_dir / "images"
        if not scene_images_dir.exists():
            try:
                os.symlink(frames_base_dir.resolve(), scene_images_dir.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(frames_base_dir, scene_images_dir)

        iterations = str(int(config.get("gsplat_iterations", 30000)))
        
        python_bin = sys.executable
        if VENV_DIR and (VENV_DIR / "Scripts" / "python.exe").exists():
            python_bin = str((VENV_DIR / "Scripts" / "python.exe").resolve())

        vis_mode = str(config.get("vis", "viewer"))
        quit_viewer = str(config.get("viewer_quit_on_train_completion", True))

        ns_cmd = [
            python_bin, "-m", "nerfstudio.scripts.train", "splatfacto",
            "--data", str(scenes_base_dir),
            "--output-dir", str(gsplat_output_dir),
            "--max-num-iterations", iterations,
            "--vis", vis_mode,
            "--viewer.quit-on-train-completion", quit_viewer,
            "colmap",
            "--colmap-path", "sparse/0",
            "--images-path", "images",
            "--downscale-factor", str(config.get("downscale_factor", 1))
        ]

        try:
            config_candidates = list(gsplat_output_dir.glob("**/config.yml"))
            if not config_candidates:
                run_logged_process(ns_cmd, "splatfacto_trainer")
                logging.info(f"GSPLAT training complete. Model saved to: {gsplat_output_dir}")
                config_candidates = list(gsplat_output_dir.glob("**/config.yml"))

            # ==========================================================
            # STAGE 5B: EXPORT PLY & GENERATE 3D MESH
            # ==========================================================
            if config.get("export_gsplat_mesh", True):
                if not config_candidates:
                    logging.error("Could not find 'config.yml' for export.")
                else:
                    latest_config = max(config_candidates, key=lambda p: p.stat().st_mtime)
                    mesh_output_dir = scenes_base_dir / "mesh"
                    mesh_output_dir.mkdir(parents=True, exist_ok=True)

                    logging.info("Exporting Gaussian Splat directly to standard .ply...")
                    export_cmd = [
                        python_bin, "-m", "nerfstudio.scripts.exporter", "gaussian-splat",
                        "--load-config", str(latest_config.resolve()),
                        "--output-dir", str(mesh_output_dir.resolve()),
                    ]
                    run_logged_process(export_cmd, "gsplat_ply_exporter")

                    exported_plys = list(mesh_output_dir.glob("*.ply"))
                    if not exported_plys:
                        logging.error("No PLY file found in export folder.")
                    else:
                        target_ply = max(exported_plys, key=lambda p: p.stat().st_mtime)

                        # Create canonical splat.ply for the frontend
                        splat_canonical = mesh_output_dir / "splat.ply"
                        if target_ply.resolve() != splat_canonical.resolve():
                            shutil.copyfile(target_ply, splat_canonical)

                        try:
                            import open3d as o3d
                            import numpy as np
                            import copy
                            from plyfile import PlyData

                            logging.info(f"Reading Gaussian data from: {splat_canonical.resolve()}")
                            plydata = PlyData.read(str(splat_canonical))
                            v = plydata['vertex']

                            pts = np.vstack([v['x'], v['y'], v['z']]).T

                            # Spherical Harmonics DC base -> RGB colors
                            if 'f_dc_0' in v:
                                SH_C0 = 0.28209479177387814
                                rgb = np.vstack([
                                    0.5 + SH_C0 * v['f_dc_0'],
                                    0.5 + SH_C0 * v['f_dc_1'],
                                    0.5 + SH_C0 * v['f_dc_2']
                                ]).T
                                rgb = np.clip(rgb, 0.0, 1.0)
                            elif 'red' in v:
                                rgb = np.vstack([v['red'], v['green'], v['blue']]).T / 255.0
                            else:
                                rgb = np.ones_like(pts) * 0.7

                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pts)
                            pcd.colors = o3d.utility.Vector3dVector(rgb)

                            logging.info(f"Estimating normals on {len(pcd.points)} points...")
                            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                            pcd.orient_normals_consistent_tangent_plane(50)

                            logging.info("Running Poisson Surface Reconstruction...")
                            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

                            densities_arr = np.asarray(densities)
                            vertices_to_remove = densities_arr < np.quantile(densities_arr, 0.05)
                            mesh.remove_vertices_by_mask(vertices_to_remove)

                            # Save Uncolored OBJ
                            uncolored_mesh = copy.deepcopy(mesh)
                            uncolored_mesh.vertex_colors = o3d.utility.Vector3dVector([])
                            path_uncolored_obj = mesh_output_dir / "uncolored.obj"
                            o3d.io.write_triangle_mesh(str(path_uncolored_obj), uncolored_mesh)

                            # Transfer vertex colors from PCD to Mesh
                            logging.info("Transferring colors to mesh vertices...")
                            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                            mesh_vertices = np.asarray(mesh.vertices)
                            pcd_colors = np.asarray(pcd.colors)
                            
                            vertex_colors = np.zeros_like(mesh_vertices)
                            for i, vertex in enumerate(mesh_vertices):
                                _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
                                vertex_colors[i] = pcd_colors[idx[0]]

                            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

                            path_colored_obj = mesh_output_dir / "colored.obj"
                            path_colored_ply = mesh_output_dir / "colored.ply"

                            o3d.io.write_triangle_mesh(str(path_colored_obj), mesh)
                            o3d.io.write_triangle_mesh(str(path_colored_ply), mesh)

                            logging.info(f"Colored 3D model saved: {path_colored_obj.resolve()}")

                        except Exception as inner_err:
                            logging.error(f"Meshing failed: {inner_err}", exc_info=True)

        except Exception as e:
            logging.error(f"GSPLAT training/meshing pipeline failed: {e}")

# ==============================================================================
# STAGE 6 (OPTIONAL): TRADITIONAL DENSE RECONSTRUCTION (COLMAP MVS)
# ==============================================================================
if config.get("generate_dense", False) and primary_sparse_model.exists():
    if not has_cuda:
        logging.warning("COLMAP PatchMatch Stereo requires CUDA. Skipping Stage 6.")
    else:
        dense_images_dir = dense_dir / "images"
        dense_sparse_dir = dense_dir / "sparse"
        stereo_depth_maps_dir = dense_dir / "stereo" / "depth_maps"
        dense_ply_path = dense_dir / "fused.ply"

        has_undistorted = (
            dense_images_dir.exists() and 
            dense_sparse_dir.exists() and 
            any(dense_images_dir.glob("*.jpg"))
        )
        
        if not has_undistorted:
            logging.info("Starting Stage 6 Step 1/3: image_undistorter...")
            run_logged_process([
                colmap_bin, "image_undistorter",
                "--image_path", str(frames_base_dir),
                "--input_path", str(primary_sparse_model),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP",
                "--max_image_size", str(int(config.get("dense_max_image_size", 2000)))
            ], "image_undistorter")

        has_depth_maps = (
            stereo_depth_maps_dir.exists() and 
            any(stereo_depth_maps_dir.glob("*.photometric.bin"))
        )
        
        if not has_depth_maps:
            logging.info("Starting Stage 6 Step 2/3: patch_match_stereo...")
            run_logged_process([
                colmap_bin, "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.gpu_index", "0"
            ], "patch_match_stereo")

        if not (dense_ply_path.exists() and dense_ply_path.stat().st_size > 0):
            logging.info("Starting Stage 6 Step 3/3: stereo_fusion (Generating fused.ply)...")
            run_logged_process([
                colmap_bin, "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", str(dense_ply_path)
            ], "stereo_fusion")

        if config.get("generate_mesh", False):
            mesh_path = dense_dir / "meshed-poisson.ply"
            if not mesh_path.exists():
                logging.info("Generating Poisson surface mesh...")
                run_logged_process([
                    colmap_bin, "poisson_mesher",
                    "--input_path", str(dense_ply_path),
                    "--output_path", str(mesh_path),
                    "--PoissonMeshing.trim", "5"
                ], "poisson_mesher")

logging.info(f"=== Entire Pipeline Complete. Results saved to: {scenes_base_dir.resolve()} ===")
update_manifest("COMPLETED")