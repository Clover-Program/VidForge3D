# Video Input Directory

Default directory used by VidForge3D for reading raw source videos and extracting frame sequences. If this directory does not exist, the backend will generate it automatically.

> **Storage Notice:** Extracting high-resolution frame datasets can rapidly consume several gigabytes of disk space. Make sure the host drive has adequate free storage before starting a pipeline run.

### Configuration
* **Custom Paths:** You can customize the video and output directory paths directly inside `config.json`.
* **Fallback Behavior:** If `config.json` is deleted, missing, or fails to parse, `Backend.py` will fall back to generating default `VIDEOS` and `SCENES` directories.
* **Note on Server Script:** Path arguments within `Server.ps1` are currently hardcoded, so changing paths in `config.json` may require updating `Server.ps1` accordingly.
