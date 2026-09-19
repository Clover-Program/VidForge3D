# Scenes & Output Directory

Default workspace used by VidForge3D to stage pipeline intermediate files, reconstructions, and final assets. If this directory does not exist, the backend creates it automatically on runtime.

### Staged Artifacts
Each processed scene populates artifacts here, including:
- **Feature & Matching Databases:** Feature extractions, match files, and COLMAP databases.
- **Reconstruction Models:** Sparse point clouds, camera poses, trajectories, and dense point clouds.
- **Final Outputs:** Trained Gaussian splats, checkpoints, export meshes, and runtime logs.

> **Storage Notice:** Dense point clouds, checkpoint archives, and matching databases scale quickly across runs. Ensure ample disk space on the target volume.

### Configuration
- **Custom Paths:** The global scenes directory can be configured in `config.json`.
- **Fallbacks:** If `config.json` is missing or invalid, `Backend.py` regenerates or falls back to `./SCENES`.
- **Note:** Any reference paths in `Server.ps1` are currently hardcoded to default locations.
