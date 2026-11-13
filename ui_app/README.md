# Forensic Search UI

A web page for the forensic search app. Type a prompt, press Run search, get the clips ranked by score.
Past searches are listed under "Recent searches" (read from the outputs/ folder).

## Run (on the GPU machine, from the repo folder)

    conda activate forensic
    pip install fastapi uvicorn
    cd /mnt/storage/Arka/video_forensic_search
    python -m uvicorn ui_app.server:app --host 0.0.0.0 --port 8000

Open http://localhost:8000 (or http://<gpu-machine-ip>:8000 from another PC on the same network).

## How it works
- The page calls the server (ui_app/server.py), which runs `run_search.py` once per video, one search at a time.
- Results are read from outputs/<video>_full/results/<name>/ (clips and review_clips, clips_index.csv, final_results.json).
- "All Cameras" = the videos in /mnt/storage/ashutosh_stuff/recording (searched with config_indoor.yaml).
  Other ingested videos (e.g. test_vid2) are listed separately. Edit the settings at the top of server.py to change this.
- Each search is saved as outputs/<video>_full/results/ui_<date>_<time>/ and its log in outputs/_ui_logs/.
