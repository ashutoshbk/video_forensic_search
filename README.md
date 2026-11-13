# Video Forensic Search

Search long CCTV / video footage with a one-line text description (and optionally a reference photo) and get back short video clips of the people or vehicles that match.

**Author:** Arkasmit15

Example queries: *"a person with tattoo"*, *"a man in a red jacket carrying a bag"*, *"blue truck"*.

---

## How it works

The work is split into two stages so the slow part is done **once per video** and every new query is fast.

**Stage 1 – Ingest (`run_ingest.py`, once per video)**
1. YOLO11 detects and tracks people and vehicles (every `stride`-th frame).
2. Three crops are saved for every detection (full, upper part, middle zone).
3. OpenCLIP (ViT-L-14) turns every crop into a vector. Everything is cached on disk.
4. The ingest is **resumable**: if it stops, run it again and it continues where it stopped.

**Stage 2 – Search (`run_search.py`, once per query)**
1. Qwen2-VL reads the one-line query and plans a set of yes/no checks.
2. OpenCLIP ranks the cached tracks against the text (and the reference photo, if given). A two-gate threshold then makes the shortlist (see *Choosing which tracks go to Qwen*).
3. Qwen2-VL answers the checks on the best crops of every shortlisted track (Yes / No / Unclear).
4. A decision step labels each track **MATCH** (confirmed) or **REVIEW** (needs a human look).
5. Clips are cut for confirmed matches (and a separate review folder), with a box drawn on the matching person.

---

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA (strongly recommended; `device: "cpu"` works for testing only and is very slow)
- Enough GPU memory for Qwen2-VL-7B (about 16 GB in fp16; set `qwen_4bit: true` for one small GPU)
- `ffmpeg` is not required; clips are cut with OpenCV

## Installation

```bash
git clone https://github.com/ashutoshbk/video_forensic_search.git
cd video_forensic_search
git checkout arkasmit/working-code

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

Model weights (YOLO, OpenCLIP, Qwen2-VL) are downloaded automatically on the first run.

---

## Quick start

**1. Put a video in `videos/` and set it in `config.yaml`** (or pass it on the command line).

**2. Ingest (a short first test of 5 minutes):**

```bash
python run_ingest.py --video videos/test_vid2.mp4 --max-seconds 300
```

For the whole video, leave out `--max-seconds`.

**3. Search:**

```bash
# from a query file
python run_search.py --query query.yaml

# several queries in one go (Qwen is loaded only once)
python run_search.py --query q1.yaml q2.yaml

# a query written on the command line
python run_search.py --text "a person with tattoo" --name tattoo --start 0 --end 300

# a vehicle search with a reference photo
python run_search.py --text "blue truck" --target vehicle --reference photos/truck.jpg

# several videos in ONE process (CLIP and Qwen are loaded only once)
python run_search.py \
  --pair config_indoor.yaml videos/cam101.mp4 \
  --pair config_indoor.yaml videos/cam201.mp4 \
  --text "a man with a grey backpack" --name backpack --target person
```

A video with nothing to check (no detections of the target, or no track above the floor) is **skipped** without an error, and the run continues with the next video.

> Use the same `--video` / `--max-seconds` (or the same `config.yaml`) for ingest and search so the right cache is found.

---

## Choosing which tracks go to Qwen

Qwen is the slow step, so only a shortlist of tracks is sent to it. The shortlist is built by two gates, with the floor and the cap applied to everything:

| Setting | Meaning |
|---|---|
| `z_min` | **Gate 1.** Send every track whose score is at least this. The score is a robust z-score *within the same video*, so it means "better than the rest of this video", not "a good match overall". `null` = old behaviour (plain top-`topk`). |
| `min_tracks` | **Gate 2.** If gate 1 sends fewer than this many tracks, add the next-best tracks until this many are reached. |
| `z_pad` | Gate 2 only adds tracks whose score is at least this (the lower band `z_pad <= z < z_min`). |
| `abs_floor_person`, `abs_floor_vehicle` | **Absolute floor** on the best raw OpenCLIP similarity of a track. Applied in both gates, so a track below it never takes a slot. If no track passes, the video is skipped. `abs_floor` is the fallback for both. |
| `topk` | Safety cap on the number of tracks sent per video. |

Values tested on the indoor cameras (person prompts): `z_min: 2.0`, `min_tracks: 20`, `z_pad: 0.75`, `abs_floor_person: 0.20`, `topk: 100`. On these cameras it cut a 12-video search from about 11 minutes to about 7 minutes with the same matches found.

Notes from the calibration:
- The raw similarity range depends on the prompt type. About 0.20 worked for person-clothing prompts, and a first check on one vehicle prompt suggests about 0.24. Check a floor on your own prompts before you rely on it.
- In busy videos gate 1 can already fill up, so gate 2 never runs. A real match with a score between `z_pad` and `z_min` is then not sent. Lower `z_min` (and raise `topk`) if you need to catch these.
- `shortlist_scores.csv` is written for every run in `results/<name>/` (track, rank, score, `txt_max`, sent). Use it to check a setting, and `calib.py <run name>` to print the `txt_max` of the confirmed matches.

## Run log (`run_log.xlsx`)

Every `run_search.py` run appends one row per query to `run_log.xlsx` in the project folder (created on the first run):

| Column | Content |
|---|---|
| Sl no, Date, Run name, Query | Row number, time, run name, query text |
| Videos targeted | Number of videos in the run |
| Total video duration (h:mm) | Combined duration of those videos |
| Tracks extracted (after filters) | Combined tracks before the shortlist |
| Tracks sent to Qwen | Combined tracks that passed the gates |
| Qwen check time (s) | Combined time of the Qwen checks |
| Matches, Total run time (s) | Confirmed matches and wall-clock time |
| Settings | The threshold values used in the run |

Close the file before a run ends, or open a copy, otherwise that row cannot be saved (a `[runlog]` warning is printed; the search itself is not affected).

## Web UI

`ui_app2/` is a small web page that starts a search over several cameras in one process and shows the clips. Start it with:

```bash
python -m uvicorn ui_app2.server:app --host 0.0.0.0 --port 8000
```

Then open `http://<server-ip>:8000`. See `ui_app2/README.md` for details. `ui_app/` is the older version. Videos in the indoor folder use `config_indoor.yaml`.

## Writing a query

A query is a small YAML file (see `query.yaml`). Only `text` is required.

| Key | Meaning |
|---|---|
| `text` | The description. Qwen plans the yes/no checks from it. |
| `name` | Results go to `<run folder>/results/<name>/` |
| `target` | `person` or `vehicle` |
| `search_start`, `search_end` | Only search this window of the video, in seconds (`null` = to the end) |
| `filter_static` | `true` drops boxes that never move (statues) |
| `filter_fixed` | `true` drops boxes that reappear at the same place (posters) |
| `reference` | Path to a reference photo, or `null` for text-only search |
| `require_reference` | `true` stops the run if the reference photo is missing |

## Configuration

All settings are in `config.yaml` with a comment on each one. The most useful:

| Setting | Meaning |
|---|---|
| `video`, `max_seconds` | Which video, and how much of it (`null` = all) |
| `device` | `cuda:0` or `cpu` |
| `stride` | Process every Nth frame. Higher = faster but may miss short appearances |
| `ingest_classes` | `[0]` = people only (faster); `[0, 2, 3, 5, 7]` = people and vehicles |
| `topk`, `z_min`, `min_tracks`, `z_pad`, `abs_floor_person` | Which tracks are sent to Qwen (see the section above) |
| `match_t`, `review_min` | Score needed for MATCH, and the lower bar for the REVIEW list |
| `qwen_4bit`, `qwen_max_memory` | Memory options for smaller GPUs |
| `pad_s`, `merge_gap_s` | Seconds added around each clip, and the gap under which clips merge |

Settings marked `(*)` in `config.yaml` change the ingest result. If you change one, the saved ingest is detected as outdated and starts again from the beginning.

---

## Output layout

Everything for one video goes into `outputs/<video>_<full|300s>/`:

```
outputs/test_vid2_300s/
├── crops/                  # saved crops (3 per detection)
├── rows.jsonl, meta.json   # detections and tracks
├── state.json              # ingest progress (resume)
├── emb_*.npz               # cached OpenCLIP vectors
├── results/
│   └── <query name>/       # shortlist sheets, final_results.json,
│                           # confirmed clips + clips_index.csv, review clips
└── logs/                   # time and resource logs (see below)
```

---

## Time and resource log

Every run of `run_ingest.py` and `run_search.py` writes a log to `outputs/<video>/logs/`, **also when the run fails or is stopped with Ctrl+C**.

| File | Content |
|---|---|
| `run_<script>_<time>.txt` | Readable summary table. **Send this one.** |
| `run_<script>_<time>.json` | Everything, machine-readable |
| `run_<script>_<time>_samples.csv` | Raw samples over time, for graphs |

It records, for every stage (video info, detect and crop, model loading, embedding, Qwen checks, clips) and the total time per query:

- **Time** taken
- **Resources:** CPU %, RAM, GPU utilisation, GPU memory (and the peak PyTorch allocated)
- **Machine:** CPU, RAM, GPU names, library versions, git commit
- **Speed:** how many times faster than real time the ingest ran (e.g. 3600 s of video in 600 s = ×6)

GPU numbers need `nvidia-ml-py` (installed through `requirements.txt`) or `nvidia-smi`. Anything that is not available is simply left out of the log.

---

## Project structure

```
video_forensic_search/
├── run_ingest.py        # stage 1 entry point
├── run_search.py        # stage 2 entry point
├── config.yaml          # all settings
├── config_indoor.yaml   # settings for the indoor cameras (with the floor)
├── calib.py             # txt_max of confirmed matches for a run
├── jobtime2.py          # sums the times of a UI job from its logs
├── ui_app2/             # web UI (ui_app/ is the older version)
├── query.yaml           # example query
├── requirements.txt
└── forensic/
    ├── config.py        # loads config and query files, run folders
    ├── ingest.py        # YOLO detection, tracking, crops (resumable)
    ├── embed.py         # OpenCLIP encoder and vector cache
    ├── reference.py     # reference photo loading
    ├── search.py        # shortlist from cached vectors
    ├── verify.py        # Qwen2-VL planning and yes/no checks
    ├── decide.py        # MATCH / REVIEW decision
    ├── clips.py         # cutting clips and review folder
    └── monitor.py       # time and resource log
```

## Notes

- Never commit videos, crops, vectors or model weights (they are in `.gitignore`).
- To test quickly, use `--max-seconds 300` and `ingest_classes: [0]`.
- If the saturation check in the search output says the shortlist is too small, raise `topk` or lower `z_min`.
- On Windows, set `num_workers: 0` in `config.yaml` if the data loader gives errors.
