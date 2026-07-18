# NExT-QA setup

## Why NExT-QA is used

NExT-QA is the first benchmark in this project because it explicitly evaluates
causal, temporal, and descriptive reasoning over everyday videos. The initial
LongVideoGuard baseline uses the official multiple-choice annotations rather
than immediately converting the task to open-ended generation.

## 1. Download annotations

From the repository root:

```bash
python scripts/download_nextqa_annotations.py
```

Expected local files:

```text
data/raw/nextqa/annotations/
├── train.csv
├── val.csv
├── test.csv
└── map_vid_vidorID.json
```

These files are ignored by Git.

## 2. Download raw videos

Raw videos are distributed separately by the NExT-QA authors. Follow the
official NExT-QA repository and dataset page rather than redistributing videos
through LongVideoGuard.

After download, use a layout such as:

```text
data/raw/nextqa/videos/
└── <mapped_video_id>.mp4
```

The official `map_vid_vidorID.json` maps the `video` field in the QA CSV files
to the source VidOR identifiers. The pilot builder can use this map when
writing `video_relpath`.

## 3. Inspect the validation split

```bash
longvideoguard nextqa-stats   data/raw/nextqa/annotations/val.csv
```

## 4. Build the first frozen pilot

The following selects 12 videos and at most 4 questions per selected video:

```bash
longvideoguard nextqa-build-pilot   data/raw/nextqa/annotations/val.csv   data/processed/nextqa/pilot_val.jsonl   --split val   --num-videos 12   --max-questions-per-video 4   --seed 42   --video-id-map data/raw/nextqa/annotations/map_vid_vidorID.json
```

It also creates:

```text
data/processed/nextqa/pilot_val.stats.json
```

## 5. Pilot rules

- Do not tune prompts on the frozen pilot after baseline evaluation begins.
- Keep all records for a source video in the same dataset partition.
- Do not commit raw videos or downloaded annotations.
- Save the exact pilot command, seed, manifest, and stats with each experiment.
- Manually verify that a sample of `video_relpath` values resolves to real files.
