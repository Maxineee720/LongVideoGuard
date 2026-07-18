# NExT-QA pilot videos

The official NExT-QA repository distributes the raw NExTVideo archive
separately from the annotations.

## Download

Keep the archive outside Git, for example:

```text
data/raw/nextqa/archive/NExTVideo.zip
```

## Extract only the pilot files

```bash
python scripts/extract_nextqa_pilot_videos.py   data/raw/nextqa/archive/NExTVideo.zip   data/processed/nextqa/pilot_val.jsonl
```

The script indexes the ZIP and extracts only the unique videos referenced by
the pilot manifest.

## Validate

```bash
longvideoguard validate-videos   data/processed/nextqa/pilot_val.jsonl   data/raw/nextqa/videos   --output data/processed/nextqa/pilot_video_validation.json
```

Proceed to VLM inference only when `all_videos_ready` is `true`.
