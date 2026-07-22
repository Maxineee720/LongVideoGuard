# Stage 9C — Streamlit demo

The demo has three tabs:

1. project overview and final figures;
2. frozen error explorer;
3. optional live VideoQA inference.

The first two tabs work without a GPU. Live inference requires CUDA and local
access to `Qwen/Qwen3-VL-2B-Instruct`.

## Install

```bash
pip install -r requirements-stage9c.txt
pip install -e .
```

## Launch locally

```bash
python scripts/run_streamlit_demo.py
```

Open:

```text
http://localhost:8501
```

## Launch in Colab

```bash
python scripts/run_streamlit_demo.py --port 8501
```

Use your preferred Colab tunnel method to expose port 8501.

## Expected inputs

The overview and error explorer expect:

```text
outputs/stage9/project_report/final_metrics.json
outputs/stage9/project_report/figures/*.png
outputs/stage9/final_error_analysis/all_frozen_cases.jsonl
```

## Live policies

```text
Uniform-8
Scene-aware-8
Query-aware-8
Uniform-16
```

Query-aware mode additionally loads CLIP. Uniform-8 is the recommended
engineering default after frozen evaluation.
