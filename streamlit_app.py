from __future__ import annotations

import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from longvideoguard.demo.app_utils import (
    DISPLAY_METHODS,
    accuracy_card_payload,
    build_multiple_choice_prompt,
    load_json,
    load_jsonl,
    parse_answer,
)
from longvideoguard.evaluation.counterfactuals import write_h264_video
from longvideoguard.retrieval.frame_sampling import (
    CLIPFrameEncoder,
    decode_uniform_candidates,
    evenly_spaced_indices,
    frame_change_scores,
    query_aware_indices,
    scene_aware_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "outputs/stage9/project_report"
ERROR_ROOT = PROJECT_ROOT / "outputs/stage9/final_error_analysis"

st.set_page_config(
    page_title="LongVideoGuard",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 LongVideoGuard")
st.caption(
    "Evidence-aware and efficiency-focused VideoQA with Qwen3-VL"
)


@st.cache_data(show_spinner=False)
def load_report_data() -> dict[str, object] | None:
    path = REPORT_ROOT / "final_metrics.json"
    if not path.is_file():
        return None
    return load_json(path)


@st.cache_data(show_spinner=False)
def load_error_cases() -> list[dict[str, object]]:
    path = ERROR_ROOT / "all_frozen_cases.jsonl"
    if not path.is_file():
        return []
    return load_jsonl(path)


@st.cache_resource(show_spinner="Loading Qwen3-VL...")
def load_qwen_model(
    model_name: str,
    dtype_name: str,
    attn_implementation: str,
):
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Live inference requires a CUDA GPU. "
            "The report and error explorer still work without one."
        )

    dtype = getattr(torch, dtype_name)
    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.to("cuda")
    model.eval()
    model.config.use_cache = True
    return processor, model


@st.cache_resource(show_spinner="Loading CLIP retriever...")
def load_clip_encoder(model_name: str):
    return CLIPFrameEncoder(model_name)


def make_selected_clip(
    source_video: Path,
    *,
    method: str,
    question: str,
    candidate_frames: int = 32,
) -> tuple[Path, list[float], list[float]]:
    candidates = decode_uniform_candidates(
        source_video,
        candidate_count=candidate_frames,
    )

    if method == "uniform_8":
        indices = evenly_spaced_indices(
            len(candidates.rgb_frames),
            min(8, len(candidates.rgb_frames)),
        )
    elif method == "scene_aware_8":
        indices = scene_aware_indices(
            frame_change_scores(candidates.rgb_frames),
            count=min(8, len(candidates.rgb_frames)),
            min_gap=2,
        )
    elif method == "query_aware_8":
        encoder = load_clip_encoder(
            "openai/clip-vit-base-patch32"
        )
        image_embeddings = encoder.encode_images(
            candidates.rgb_frames,
            batch_size=16,
        )
        text_embedding = encoder.encode_text(question)
        indices, _ = query_aware_indices(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            count=min(8, len(candidates.rgb_frames)),
            relevance_weight=0.8,
            min_gap=2,
        )
    else:
        raise ValueError(f"Unsupported selected-frame method: {method}")

    selected_frames = [
        candidates.rgb_frames[index] for index in indices
    ]
    selected_times = [
        candidates.timestamps_seconds[index] for index in indices
    ]

    temporary_root = Path(tempfile.mkdtemp(
        prefix="longvideoguard_demo_"
    ))
    output_path = temporary_root / f"{method}.mp4"
    write_h264_video(
        selected_frames,
        output_path,
        fps=1.0,
    )
    return output_path, selected_times, [
        float(index) for index in indices
    ]


def run_videoqa(
    video_path: Path,
    *,
    question: str,
    options: list[str],
    method: str,
    model_name: str,
    dtype_name: str,
    attn_implementation: str,
) -> dict[str, object]:
    import torch

    if method == "uniform_16":
        inference_video = video_path
        num_frames = 16
        selected_times = None
    else:
        inference_video, selected_times, _ = make_selected_clip(
            video_path,
            method=method,
            question=question,
        )
        num_frames = 8

    processor, model = load_qwen_model(
        model_name,
        dtype_name,
        attn_implementation,
    )
    prompt = build_multiple_choice_prompt(question, options)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": str(inference_video),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={
            "text_kwargs": {
                "return_tensors": "pt",
            },
            "videos_kwargs": {
                "do_sample_frames": True,
                "num_frames": num_frames,
            },
        },
    )
    inputs.pop("token_type_ids", None)
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_length = int(inputs["input_ids"].shape[-1])

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**2)
    )

    new_tokens = generated[:, input_length:]
    raw_output = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return {
        "prediction": parse_answer(raw_output),
        "raw_output": raw_output,
        "latency_seconds": latency,
        "input_token_count": input_length,
        "generated_token_count": int(new_tokens.shape[-1]),
        "peak_gpu_memory_mb": peak_memory,
        "selected_video_path": str(inference_video),
        "selected_timestamps_seconds": selected_times,
        "num_frames": num_frames,
    }


tabs = st.tabs(
    [
        "Project overview",
        "Frozen error explorer",
        "Live VideoQA",
    ]
)

with tabs[0]:
    metrics = load_report_data()
    if metrics is None:
        st.warning(
            "Run Stage 9B first so outputs/stage9/project_report/"
            "final_metrics.json exists."
        )
    else:
        cards = accuracy_card_payload(metrics)
        columns = st.columns(3)
        for column, row in zip(columns, cards, strict=True):
            with column:
                st.metric(
                    row["method"],
                    f"{100 * row['accuracy']:.2f}%",
                    help=(
                        f"{row['correct']}/{row['count']} correct; "
                        f"{row['mean_input_tokens']:.1f} mean input tokens"
                    ),
                )

        st.subheader("Frozen accuracy–efficiency result")
        st.write(
            "Uniform-8 reduced the visual-token budget by "
            f"**{metrics['derived_takeaways']['uniform8_visual_token_reduction_percent']:.1f}%** "
            "and model inference latency by "
            f"**{metrics['derived_takeaways']['uniform8_latency_reduction_percent']:.1f}%** "
            "relative to Uniform-16, while losing only "
            f"**{abs(metrics['derived_takeaways']['uniform8_vs_uniform16_accuracy_delta_pp']):.2f} "
            "percentage points** of frozen accuracy."
        )

        figure_paths = [
            (
                "Frozen accuracy",
                REPORT_ROOT / "figures/frozen_accuracy.png",
            ),
            (
                "Accuracy–token trade-off",
                REPORT_ROOT / "figures/accuracy_efficiency.png",
            ),
            (
                "Counterfactual evaluation",
                REPORT_ROOT / "figures/counterfactual_accuracy.png",
            ),
        ]
        figure_columns = st.columns(3)
        for column, (title, path) in zip(
            figure_columns,
            figure_paths,
            strict=True,
        ):
            with column:
                st.markdown(f"**{title}**")
                if path.is_file():
                    st.image(str(path), use_container_width=True)

        st.subheader("Counterfactual diagnostics")
        diagnostics = metrics["counterfactual_diagnostics"]
        st.json(diagnostics, expanded=False)

with tabs[1]:
    cases = load_error_cases()
    if not cases:
        st.warning(
            "Run Stage 9A first so all_frozen_cases.jsonl exists."
        )
    else:
        frame = pd.DataFrame(cases)
        category = st.selectbox(
            "Question category",
            ["all"] + sorted(
                frame["question_category"].dropna().unique().tolist()
            ),
        )
        bucket = st.selectbox(
            "Error pattern",
            ["all"] + sorted(
                frame["error_pattern"].dropna().unique().tolist()
            ),
        )
        only_disagreement = st.checkbox(
            "Only prediction disagreements"
        )

        filtered = frame.copy()
        if category != "all":
            filtered = filtered[
                filtered["question_category"] == category
            ]
        if bucket != "all":
            filtered = filtered[
                filtered["error_pattern"] == bucket
            ]
        if only_disagreement:
            filtered = filtered[
                ~filtered["all_predictions_same"]
            ]

        st.write(f"Matching cases: **{len(filtered)}**")
        if len(filtered):
            sample_id = st.selectbox(
                "Frozen sample",
                filtered["sample_id"].tolist(),
            )
            row = filtered[
                filtered["sample_id"] == sample_id
            ].iloc[0]

            st.markdown(f"### {row['question']}")
            options = row["options"]
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except json.JSONDecodeError:
                    options = [options]

            for letter, option in zip(
                ("A", "B", "C", "D", "E"),
                options,
            ):
                st.write(f"**{letter}.** {option}")

            result_columns = st.columns(4)
            result_columns[0].metric(
                "Gold",
                str(row["gold_answer_letter"]),
            )
            result_columns[1].metric(
                "Uniform-8",
                str(row["uniform_8_prediction"]),
            )
            result_columns[2].metric(
                "Scene-aware-8",
                str(row["scene_aware_8_prediction"]),
            )
            result_columns[3].metric(
                "Uniform-16",
                str(row["uniform_16_prediction"]),
            )

            st.json(
                {
                    "category": row["question_category"],
                    "error_pattern": row["error_pattern"],
                    "all_predictions_same": bool(
                        row["all_predictions_same"]
                    ),
                },
                expanded=False,
            )

with tabs[2]:
    st.info(
        "Live inference requires a CUDA GPU and local access to "
        "Qwen3-VL-2B-Instruct. The report and error explorer do not."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
    )
    question = st.text_input(
        "Question",
        placeholder="What does the person do after opening the door?",
    )
    options = [
        st.text_input(f"Option {letter}")
        for letter in ("A", "B", "C", "D", "E")
    ]
    method = st.selectbox(
        "Sampling policy",
        [
            "uniform_8",
            "scene_aware_8",
            "query_aware_8",
            "uniform_16",
        ],
        format_func=lambda value: DISPLAY_METHODS[value],
    )

    with st.expander("Model settings"):
        model_name = st.text_input(
            "Model",
            value="Qwen/Qwen3-VL-2B-Instruct",
        )
        dtype_name = st.selectbox(
            "Dtype",
            ["bfloat16", "float16"],
        )
        attn_implementation = st.selectbox(
            "Attention implementation",
            ["sdpa", "eager"],
        )

    if st.button("Run VideoQA", type="primary"):
        if uploaded_video is None:
            st.error("Upload a video first.")
        else:
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(uploaded_video.name).suffix,
                    delete=False,
                ) as handle:
                    handle.write(uploaded_video.getbuffer())
                    source_path = Path(handle.name)

                with st.spinner("Running selected-frame VideoQA..."):
                    result = run_videoqa(
                        source_path,
                        question=question,
                        options=options,
                        method=method,
                        model_name=model_name,
                        dtype_name=dtype_name,
                        attn_implementation=attn_implementation,
                    )

                prediction = result["prediction"]
                st.success(
                    f"Prediction: {prediction or 'Invalid output'}"
                )
                metric_columns = st.columns(4)
                metric_columns[0].metric(
                    "Frames",
                    result["num_frames"],
                )
                metric_columns[1].metric(
                    "Input tokens",
                    result["input_token_count"],
                )
                metric_columns[2].metric(
                    "Latency",
                    f"{result['latency_seconds']:.3f}s",
                )
                metric_columns[3].metric(
                    "Peak GPU memory",
                    f"{result['peak_gpu_memory_mb']:.1f} MB",
                )

                if result["selected_timestamps_seconds"] is not None:
                    st.write(
                        "Selected timestamps (seconds):",
                        [
                            round(value, 3)
                            for value in result[
                                "selected_timestamps_seconds"
                            ]
                        ],
                    )
                    st.video(result["selected_video_path"])

                with st.expander("Raw generation"):
                    st.code(result["raw_output"])
            except Exception as exc:
                st.exception(exc)
            finally:
                gc.collect()
