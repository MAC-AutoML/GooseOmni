import argparse
import os
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.config.settings import CONFIG
from gooseomni.models.model_server.local_common.gpu_visibility import configure_cuda_visible_devices
from gooseomni.models.model_server.local_common.http import parse_infer_request
from gooseomni.models.model_server.local_common.media_masking import create_black_frame_video
from gooseomni.models.model_server.local_common.remote_code_cache import sync_remote_code_cache
from gooseomni.models.model_server.local_common.transformers_compat import (
    ensure_all_tied_weights_keys,
    ensure_transformers_no_init_weights,
)

warnings.filterwarnings("ignore")

app = Flask(__name__)


# IMPORTANT: GPU visibility must be set before importing transformers
SPECIFIED_GPUS = configure_cuda_visible_devices(
    CONFIG.model("omnivinci").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", [])
)

# Global configuration
MODEL_PATH = CONFIG.model("omnivinci").get("model_path") or "/publicssd/xty/models/omnivinci"
LOAD_AUDIO_IN_VIDEO = CONFIG.model("omnivinci").get("use_audio_in_video", True)
NUM_VIDEO_FRAMES = CONFIG.model("omnivinci").get("num_video_frames", 256)
AUDIO_LENGTH = "max_7200"
MAX_NEW_TOKENS = CONFIG.model("omnivinci").get("max_new_tokens", 4096)

# Global runtime objects
model = None
processor = None
config = None
generation_config = None
model_loaded = False


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def _load_processor():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except FileNotFoundError:
        sync_remote_code_cache(MODEL_PATH, "omnivinci")
        return AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


def load_model():
    global model, processor, config, generation_config, model_loaded

    if model_loaded:
        return

    from transformers import AutoConfig, AutoModel
    import torch

    print(f"Loading OmniVinci model to GPUs {os.environ.get('CUDA_VISIBLE_DEVICES')}...")
    sync_remote_code_cache(MODEL_PATH, "omnivinci")

    ensure_transformers_no_init_weights()
    ensure_all_tied_weights_keys()

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = _load_processor()

    generation_config = model.default_generation_config
    generation_config.update(max_new_tokens=MAX_NEW_TOKENS)
    generation_config.max_length = None

    model.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO
    processor.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO

    if NUM_VIDEO_FRAMES > 0:
        model.config.num_video_frames = NUM_VIDEO_FRAMES
        processor.config.num_video_frames = NUM_VIDEO_FRAMES

    if AUDIO_LENGTH != -1:
        model.config.audio_chunk_length = AUDIO_LENGTH
        processor.config.audio_chunk_length = AUDIO_LENGTH

    model_loaded = True
    print("Model loaded successfully.")


def _build_conversation(
    media_path: str, question: str, use_video: bool, use_audio: bool
):
    content = []
    if use_video:
        content.append({"type": "video", "video": str(media_path)})
    elif use_audio:
        content.append({"type": "audio", "audio": str(media_path)})
    content.append({"type": "text", "text": question})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def run_inference(
    video_path: str,
    question: str,
    use_video: bool,
    use_audio: bool,
    visual_mask: bool = False,
    temp_dir: str | None = None,
) -> str:
    assert model_loaded and model is not None and processor is not None, "Model is not loaded"
    inference_video_path = video_path
    if visual_mask and use_video:
        if temp_dir is None:
            raise RuntimeError("temp_dir is required for visual_mask=True")
        inference_video_path = create_black_frame_video(video_path, temp_dir)

    model.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO and use_audio
    processor.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO and use_audio

    if use_video:
        model.config.num_video_frames = NUM_VIDEO_FRAMES
        processor.config.num_video_frames = NUM_VIDEO_FRAMES
    else:
        model.config.num_video_frames = 0
        processor.config.num_video_frames = 0

    conversation = _build_conversation(
        inference_video_path, question, use_video, use_audio
    )
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    inputs = processor([text])

    media = inputs.media if hasattr(inputs, "media") else {}
    has_video = bool(media.get("video")) if isinstance(media, dict) else False
    has_audio = False
    if isinstance(media, dict):
        video_info = media.get("video_info")
        if isinstance(video_info, list) and video_info:
            first_info = video_info[0]
            if isinstance(first_info, dict) and "has_audio" in first_info:
                has_audio = bool(first_info.get("has_audio"))
        if not has_audio:
            has_audio = bool(media.get("sound")) or bool(media.get("audio_info"))
    print(f"Request media: video={has_video} audio={has_audio}")

    input_ids = inputs.input_ids
    if hasattr(inputs, "media"):
        media = inputs.media
    else:
        media = None

    if hasattr(inputs, "media_config"):
        media_config = inputs.media_config
    else:
        media_config = None

    output_ids = model.generate(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        generation_config=generation_config,
    )

    response = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response.strip()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_loaded": bool(model_loaded)})


@app.route("/v1/infer", methods=["POST"])
def analyze_video():
    try:
        payload = parse_infer_request(request, default_audio=LOAD_AUDIO_IN_VIDEO)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    question = payload.prompt
    use_video = payload.use_video
    use_audio = payload.use_audio
    visual_mask = payload.visual_mask

    temp_dir = None
    temp_path = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="omnivinci_server_")
        if payload.upload is not None:
            temp_path = os.path.join(temp_dir, payload.upload.filename)
            payload.upload.save(temp_path)

        answer = run_inference(temp_path, question, use_video, use_audio, visual_mask, temp_dir)
        return jsonify({"status": "success", "answer": answer.strip()})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(exc)}), 500
    finally:
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="OmniVinci Video Analysis Server")
    parser.add_argument("--port", type=int, default=5091, help="Server port (default: 5091)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address (default: 0.0.0.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model()
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
