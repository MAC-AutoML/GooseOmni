import argparse
import os
import sys
import tempfile
import warnings
import traceback
import re
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VITA_ROOT = Path(__file__).resolve().parent / "vita_lib"
if str(VITA_ROOT) not in sys.path:
    sys.path.insert(0, str(VITA_ROOT))

from gooseomni.config.settings import CONFIG
from gooseomni.models.model_server.local_common.gpu_visibility import configure_cuda_visible_devices
from gooseomni.models.model_server.local_common.http import parse_infer_request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Set GPU visibility before importing torch
PHYSICAL_GPUS = configure_cuda_visible_devices(
    CONFIG.model("vita_1_5").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", [])
)

os.environ.setdefault("VITA_DISABLE_AUDIO", "0")
os.environ.setdefault("VITA_DELAY_VISION_TOWER", "1")

# Global configuration
MODEL_PATH = CONFIG.model("vita_1_5").get("model_path") or "/publicssd/xty/models/VITA-1.5"
os.environ.setdefault(
    "VITA_AUDIO_ENCODER_PATH",
    str(
        CONFIG.model("vita_1_5").get("audio_encoder_path")
        or Path(MODEL_PATH)
        / "audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning"
    ),
)
os.environ.setdefault(
    "VITA_VISION_TOWER_PATH",
    str(CONFIG.model("vita_1_5").get("vision_tower_path") or ""),
)
MODEL_BASE = None
MODEL_TYPE = "qwen2p5_instruct"
CONV_MODE = "qwen2p5_instruct"
MAX_NEW_TOKENS = CONFIG.model("vita_1_5").get("max_tokens", 256)
TEMPERATURE = CONFIG.model("vita_1_5").get("temperature", 0.3)
TOP_P = 1.0
NUM_BEAMS = 1
MAX_FRAMES = None
VIDEO_FRAMERATE = 1

# Global runtime objects
model = None
tokenizer = None
image_processor = None
audio_processor = None
model_loaded = False


def _parse_bool(value, default=True):
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=None,
    min_frames=4,
    image_resolution=384,
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    from decord import VideoReader, cpu
    import numpy as np
    from PIL import Image
    import torch

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if max_frames is not None and len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                if width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

            patch_images = [
                expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        patch_images = torch.stack(patch_images)
        slice_len = patch_images.shape[0]
        return patch_images, slice_len

    raise RuntimeError(f"No frames decoded from {video_path}.")


def load_model():
    global model, tokenizer, image_processor, audio_processor, model_loaded

    if model_loaded:
        return

    import torch
    from vita.model.builder import load_pretrained_model
    from vita.util.utils import disable_torch_init
    from vita.util.mm_utils import get_model_name_from_path

    if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
        try:
            if str(torch.get_default_device()) == "meta":
                torch.set_default_device("cuda")
        except Exception:
            pass

    disable_torch_init()
    model_path = os.path.expanduser(MODEL_PATH)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        MODEL_BASE,
        model_name,
        MODEL_TYPE,
        device_map={"": 0},
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    if audio_encoder is None:
        raise RuntimeError("VITA-1.5 audio encoder is not loaded")
    audio_encoder = audio_encoder.to(device="cuda", dtype=torch.float16)
    _materialize_audio_runtime_tensors(audio_encoder)
    audio_processor = audio_encoder.audio_processor

    model.eval()
    model_loaded = True


def _materialize_audio_runtime_tensors(audio_encoder):
    """Rebuild non-parameter tensors created inside HF's meta init context."""
    import math
    import torch
    import torchaudio.compliance.kaldi as kaldi

    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
    kaldi.EPSILON = torch.tensor(torch.finfo(torch.float).eps, device="cpu")

    for module in audio_encoder.modules():
        pe = getattr(module, "pe", None)
        if isinstance(pe, torch.Tensor) and pe.device.type == "meta":
            max_len = pe.shape[-2]
            d_model = pe.shape[-1]
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / d_model)
            )
            rebuilt = torch.zeros(max_len, d_model)
            rebuilt[:, 0::2] = torch.sin(position * div_term)
            rebuilt[:, 1::2] = torch.cos(position * div_term)
            module.pe = rebuilt.unsqueeze(0)

        div_term = getattr(module, "div_term", None)
        if isinstance(div_term, torch.Tensor) and div_term.device.type == "meta":
            d_model = int(getattr(module, "d_model", div_term.numel() * 2))
            module.div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / d_model)
            )


def _build_audio_placeholder():
    import torch

    audio = torch.zeros(400, 80)
    audio_length = audio.shape[0]
    audio_for_llm_lens = 60
    audio = torch.unsqueeze(audio, dim=0)
    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
    audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)

    audios = {
        "audios": audio.half().cuda(),
        "lengths": audio_length.half().cuda(),
        "lengths_for_llm": audio_for_llm_lens.cuda(),
    }
    return audios


def _extract_audio(video_path: str, temp_dir: str) -> str:
    import subprocess

    audio_path = os.path.join(temp_dir, "input.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            audio_path,
        ],
        check=True,
        timeout=300,
    )
    return audio_path


def _prepare_audio(video_path: str, temp_dir: str):
    import torch

    assert audio_processor is not None
    audio_path = _extract_audio(video_path, temp_dir)
    audio, audio_for_llm_lens = audio_processor.process(audio_path)
    audio_length = audio.shape[0]
    return {
        "audios": torch.unsqueeze(audio, dim=0).half().cuda(),
        "lengths": torch.unsqueeze(torch.tensor(audio_length), dim=0).half().cuda(),
        "lengths_for_llm": torch.unsqueeze(
            torch.tensor(audio_for_llm_lens), dim=0
        ).cuda(),
    }


def run_inference(
    video_path: str,
    question: str,
    use_video: bool,
    use_audio: bool,
    temp_dir: str,
):
    assert model_loaded and model is not None, "Model is not loaded"

    from vita.constants import (
        DEFAULT_AUDIO_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
        MAX_IMAGE_LENGTH,
    )
    from vita.conversation import SeparatorStyle, conv_templates
    from vita.util.mm_utils import (
        KeywordsStoppingCriteria,
        tokenizer_image_audio_token,
        tokenizer_image_token,
    )

    import torch

    if use_video:
        max_frames = MAX_FRAMES or MAX_IMAGE_LENGTH
        video_frames, slice_len = _get_rawvideo_dec(
            video_path,
            image_processor,
            max_frames=max_frames,
            video_framerate=VIDEO_FRAMERATE,
            image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
        )
        image_tensor = video_frames.half().cuda()
        prompt_text = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + question
        modality = "video"
    else:
        image_tensor = torch.zeros((1, 3, 448, 448)).half().cuda()
        prompt_text = question
        modality = "lang"

    if use_audio:
        audios = _prepare_audio(video_path, temp_dir)
        prompt_text += DEFAULT_AUDIO_TOKEN
    else:
        audios = _build_audio_placeholder()

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt(modality=modality)

    tokenize = tokenizer_image_audio_token if use_audio else tokenizer_image_token
    input_ids = tokenize(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if conv.version == "qwen2p5_instruct":
        stop_str = "<|im_end|>"
        stopping_criteria = []
    else:
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=1,
            use_cache=True,
            stopping_criteria=stopping_criteria if stopping_criteria else None,
        )

    output_ids = output_ids.sequences
    input_token_len = input_ids.shape[1]
    gen_ids = output_ids[:, input_token_len:] if output_ids.shape[1] > input_token_len else output_ids
    outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    if not outputs:
        outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    if not outputs:
        raw_full = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        match = re.search(r"\\b([A-D])\\b", raw_full.upper())
        if match:
            outputs = match.group(1)

    return outputs


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_loaded": model_loaded})


@app.route("/v1/infer", methods=["POST"])
def analyze_video():
    try:
        payload = parse_infer_request(request, default_audio=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    question = payload.prompt
    use_video = payload.use_video
    use_audio = payload.use_audio

    temp_dir = None
    temp_path = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="vita_server_")
        if payload.upload is not None:
            temp_path = os.path.join(temp_dir, payload.upload.filename)
            payload.upload.save(temp_path)

        answer = run_inference(temp_path, question, use_video, use_audio, temp_dir)
        return jsonify({"status": "success", "answer": answer})
    except Exception as exc:  # noqa: BLE001
        print("VITA server analyze error:", exc, flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"status": "error", "error": str(exc)}), 500
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="VITA-1.5 Video Analysis Server")
    parser.add_argument("--port", type=int, default=5093, help="Server port (default: 5093)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address (default: 0.0.0.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model()
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
