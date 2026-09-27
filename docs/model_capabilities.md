# GooseOmni model capabilities

This document separates three different claims:

- **Native**: modalities claimed by the model provider for the exact model family.
- **Pipeline**: inputs accepted by GooseOmni's unified `InferenceRequest` and adapter.
- **Verified**: real inference completed in the current SICAU HPC environment.

An adapter accepting a modality does not by itself prove that an upstream API route supports it.

## Local models

| Model | Provider-native inputs | Provider outputs | GooseOmni pipeline | Real AV / V / A / text |
| --- | --- | --- | --- | --- |
| Qwen2.5-Omni | text, image, video, audio | text, audio | AV / V / A / text | PASS, job 512256 |
| Qwen3-Omni Instruct | text, image, video, audio | text, audio | AV / V / A / text | PASS, job 512137 |
| Qwen3-Omni Thinking | text, image, video, audio | text | AV / V / A / text | PASS, job 512257 |
| MiniOmni2 | text, image, audio | text, audio | video frame + audio / V / A / null-media text | PASS, job 512286 |
| OmniVinci | text, image, video, audio | text | AV / V / A / text | PASS, job 512287 |
| VITA-1.5 | text, image, video, audio/speech | text, speech | AV / V / A / text | PASS, job 512314 |
| Baichuan-Omni-1.5 | text, image, video, audio | text, audio | AV / V / A / text | PASS, job 512300 |
| Ming-flash-omni 2.0 | text, image, video, audio | text, image, audio | AV / V / A / text | PASS, job 512541 |

MiniOmni2's visual branch is image-native. GooseOmni samples video into visual input, so the pipeline must not be interpreted as native temporal-video support.

Ming was verified with its provider-pinned `transformers==4.57.1` in an isolated runtime. The smoke job used 8 sampled video frames and 32 generated tokens; AV, video-only, audio-only, and text-only inference all returned semantically valid answers. The shared Transformers 5 development build completed HTTP requests but produced degenerate repeated text, so it is not a supported Ming runtime.

## API models

| Model | Provider-native inputs | GooseOmni pipeline | Current gateway evidence |
| --- | --- | --- | --- |
| GPT-4o | text, image | text, extracted video frames | text and visual PASS; audio is rejected in favor of the dedicated GPT Audio adapter |
| GPT Audio (`gpt-audio-2025-08-28`) | text, audio | audio extracted from the supplied video container | audio PASS; text-only and combined visual/audio returned `429 model_not_found` |
| Gemini 2.5 Flash | text, image, video, audio | text, extracted video frames, audio | text / visual / audio / AV PASS |
| Gemini 2.5 Pro | text, image, video, audio | text, extracted video frames, audio | text / visual / audio PASS |
| Gemini 3 Flash Preview | text, image, video, audio | text, extracted video frames, audio | text / visual / audio PASS |
| Gemini 3 Pro Preview | text, image, video, audio | text, extracted video frames, audio | text / visual / audio PASS |

The gateway catalog exposes `gpt-4o-audio-preview` and `gpt-4o-audio-preview-2024-12-17`, but real text, audio, and combined visual/audio requests to the dated route all returned HTTP 429 because the upstream was saturated. GooseOmni therefore keeps `gpt4o` for text/visual requests and uses a separate `gpt_audio` adapter only for the audio path verified against `gpt-audio-2025-08-28`.

## Provider references

- VITA-1.5: <https://github.com/VITA-MLLM/VITA>
- Baichuan-Omni-1.5: <https://github.com/baichuan-inc/Baichuan-Omni-1.5>
- Ming-flash-omni 2.0: <https://github.com/inclusionAI/Ming>
- Gemini audio cookbook: <https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb>
- Gemini video cookbook: <https://github.com/google-gemini/cookbook/blob/main/quickstarts/Video_understanding.ipynb>
- OpenAI models: <https://platform.openai.com/docs/models>

The OpenAI documentation endpoint was not reachable from the documentation connector during this verification. The GPT-4o row is intentionally conservative and does not infer native audio support from a gateway adapter.
