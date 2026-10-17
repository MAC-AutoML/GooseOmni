# data_preprocess

Standalone **episode split → pack → enter-game align** for multi-POV Goose Goose Duck recordings.

- **No imports from `gooseomni`.** Self-contained under this folder.
- Paths are relative to the repo root / a workspace directory (no machine-specific absolute paths).
- Do not commit videos, review frames, or private rosters.

## Layout

```text
src/data_preprocess/             # this package
data_preprocess_workspace/       # default local data root (gitignored)
  raw/<session>/<player>.mp4
  episodes/<session>/<player>/episode_XXX_<label>.mp4
  games/<session>/game_XX_<label>_<sec>s/<player>.mp4
  aligned/<session>/game_XX_*/<player>.mp4
  runs/split/<session>/<player>/
  sessions/<session>/players.txt   # optional expected POV list
  pipeline_gates/<session>.json
```

Override the workspace with env `GOOSEOMNI_PREPROCESS_WORKSPACE`.

Add `data_preprocess_workspace/` to the repository root `.gitignore` (default data root). Do not commit videos or review frames.

## Pipeline

```text
raw long VOD
  → split   (role-reveal start → settlement end)
  → pack    (match POVs by settlement label + duration)
  → gate    (no gaps / pending / sticky before align)
  → align   (trim each POV to first in_game after role reveal)
```

## Commands

Run from the **repository root**. Scripts add `src/` to `sys.path` themselves:

```bash
# 1) Split one POV
python src/data_preprocess/split/run_split.py \
  --session 2024-01-01 --player-id player_a \
  --backend local --server-url "$QWEN3_OMNI_SERVER_URL"

# 2) Pack a session
python src/data_preprocess/pack/run_pack.py --session 2024-01-01

# 3) Check gate only
python src/data_preprocess/pack/run_pack.py --session 2024-01-01 --gate-only

# 4) Align (needs src/data_preprocess/assets/start_template.png for --backend local)
python src/data_preprocess/align/run_align.py \
  --session 2024-01-01 \
  --backend local --server-url "$QWEN3_OMNI_SERVER_URL"
```

`--backend mock` skips the model (for wiring tests).

## Dependencies

- Python 3.10+
- `ffmpeg` / `ffprobe` on `PATH`
- `requests` (local Omni HTTP backend)

## Notes

- Split windows are 90s with 5s overlap; after a locked start, end search skips ~3 minutes.
- Pack keeps games with ≥2 POVs; leftovers go to `games/<session>/pending/`.
- Align skips games with fewer than 4 POVs by default (`--min-povs-per-game`).
- Gate blocks align when episode indices have holes, pack is stale, or pending clips remain.
