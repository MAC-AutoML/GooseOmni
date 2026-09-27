"""Postprocessing must never implicitly consume shared historical artifacts."""

import importlib

import pytest

from gooseomni.cli import main


@pytest.mark.parametrize(
    "module_name",
    [
        "tools.build.build_meeting_utterances",
        "tools.build.build_information_states",
        "tools.build.merge_global_events",
        "tools.build.build_candidate_trials",
        "tools.package.export_event_aligned_omni_eval",
    ],
)
def test_postprocess_requires_explicit_paths(module_name, monkeypatch, capsys):
    module = importlib.import_module(module_name)
    monkeypatch.setattr("sys.argv", [module_name])
    with pytest.raises(SystemExit) as exc:
        module.parse_args()
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err


def test_removed_batch_postprocess_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["annotate", "postprocess"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
