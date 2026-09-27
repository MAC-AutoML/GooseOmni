"""GooseOmni declarative data-production pipeline.

The runner imports the video perception stages, which in turn depend on the
optional media stack.  Keep that import lazy so configuration and metadata
helpers remain usable in a base installation.
"""

from typing import TYPE_CHECKING, Any

from .config import DataPipelineConfig, load_data_config

if TYPE_CHECKING:
    from .runner import DataPipelineRunner

__all__ = ["DataPipelineConfig", "DataPipelineRunner", "load_data_config"]


def __getattr__(name: str) -> Any:
    if name == "DataPipelineRunner":
        from .runner import DataPipelineRunner

        return DataPipelineRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
