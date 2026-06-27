"""Schematic OCR to SPICE conversion package.

The package keeps heavyweight ML imports lazy so lightweight commands such as
``python -m statics_ocv --help`` work before the full runtime is installed.
"""

from .config import AppConfig

__all__ = [
    "AppConfig",
    "TraceSegmenter",
    "ComponentDetector",
    "CircuitGraphBuilder",
    "NetlistExporter",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "TraceSegmenter":
        from .segmentation import TraceSegmenter

        return TraceSegmenter
    if name == "ComponentDetector":
        from .detection import ComponentDetector

        return ComponentDetector
    if name == "CircuitGraphBuilder":
        from .graph_builder import CircuitGraphBuilder

        return CircuitGraphBuilder
    if name == "NetlistExporter":
        from .netlist_exporter import NetlistExporter

        return NetlistExporter
    raise AttributeError(name)
