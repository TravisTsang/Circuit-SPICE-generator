"""Training utilities for the Circuit-SPICE-generator models.

This package contains everything needed to produce the four weight files the
inference app expects but does not ship:

    models/hand_drawn_unet_trace_segmentation.pt
    models/hand_drawn_yolov8_components.pt
    models/digital_unet_trace_segmentation.pt
    models/digital_yolov8_components.pt

It is intentionally separate from the ``statics_ocv`` inference package so that
the runtime image does not need training-only dependencies.
"""
