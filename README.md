# Edge AI Runtime Evaluation

This repository documents an on-device AI evaluation project comparing practical deployment paths for Jetson- or Snapdragon-class hardware.

## Domain
Logistics / Edge AI

## Overview
Focused on runtime choice, model conversion, and measuring whether a small model could run smoothly within real device limits.

## Methodology
1. Scoped the work around one target device at a time so runtime tuning and packaging stayed realistic instead of becoming an overly broad benchmark exercise.
2. Selected compact models and input paths that could support live demos while fitting tight memory and power budgets on edge hardware.
3. Prepared device-friendly model exports and compared runtime options such as ONNX Runtime, TensorRT, and TensorFlow Lite based on the chosen target.
4. Integrated OpenCV-driven live or sample input so the demo showed local inference behaviour rather than only offline benchmark numbers.
5. Measured latency, memory use, and power behaviour together because edge-readiness depends on the overall operating profile, not raw speed alone.
6. Documented the tradeoffs clearly so the client could choose a runtime and device path based on deployability rather than hype.

## Skills
- Edge AI Deployment
- ONNX Runtime
- TensorRT
- TensorFlow Lite
- OpenCV
- Model Conversion
- Latency / Power Profiling
- Runtime Optimization

## Source
This README was generated from the portfolio project data used by `/Users/harshitpanikar/Documents/Test_Projs/harshitpaunikar1.github.io/index.html`.
