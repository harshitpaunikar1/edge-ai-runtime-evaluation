# Edge AI Runtime Evaluation Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## Runtime comparison bar chart (latency/memory/power)

```mermaid
flowchart TD
    N1["Step 1\nScoped the work around one target device at a time so runtime tuning and packaging"]
    N2["Step 2\nSelected compact models and input paths that could support live demos while fittin"]
    N1 --> N2
    N3["Step 3\nPrepared device-friendly model exports and compared runtime options such as ONNX R"]
    N2 --> N3
    N4["Step 4\nIntegrated OpenCV-driven live or sample input so the demo showed local inference b"]
    N3 --> N4
    N5["Step 5\nMeasured latency, memory use, and power behaviour together because edge-readiness "]
    N4 --> N5
```

## Model conversion pipeline

```mermaid
flowchart LR
    N1["Inputs\nPrompt variants, evaluation examples, and scoring notes"]
    N2["Decision Layer\nModel conversion pipeline"]
    N1 --> N2
    N3["User Surface\nOnly the narrative workflow surface is present; no runnable interface artifa"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
