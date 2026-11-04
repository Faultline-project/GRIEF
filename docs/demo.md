# Demo walkthrough

The included MP4 and GIF render the actual Textual interface using the
simulated teaching session. Every frame is labelled **SIMULATED · no server**.
The example illustrates how to read concurrent traces, serving-state feedback,
finding evidence and confirmation records. Its counts and outcomes are
synthetic; it is not evidence of an engine vulnerability.

Rebuild the recording:

```bash
pip install -e '.[tui,video]'
python scripts/record_demo.py
```

The script drives the application through Textual's headless test driver,
captures real screen states as SVG, rasterizes them, and encodes an MP4 using
the FFmpeg binary provided by `imageio-ffmpeg`. It also exports a short README
GIF. Intermediate frames go to the ignored `demo-frames/` directory.

## Recording a real engine case

Choose a publicly documented issue with a small model and a repeatable
trigger. Pin the engine build, model revision, launch flags and GPU type.
Confirm the trigger on that configuration before recording. Run through the
live TUI, then retain the finding, original trace, confirmation results and
run events. Present replays as recordings and report the actual verdict.

The intended live story is: connect, show overlapping requests, observe a
candidate, inspect its evidence, stop exploration, and replay the workload.
An inconclusive result remains inconclusive in the video. A real-GPU example
can replace the teaching video once that validation is available.
