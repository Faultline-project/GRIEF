"""Render the real TUI through Textual's headless driver into a reproducible video.

Requires: pip install '.[tui,video]'
The fixture and the video are explicitly simulated, never a real bug discovery.
"""
from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
import subprocess

# Capture in color even on CI runners that set NO_COLOR/TERM=dumb.
os.environ.pop("NO_COLOR", None)
os.environ["TERM"] = "xterm-256color"
os.environ["COLORTERM"] = "truecolor"

import cairosvg
import imageio_ffmpeg
from PIL import Image

from fuzzer.demo import demo_events
from fuzzer.tui.app import GriefApp


async def record(root: Path):
    out = root / "docs" / "media"
    out.mkdir(parents=True, exist_ok=True)
    frames = root / "demo-frames"
    frames.mkdir(exist_ok=True)
    shots = []
    durations = []
    app = GriefApp(demo=True, autoplay=False)
    async with app.run_test(size=(130, 42)) as pilot:
        async def capture(seconds):
            await pilot.pause()
            svg = app.export_screenshot(title="GRIEF · simulated walkthrough")
            png = cairosvg.svg2png(bytestring=svg.encode())
            shot = Image.open(io.BytesIO(png)).convert("RGB")
            shot.thumbnail((1840, 990), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1920, 1080), "#0b1220")
            canvas.paste(shot, ((1920 - shot.width) // 2, (1080 - shot.height) // 2))
            path = frames / f"frame-{len(shots):03d}.png"
            canvas.save(path)
            shots.append(path)
            durations.append(seconds)
            return svg

        # The app consumes the first record in on_mount.
        holds = [8, 12, 12, 10, 14, 10, 10, 12, 10, 12, 10, 12, 10, 8]
        await capture(holds[0])
        for index in range(1, len(demo_events())):
            app.advance_demo()
            svg = await capture(holds[index])
            if index == 8:
                (out / "screenshot.svg").write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
                await pilot.press("2")
                await capture(16)
                await pilot.press("1")
        await pilot.press("3")
        await capture(8)
        await pilot.press("1")
        await capture(6)
    manifest = frames / "frames.txt"
    lines = []
    for path, duration in zip(shots, durations):
        lines.extend([f"file '{path.name}'", f"duration {duration}"])
    lines.append(f"file '{shots[-1].name}'")
    manifest.write_text("\n".join(lines) + "\n")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(manifest), "-t", str(sum(durations)),
        "-r", "10", "-c:v", "libx264", "-preset", "fast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "demo.mp4")], check=True)
    thumbnails = []
    for idx in [2, 4, 6, 8, 9, 11, 14]:
        im = Image.open(shots[idx]).convert("RGB")
        im.thumbnail((880, 495), Image.Resampling.LANCZOS)
        thumbnails.append(im)
    thumbnails[0].save(out / "demo.gif", save_all=True, append_images=thumbnails[1:],
                       duration=3000, loop=0, optimize=True)
    print(f"Rendered {len(shots)} TUI views; {sum(durations)} seconds; {out / 'demo.mp4'}")


if __name__ == "__main__":
    asyncio.run(record(Path(__file__).resolve().parents[1]))
