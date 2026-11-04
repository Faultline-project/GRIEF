"""User-visible navigation, terminal sizes and trace rendering."""
import pytest

pytest.importorskip("textual")
from textual.widgets import DataTable, TabbedContent
from fuzzer.tui.app import GriefApp
from fuzzer.demo import demo_events


@pytest.mark.parametrize("size", [(130, 42), (100, 32), (80, 24)])
async def test_demo_navigation_and_restart(size):
    app = GriefApp(demo=True, autoplay=False)
    async with app.run_test(size=size) as pilot:
        for _ in demo_events():
            app.advance_demo()
        await pilot.pause()
        assert app.query_one("#findings", DataTable).row_count == 1
        await pilot.press("2")
        assert app.query_one("#tabs", TabbedContent).active == "evidence"
        assert app.findings[0]["finding"]["demo"] is True
        await pilot.press("r")
        assert app.replay_process is None
        await pilot.press("d")
        assert app.query_one("#findings", DataTable).row_count == 0


async def test_stop_reaps_owned_worker():
    import asyncio
    import sys
    app = GriefApp(demo=True, autoplay=False)
    async with app.run_test() as pilot:
        app.process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await app.action_stop()
        assert app.process.returncode is not None


async def test_watch_partial_stream_does_not_start_server(tmp_path):
    from fuzzer.events import EventWriter
    writer = EventWriter(str(tmp_path / "events.jsonl"))
    writer.emit("run_started", config={"model_name": "recorded", "base_url": "http://old-server"})
    app = GriefApp(run_dir=tmp_path)
    async with app.run_test() as pilot:
        app.poll_events()
        await pilot.pause()
        assert app.process is None
        assert app.reader.offset > 0
