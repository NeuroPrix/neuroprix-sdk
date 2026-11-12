"""Playwright bridge: drives game/index.html in headless Chromium like a gym environment."""
from __future__ import annotations

import json
import random
import os
from pathlib import Path

GAME_HTML = Path(os.environ.get("NEUROPRIX_GAME_HTML") or Path(__file__).resolve().parent.parent / "game" / "index.html")   # override: experiments


def load_track_spec(arg: str | int | dict) -> dict:
    """Accepts a seed (int / digit string), a path to a track JSON from the Gym Editor, or a spec dict."""
    if isinstance(arg, dict):
        return arg
    if isinstance(arg, int) or (isinstance(arg, str) and arg.lstrip("-").isdigit()):
        return {"seed": int(arg)}
    return json.loads(Path(arg).read_text())


class RaceEnv:
    """One browser page = one car. `tracks` is a list of track specs; reset() can pick one at random
    (a training curriculum) or you can pass a specific spec."""

    def __init__(self, tracks: list[dict] | None = None, headless: bool = True, random_tracks: bool = False):
        self.tracks = tracks or [{"seed": 1}]
        self.random_tracks = random_tracks
        from playwright.sync_api import sync_playwright   # only for the browser engine: runners don't need Playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=headless,
            args=["--disable-dev-shm-usage", "--disable-gpu", "--no-first-run", "--disable-extensions"],
        )
        self.page = self._browser.new_page(viewport={"width": 1300, "height": 760})
        url = GAME_HTML.as_uri() + ("?headless=1" if headless else "")
        self.page.goto(url)
        self.page.wait_for_function("typeof window.stepGame === 'function'")
        self._loaded_key = None
        self.track_info: dict = {}

    def load_track(self, spec: dict) -> dict:
        key = json.dumps(spec, sort_keys=True)
        if key != self._loaded_key:
            self.track_info = self.page.evaluate("spec => window.loadTrack(spec)", spec)
            self._loaded_key = key
        return self.track_info

    def reset(self, spec: dict | None = None) -> dict:
        if spec is None:
            if self.random_tracks:
                spec = {"seed": random.randrange(1_000_000)}
            else:
                spec = random.choice(self.tracks)
        self.load_track(spec)
        return self.page.evaluate("() => window.resetGame()")

    def step(self, action: int) -> dict:
        return self.page.evaluate("a => window.stepGame(a)", int(action))

    # ----- generation 2: a field of cars (and simulated people) on one track -----
    def setup_world(self, spec: dict, cars: int = 1, random_start: bool = False, tier: str = "n3", speed_cap: float = 1.0,
                    fragile: bool = False, caps: list[float] | None = None, gen: int | None = None) -> dict:
        """tier: "n3" (steering + pedals) or "n4" (steering only). random_start: put the grid at a random
        point round the circuit (training episodes only). gen: a particular brain kind (an earlier brain in a time trial
        sees only the inputs it was built for); by default the class's current one."""
        self.track_info = self.page.evaluate("([s, o]) => window.setupWorld(s, o)",
                                             [spec, {"cars": cars, "randomStart": random_start, "tier": tier, "speedCap": speed_cap,
                                                     "fragile": fragile, "caps": caps or [], "gen": gen}])
        self._loaded_key = None
        return self.track_info

    def reset_world(self) -> list[dict]:
        return self.page.evaluate("() => window.resetWorld()")

    def step_world(self, actions: list[int]) -> list[dict]:
        return self.page.evaluate("a => window.stepWorld(a)", [int(a) for a in actions])

    def close(self) -> None:
        try:
            self._browser.close()
        finally:
            self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
