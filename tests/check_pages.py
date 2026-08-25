#!/usr/bin/env python
"""Load each GUI's page script in Node against a stub DOM and fail on any error.

    python tests/check_pages.py            # starts each server, checks, stops it

`node --check` only parses. It cannot see a `const` referenced before its
declaration, which is a ReferenceError that kills the entire script at load and
leaves every button dead and every panel black -- exactly the failure this
catches. The stub is deliberately dumb: enough DOM for the page to build its
controls and attach its handlers, and nothing more.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PAGES = [("scripts/gui.py", 8931), ("scripts/gui_image.py", 8932)]

STUB = r"""
// Minimal DOM: every element answers every call, so the page can build itself.
const noop = () => {};
function ctx() {
  return new Proxy({}, {get: (t, k) => {
    if (k === "canvas") return {width: 330, height: 460};
    if (["fillStyle","strokeStyle","lineWidth","font","globalAlpha",
         "imageSmoothingEnabled","textAlign"].includes(k)) return "";
    return noop;
  }, set: () => true});
}
function el(id) {
  const e = {
    id: id || "", innerHTML: "", textContent: "", value: 0, type: "",
    width: 330, height: 460, style: {},
    children: [], className: "", checked: false,
    classList: {add: noop, remove: noop, toggle: noop, contains: () => false},
    appendChild: c => { e.children.push(c); return c; },
    append: (...c) => { e.children.push(...c); },
    setAttribute: noop, getAttribute: () => null, addEventListener: noop,
    removeEventListener: noop, getContext: ctx, focus: noop, click: noop,
    getBoundingClientRect: () => ({left: 0, top: 0, width: 330, height: 460}),
  };
  return e;
}
const ELS = {};
global.document = {
  getElementById: id => (ELS[id] = ELS[id] || el(id)),
  createElement: tag => el(""),
  addEventListener: noop, querySelector: () => el(""),
  querySelectorAll: () => [],
};
global.window = global;
global.Image = class { constructor() { this.width = 330; this.height = 460; } };
global.fetch = () => Promise.resolve({json: () => Promise.resolve({
  running: false, step: 0, steps: 0, seconds: 0, curve: [], metrics: {},
  images: {}, grid: {}, blocks: {}, ladder: [], history: [], note: "",
  stamp: 0, info: {}})});
global.setInterval = () => 0;
global.clearInterval = noop;
global.setTimeout = (f) => 0;
global.clearTimeout = noop;
global.requestAnimationFrame = () => 0;
"""


def check(script: str, port: int) -> bool:
    proc = subprocess.Popen([PY, "-u", os.path.join(ROOT, script), "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        html = None
        for _ in range(40):
            time.sleep(1.0)
            if proc.poll() is not None:
                print(f"  {script}: server exited: "
                      f"{proc.stderr.read().decode()[-300:]}")
                return False
            try:
                html = urllib.request.urlopen(f"http://127.0.0.1:{port}/",
                                              timeout=2).read().decode()
                break
            except Exception:
                continue
        if html is None:
            print(f"  {script}: server never answered")
            return False
        left = [t for t in re.findall(r"__[A-Z_]+__", html)]
        if left:
            print(f"  {script}: unreplaced placeholders {sorted(set(left))}")
            return False
        # The explainer is a modal. Without the rule that hides it, it renders
        # inline and pushes the panels off the page -- which looks exactly like
        # "clicking run does nothing".
        if 'class="modal"' in html and ".modal { position:fixed" not in html:
            print(f"  {script}: modal markup present but the stylesheet that "
                  "hides it is not")
            return False
        js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        path = f"/tmp/_page_{port}.js"
        with open(path, "w") as f:
            f.write(STUB + "\n" + js)
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"  {script}: page script threw\n"
                  + "\n".join(r.stderr.strip().splitlines()[:12]))
            return False
        print(f"  {script}: page builds cleanly ({len(js):,} chars of script)")
        return True
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    ok = all([check(s, p) for s, p in PAGES])
    print("\nall pages ok" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
