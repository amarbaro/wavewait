#!/usr/bin/env python3
"""Runs ComfyUI's main.py with a 5 Hz sampler of the prompt thread's Python stack: which function holds the non-kernel time.
No ptrace (a live server must not be attached to). usage: comfy_frames.py OUT.tsv main.py [ARGS...]; lines: ns, frames leaf first.
A C call that holds the GIL delays the next sample: its frame then spans the gap."""
import os
import runpy
import sys
import threading
import time


def sampler(path):
    with open(path, "w", buffering=1) as f:
        while True:
            time.sleep(0.2)
            frames = sys._current_frames()
            for t in threading.enumerate():
                fr = frames.get(t.ident) if "prompt_worker" in t.name else None
                stack = []
                while fr is not None and len(stack) < 8:
                    stack.append(f"{os.path.basename(fr.f_code.co_filename)}:{fr.f_code.co_name}:{fr.f_lineno}")
                    fr = fr.f_back
                if stack:
                    print(time.clock_gettime_ns(time.CLOCK_BOOTTIME), " < ".join(stack), sep="\t", file=f)


out, script = sys.argv[1:3]
sys.argv = sys.argv[2:]
sys.path[0] = os.getcwd()
threading.Thread(target=sampler, args=(out,), daemon=True).start()
runpy.run_path(script, run_name="__main__")
