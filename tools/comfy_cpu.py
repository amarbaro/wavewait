#!/usr/bin/env python3
"""One line per second of a process and the card, stamped with CLOCK_BOOTTIME: is the time with no kernel CPU, disk or waiting?
usage: comfy_cpu.py PID OUT.tsv [CARD]. Columns: ns, cpu_ticks (user + system, all threads), running threads, disk read bytes, rss kB,
vram bytes, GPU busy %, sclk MHz, mclk MHz. Reads only /proc and sysfs. Ends with the process."""
import glob
import os
import sys
import time

pid, out = sys.argv[1:3]
dev = f"/sys/class/drm/{sys.argv[3] if len(sys.argv) > 3 else 'card1'}/device/"


def read(p):
    try:
        return open(p).read()
    except OSError:
        return ""


def now_mhz(name):
    for line in read(dev + name).splitlines():
        if line.endswith("*"):
            return line.split()[1].lower().replace("mhz", "")
    return ""


with open(out, "w", buffering=1) as f:
    while os.path.exists(f"/proc/{pid}/stat"):
        st = read(f"/proc/{pid}/stat").rsplit(")", 1)[-1].split()
        if len(st) < 22:
            break
        running = sum(read(t).rsplit(")", 1)[-1].split()[:1] == ["R"] for t in glob.glob(f"/proc/{pid}/task/*/stat"))
        io = dict(l.split(": ") for l in read(f"/proc/{pid}/io").splitlines())
        rss = next((l.split()[1] for l in read(f"/proc/{pid}/status").splitlines() if l.startswith("VmRSS")), "")
        print(time.clock_gettime_ns(time.CLOCK_BOOTTIME), int(st[11]) + int(st[12]), running, io.get("read_bytes", ""), rss,
              read(dev + "mem_info_vram_used").strip(), read(dev + "gpu_busy_percent").strip(), now_mhz("pp_dpm_sclk"), now_mhz("pp_dpm_mclk"), sep="\t", file=f)
        time.sleep(1)
