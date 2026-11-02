#!/usr/bin/env python3
"""ComfyUI's progress events with CLOCK_BOOTTIME stamps, the clock of rocprofv3's kernel trace, so stages line up with dispatches.
Run it with ComfyUI's venv python, which has aiohttp. usage: comfy_stages.py OUT.tsv API.json [HOST:PORT]
Lines: ns, event, node id, class_type (once API.json exists), step, steps. Ends at execution_success or execution_error."""
import asyncio
import json
import os
import sys
import time

import aiohttp


async def main(out, api, host):
    async with aiohttp.ClientSession() as s, s.ws_connect(f"http://{host}/ws?clientId=comfy_stages") as ws:
        with open(out, "w") as f:
            async for m in ws:
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                now = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                e = json.loads(m.data)
                t, d = e.get("type"), e.get("data") or {}
                if t not in ("execution_start", "executing", "progress", "execution_success", "execution_error"):
                    continue
                node = str(d.get("node") or "")
                nodes = json.load(open(api)) if os.path.exists(api) else {}
                cls = nodes.get(node, {}).get("class_type", "")
                print(now, t, node, cls, d.get("value", ""), d.get("max", ""), sep="\t", file=f, flush=True)
                if t in ("execution_success", "execution_error"):
                    return


asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1:8188"))
