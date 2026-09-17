"""
Single-Command Launcher for HMR-CE Live Proof of Concept.
Starts the FastAPI backend, mounts the interactive dark-tech visualizer,
and connects to the local llama-server (Qwen 3.8 27B) and RTX 5090.
"""

import sys
import os
import time
import webbrowser
import urllib.request
import uvicorn

# Ensure hmr_ce is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from hmr_ce.config import default_config
from hmr_ce.api.server import create_app


def print_banner():
    banner = """
=============================================================================
  HMR-CE: Hierarchical Multi-Resolution Context Engine
  Surprise-Gated Memory with Titans Dynamics & Matryoshka Traversal
=============================================================================
  [Tier 0] Active Working Context Buffer (Sliding Window K=6)
  [Tier 1] Ground-Truth Pointer Store (Immutable SQLite WAL)
  [Tier 2] Episodic Memory Store (Fine MRL d=1024 + Belief DAG)
  [Tier 3] Topological Macro-Centroid Map (Coarse MRL d=64)
-----------------------------------------------------------------------------
  Pillars:
  * Matryoshka Representation Learning (MRL) + ReverseEOL
  * Dual-Signal Surprise Gating (Topical Drift + Token Perplexity)
  * Titans Forward Momentum Window (alpha_{t+m} = alpha_t * gamma^m)
  * Directed Belief Revision & Supersession Graph Links
  * Bi-directional Pointer Hydration (Verbatim Text vs. Summary)
=============================================================================
"""
    print(banner)


def check_llama_server():
    url = f"{default_config.llama_server_url}/v1/models"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                print(f"[+] Local LLM Server detected online at {default_config.llama_server_url}")
                return True
    except Exception:
        pass
    print(f"[-] Local LLM Server not detected at {default_config.llama_server_url} (Fallback mode active)")
    return False


def main():
    print_banner()
    check_llama_server()
    
    app = create_app(default_config)

    host = default_config.api_host
    port = default_config.api_port
    url = f"http://{host}:{port}"
    print(f"\n[+] Launching HMR-CE Dashboard at: {url}")
    print("[+] Press CTRL+C to terminate the server.\n")

    # Launch uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
