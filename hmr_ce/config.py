"""
Configuration and hyperparameters for HMR-CE (Hierarchical Multi-Resolution Context Engine).
All values follow the Technical Whitepaper & Proof-of-Concept Implementation Blueprint
and the empirical findings from Titans, Miras, ZSinvert, and ReverseEOL.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
import os


@dataclass
class HMRCEConfig:
    # --- Matryoshka Representation Learning (MRL) Dimensions ---
    d_coarse: int = 64          # Coarse prefix dimension for Tier 3 topological sweeps
    d_fine: int = 1024          # Full fine dimension for Tier 2 episodic re-ranking (or 768)
    
    # --- Dual-Signal Surprise Gating Parameters (Titans Framework) ---
    beta_drift: float = 0.80    # Exponential smoothing factor for running topic centroid C_t in [0.75, 0.90]
    theta_dist: float = 0.50    # Normalized cosine drift threshold S_topic separating on-topic vs. pivot
    theta_ppl: float = 40.0     # Context-conditioned perplexity threshold P_entropy separating routine vs. novel
    
    # --- Titans Exponential Momentum Propagation ---
    momentum_decay_gamma: float = 0.75  # Decay multiplier gamma in [0.70, 0.85]
    momentum_horizon_M: int = 4         # Forward turn window M in [3, 5]
    
    # --- Directed Belief Revision & Supersession ---
    tau_contradict: float = 0.82        # NLI contradiction confidence threshold in [0.80, 0.90]
    
    # --- Hierarchical Retrieval & Scoring Formulation ---
    lambda_floor: float = 0.20          # Retention floor lambda in [0.15, 0.30]
    decay_tau: float = 10.0             # Temporal decay half-life in turns
    omega_surp: float = 0.25            # Surprise weighting parameter in [0.20, 0.35]
    h_thresh: float = 0.85              # Verbatim hydration decision threshold
    top_k_clusters: int = 3             # Top-K candidate clusters in Phase 1 Macro-Sweep
    top_n_nodes: int = 3                # Top-N candidate episodic nodes in Phase 2 Fine Search
    
    # --- Tier 0 Active Context Buffer ---
    tier0_max_turns: int = 6            # Sliding ring buffer capacity K
    
    # --- Storage & Paths ---
    db_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    db_name: str = "hmr_ce_memory.db"
    
    # --- Local Inference & LLM Configuration ---
    llama_server_url: str = "http://127.0.0.1:8080"
    llm_model_name: str = "qwen3.8:27B"
    use_mock_models: bool = False       # True for ultra-fast unit testing, False for live models
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # --- Server Settings ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    
    @property
    def db_path(self) -> Union[Path, str]:
        if str(self.db_name) == ":memory:":
            return ":memory:"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        return self.db_dir / self.db_name


# Default global instance
default_config = HMRCEConfig()
