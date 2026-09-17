"""
Hierarchical Traversal & Retrieval Protocol (Section 5 & Algorithm 2)
Executes the three-phase coarse-to-fine traversal pipeline:
- Phase 1: Coarse Pruning (d=64 MRL dot-product sweeps against Tier 3 centroids)
- Phase 2: Fine Episodic Re-Ranking (d=1024 directional cosine + temporal decay + surprise salience)
- Phase 3: Bi-Directional Pointer Hydration Gate (verbatim prompt injection vs. compressed summary)
"""

import re
import math
from typing import List, Optional, Tuple
import numpy as np

from hmr_ce.config import HMRCEConfig
from hmr_ce.schemas import (
    ScoredEpisodicNode,
    TraversalResult,
    ActiveStatus
)
from hmr_ce.storage.tier0_buffer import Tier0Buffer
from hmr_ce.storage.tier1_store import Tier1Store
from hmr_ce.storage.tier2_store import Tier2Store
from hmr_ce.storage.tier3_store import Tier3Store
from hmr_ce.models.embedding import MRLEmbeddingEngine


class HierarchicalTraversalEngine:
    def __init__(
        self,
        config: HMRCEConfig,
        tier0_buffer: Tier0Buffer,
        tier1_store: Tier1Store,
        tier2_store: Tier2Store,
        tier3_store: Tier3Store,
        embedding_engine: MRLEmbeddingEngine
    ):
        self.config = config
        self.tier0_buffer = tier0_buffer
        self.tier1_store = tier1_store
        self.tier2_store = tier2_store
        self.tier3_store = tier3_store
        self.embedding_engine = embedding_engine

    def query_requires_verbatim(self, query: str) -> bool:
        """
        Hydration Gate heuristic: Detects if query demands exact code, exact syntax,
        parameters, configuration values, or quotes.
        """
        q_lower = query.lower()
        verbatim_triggers = [
            r"\b(?:code|syntax|exact|verbatim|quote|config|command|parameter|port|key|value)\b",
            r"\b(?:sql|json|script|function|class|method|prompt|api)\b",
            r"['\"`]"  # Quoted strings
        ]
        return any(re.search(pat, q_lower) for pat in verbatim_triggers)

    def is_retrospective_query(self, query: str) -> bool:
        """
        Detects retrospective debugging queries that intentionally ask about
        superseded/past premises (e.g. 'Why did we decide against DNS?').
        """
        q_lower = query.lower()
        retrospective_triggers = [
            r"\b(?:why did we|why were|why was|decide against|switch from|previously|earlier plan)\b",
            r"\b(?:what happened to|why not|history of|retrospect)\b"
        ]
        return any(re.search(pat, q_lower) for pat in retrospective_triggers)

    def retrieve(
        self,
        query: str,
        current_turn_id: Optional[int] = None,
        force_verbatim: bool = False,
        is_retrospective: Optional[bool] = None
    ) -> TraversalResult:
        """
        Executes Algorithm 2: Coarse-to-Fine Retrieval Lifecycle.
        """
        if current_turn_id is None:
            current_turn_id = self.tier1_store.get_max_turn_id()

        if is_retrospective is None:
            is_retrospective = self.is_retrospective_query(query)

        requires_verbatim = force_verbatim or self.query_requires_verbatim(query)

        # 1. Generate query embeddings & extract MRL slices
        q_fine, q_coarse = self.embedding_engine.embed(query)

        # 2. Phase 1: Macro-Sweep (Tier 3)
        clusters = self.tier3_store.get_all_clusters()
        candidate_cluster_ids: List[str] = []

        if clusters:
            scored_clusters = []
            for cl in clusters:
                c_vec = np.array(cl.centroid_coarse, dtype=np.float32)
                # Dot product of unit-normalized d=64 coarse vectors
                score = float(np.dot(q_coarse, c_vec))
                scored_clusters.append((score, cl.cluster_id))
            
            scored_clusters.sort(key=lambda x: x[0], reverse=True)
            candidate_cluster_ids = [cid for _, cid in scored_clusters[:self.config.top_k_clusters]]

        # 3. Phase 2: Episodic Search (Tier 2)
        candidate_nodes = []
        if candidate_cluster_ids:
            if is_retrospective:
                # Include superseded nodes in candidate clusters
                for cid in candidate_cluster_ids:
                    candidate_nodes.extend(self.tier2_store.get_nodes_by_cluster(cid, only_active=False))
            else:
                # Standard search filters WHERE status = 'ACTIVE'
                candidate_nodes = self.tier2_store.get_active_nodes_in_clusters(candidate_cluster_ids)

        # Fallback: if no nodes in clusters yet, check all nodes
        if not candidate_nodes:
            candidate_nodes = self.tier2_store.get_all_nodes(only_active=not is_retrospective)

        scored_nodes: List[ScoredEpisodicNode] = []
        lambda_floor = self.config.lambda_floor
        tau = self.config.decay_tau
        omega_surp = self.config.omega_surp

        for node in candidate_nodes:
            if not node.embedding_fine:
                continue

            node_vec = np.array(node.embedding_fine, dtype=np.float32)
            n_norm = np.linalg.norm(node_vec)
            if n_norm > 1e-8:
                sim_fine = max(0.0, float(np.dot(q_fine, node_vec) / n_norm))
            else:
                sim_fine = 0.0

            delta_t = current_turn_id - max(node.turn_references) if node.turn_references else 0
            delta_t = max(0, delta_t)

            # Temporal decay envelope T(Delta t) = lambda + (1 - lambda) * exp(-Delta t / tau)
            time_decay = float(lambda_floor + (1.0 - lambda_floor) * math.exp(-delta_t / tau))

            # Composite ranking score: (sim_fine * time_decay) + (omega_surp * surprise_salience)
            composite_score = float((sim_fine * time_decay) + (omega_surp * node.surprise_salience))

            scored_nodes.append(ScoredEpisodicNode(
                node_id=node.node_id,
                centroid_cluster_id=node.centroid_cluster_id,
                turn_references=node.turn_references,
                sim_fine=sim_fine,
                delta_t=delta_t,
                time_decay=time_decay,
                surprise_salience=node.surprise_salience,
                composite_score=composite_score,
                status=node.status,
                abstract=node.metadata.get("abstract", "")
            ))

        # Sort candidate nodes by composite_score descending
        scored_nodes.sort(key=lambda x: x.composite_score, reverse=True)
        top_n_nodes = scored_nodes[:self.config.top_n_nodes]

        # 4. Phase 3: Hydration Decision Gate
        context_blocks = []
        for scored in top_n_nodes:
            hydrate_verbatim = requires_verbatim or is_retrospective or (scored.composite_score >= self.config.h_thresh)
            status_flag = " [SUPERSEDED]" if scored.status == ActiveStatus.SUPERSEDED else ""
            
            if hydrate_verbatim:
                # Fetch raw verbatim text from Tier 1 SQLite via turn_references
                t1_records = self.tier1_store.get_turns_batch(scored.turn_references)
                verbatim_texts = [f"[{r.speaker_role.value} Turn {r.turn_id}]: {r.raw_text}" for r in t1_records]
                full_text = " | ".join(verbatim_texts)
                scored.hydrated_verbatim = full_text

                context_blocks.append(f"[Turn verbatim{status_flag}] {full_text}")
            else:
                # Inject compressed semantic summary/abstract
                abstract = scored.abstract or f"Episodic Node {scored.node_id[:8]} across turns {scored.turn_references}"
                context_blocks.append(f"[Summary/Episodic Reference{status_flag}] {abstract}")

        # 5. Combine with Tier 0 Working Context Buffer
        tier0_turns = [f"[{t.speaker_role.value} Turn {t.turn_id}]: {t.raw_text}" for t in self.tier0_buffer.get_turns()]

        sections = []
        if context_blocks:
            sections.append("=== HYDRATED LONG-TERM MEMORY (Tiers 1 & 2) ===\n" + "\n".join(context_blocks))
        if tier0_turns:
            sections.append("=== ACTIVE WORKING CONTEXT (Tier 0) ===\n" + "\n".join(tier0_turns))

        assembled_prompt_context = "\n\n".join(sections)

        return TraversalResult(
            query=query,
            candidate_clusters=candidate_cluster_ids,
            scored_nodes=top_n_nodes,
            context_blocks=context_blocks,
            tier0_working_context=tier0_turns,
            assembled_prompt_context=assembled_prompt_context,
            requires_verbatim=requires_verbatim,
            is_retrospective=is_retrospective
        )
