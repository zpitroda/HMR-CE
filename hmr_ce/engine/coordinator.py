"""
Master HMR-CE Memory Coordinator
Orchestrates Algorithm 1 (Ingestion Lifecycle) and Algorithm 2 (Hierarchical Traversal)
across Tiers 0, 1, 2, and 3.
"""

import uuid
from typing import Optional, List, Dict, Any
import numpy as np

from hmr_ce.config import HMRCEConfig, default_config
from hmr_ce.schemas import (
    SpeakerRole,
    ActiveStatus,
    SurpriseQuadrant,
    Tier1Record,
    Tier2EpisodicNode,
    TurnIngestResult,
    TraversalResult
)
from hmr_ce.storage.tier0_buffer import Tier0Buffer
from hmr_ce.storage.tier1_store import Tier1Store
from hmr_ce.storage.tier2_store import Tier2Store
from hmr_ce.storage.tier3_store import Tier3Store
from hmr_ce.models.embedding import MRLEmbeddingEngine
from hmr_ce.models.perplexity import PerplexityScorer
from hmr_ce.models.nli import NLIContradictionDetector
from hmr_ce.engine.surprise_gate import DualSignalSurpriseGate
from hmr_ce.engine.invalidation import BeliefInvalidationEngine
from hmr_ce.engine.traversal import HierarchicalTraversalEngine


class HMRCECoordinator:
    def __init__(self, config: Optional[HMRCEConfig] = None, session_id: Optional[str] = None):
        self.config = config or default_config
        self.session_id = session_id or str(uuid.uuid4())

        # Storage
        self.tier0_buffer = Tier0Buffer(max_turns=self.config.tier0_max_turns)
        self.tier1_store = Tier1Store(db_path=self.config.db_path)
        self.tier2_store = Tier2Store(db_path=self.config.db_path)
        self.tier3_store = Tier3Store(db_path=self.config.db_path)

        # ML Models
        self.embedding_engine = MRLEmbeddingEngine(
            d_fine=self.config.d_fine,
            d_coarse=self.config.d_coarse,
            use_mock=self.config.use_mock_models
        )
        self.perplexity_scorer = PerplexityScorer(
            use_mock=self.config.use_mock_models,
            llama_server_url=self.config.llama_server_url
        )
        self.nli_detector = NLIContradictionDetector(
            tau_contradict=self.config.tau_contradict,
            use_mock=self.config.use_mock_models,
            llama_server_url=self.config.llama_server_url
        )

        # Engine Modules
        self.surprise_gate = DualSignalSurpriseGate(self.config, self.perplexity_scorer)
        self.invalidation_engine = BeliefInvalidationEngine(
            self.config, self.tier1_store, self.tier2_store, self.nli_detector
        )
        self.traversal_engine = HierarchicalTraversalEngine(
            self.config, self.tier0_buffer, self.tier1_store,
            self.tier2_store, self.tier3_store, self.embedding_engine
        )

        # Dynamic State
        self.active_cluster_id: Optional[str] = None
        self.cluster_counter: int = 0

    def ingest_turn(
        self,
        text: str,
        speaker: SpeakerRole = SpeakerRole.USER,
        timestamp: Optional[str] = None
    ) -> TurnIngestResult:
        """
        Executes Algorithm 1: Ingestion Lifecycle.
        """
        # 1. Compute MRL Embedding across fine and coarse dimensions
        e_fine, e_coarse = self.embedding_engine.embed(text)

        # Context for perplexity calculation
        working_context = self.tier0_buffer.get_context_text()

        # Temporary next turn ID
        next_turn_id = self.tier1_store.get_max_turn_id() + 1

        # 2. Evaluate Dual-Signal Surprise Gate
        gate_result = self.surprise_gate.evaluate_turn(
            turn_id=next_turn_id,
            text=text,
            embedding_fine=e_fine,
            working_context=working_context
        )

        # 3. Write verbatim payload to Tier 1 Pointer Store
        t1_record = self.tier1_store.insert_turn(
            session_id=self.session_id,
            speaker_role=speaker,
            raw_text=text,
            token_count=len(text.split()),
            active_status=ActiveStatus.ACTIVE,
            timestamp=timestamp
        )

        # Correct gate turn_id if needed
        gate_result.turn_id = t1_record.turn_id

        # 4. Handle Quadrant 4: Routine Interruption
        if gate_result.quadrant == SurpriseQuadrant.ROUTINE_INTERRUPT:
            # Buffer into Tier 0 only; do NOT mutate centroids or spawn Tier 2 node
            self.tier0_buffer.append(t1_record)
            return TurnIngestResult(
                tier1_record=t1_record,
                gate_result=gate_result,
                tier2_node=None,
                tier3_cluster_id=self.active_cluster_id,
                invalidations=[],
                tier0_size=len(self.tier0_buffer)
            )

        # 5. Manage Tier 3 Cluster Boundary
        cluster_just_created = False
        if self.active_cluster_id is None or gate_result.quadrant == SurpriseQuadrant.DOMAIN_PIVOT:
            self.cluster_counter += 1
            self.active_cluster_id = f"cluster_{self.cluster_counter}_{t1_record.turn_id}"
            self.tier3_store.create_cluster(
                cluster_id=self.active_cluster_id,
                initial_coarse=e_coarse,
                turn_id=t1_record.turn_id
            )
            cluster_just_created = True

        # 6. Check for Contradiction / Invalidation against active Tier 2 nodes
        current_node_id = str(uuid.uuid4())
        is_novelty = (gate_result.quadrant == SurpriseQuadrant.CONCEPTUAL_NOVELTY)
        
        invalidations = []
        speaker_val = getattr(speaker, "value", str(speaker)).upper()
        if speaker_val == "USER":
            invalidations = self.invalidation_engine.check_and_apply_invalidation(
                current_node_id=current_node_id,
                current_turn_id=t1_record.turn_id,
                current_text=text,
                current_cluster_id=self.active_cluster_id,
                current_embedding_fine=e_fine,
                is_conceptual_novelty=is_novelty
            )

        # 7. Construct Tier 2 Episodic Node
        abstract_summary = text if len(text) <= 120 else text[:117] + "..."
        t2_node = Tier2EpisodicNode(
            node_id=current_node_id,
            turn_references=[t1_record.turn_id],
            embedding_fine=e_fine.tolist(),
            embedding_coarse=e_coarse.tolist(),
            surprise_salience=gate_result.salience,
            momentum_inherited=(gate_result.active_momentum > 0.0 and gate_result.quadrant == SurpriseQuadrant.EXPECTED_PROGRESS),
            centroid_cluster_id=self.active_cluster_id,
            status=ActiveStatus.ACTIVE,
            metadata={"abstract": abstract_summary, "speaker": speaker.value}
        )
        self.tier2_store.insert_node(t2_node)

        # 8. Update Tier 3 Macro-Centroid (with Surprise Isolation)
        if not gate_result.is_isolated and not cluster_just_created:
            self.tier3_store.aggregate_node(
                cluster_id=self.active_cluster_id,
                node_coarse=e_coarse,
                turn_id=t1_record.turn_id
            )

        # 9. Maintain Tier 0 Ring Buffer
        self.tier0_buffer.append(t1_record)

        return TurnIngestResult(
            tier1_record=t1_record,
            gate_result=gate_result,
            tier2_node=t2_node,
            tier3_cluster_id=self.active_cluster_id,
            invalidations=invalidations,
            tier0_size=len(self.tier0_buffer)
        )

    def retrieve(
        self,
        query: str,
        force_verbatim: bool = False,
        is_retrospective: Optional[bool] = None
    ) -> TraversalResult:
        """
        Executes Algorithm 2: Hierarchical Traversal & Retrieval.
        """
        return self.traversal_engine.retrieve(
            query=query,
            current_turn_id=self.tier1_store.get_max_turn_id(),
            force_verbatim=force_verbatim,
            is_retrospective=is_retrospective
        )

    def get_full_topology_state(self) -> Dict[str, Any]:
        """Returns comprehensive state across Tiers 0, 1, 2, 3 for UI inspection."""
        tier0_items = [t.model_dump() for t in self.tier0_buffer.get_turns()]
        tier1_items = [t.model_dump() for t in self.tier1_store.get_all_turns(self.session_id)]
        tier2_items = [n.model_dump() for n in self.tier2_store.get_all_nodes()]
        tier3_items = [c.model_dump() for c in self.tier3_store.get_all_clusters()]

        return {
            "session_id": self.session_id,
            "active_cluster_id": self.active_cluster_id,
            "active_momentum": self.surprise_gate.active_momentum,
            "tier0": tier0_items,
            "tier1": tier1_items,
            "tier2": tier2_items,
            "tier3": tier3_items
        }

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session memory and state."""
        self.session_id = session_id or str(uuid.uuid4())
        self.tier0_buffer.clear()
        self.tier1_store.clear()
        self.tier2_store.clear()
        self.tier3_store.clear()
        self.surprise_gate.reset()
        self.active_cluster_id = None
        self.cluster_counter = 0

    def close(self) -> None:
        """Gracefully close all underlying storage handles."""
        self.tier1_store.close()
        self.tier2_store.close()
        self.tier3_store.close()
