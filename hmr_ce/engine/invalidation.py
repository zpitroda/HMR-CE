"""
Directed Belief Revision & Supersession Logic (Section 4)
Enforces directed acyclic state transitions across memory nodes to resolve
the Belief Invalidation failure mode. Refuted premises are transitioned
from ACTIVE to SUPERSEDED with explicit directed links, suppressing them in
standard retrieval while maintaining complete historical traceability.
"""

from typing import List, Optional
import re
import numpy as np

from hmr_ce.config import HMRCEConfig
from hmr_ce.schemas import (
    Tier2EpisodicNode,
    ContradictionResult,
    ActiveStatus
)
from hmr_ce.storage.tier2_store import Tier2Store
from hmr_ce.storage.tier1_store import Tier1Store
from hmr_ce.models.nli import NLIContradictionDetector


class BeliefInvalidationEngine:
    def __init__(
        self,
        config: HMRCEConfig,
        tier1_store: Tier1Store,
        tier2_store: Tier2Store,
        nli_detector: Optional[NLIContradictionDetector] = None
    ):
        self.config = config
        self.tier1_store = tier1_store
        self.tier2_store = tier2_store
        self.nli_detector = nli_detector or NLIContradictionDetector(
            tau_contradict=config.tau_contradict,
            use_mock=config.use_mock_models
        )

    def check_and_apply_invalidation(
        self,
        current_node_id: str,
        current_turn_id: int,
        current_text: str,
        current_cluster_id: str,
        current_embedding_fine: np.ndarray,
        is_conceptual_novelty: bool = False
    ) -> List[ContradictionResult]:
        """
        Scans active, high-salience Tier 2 nodes within the same topic space.
        Evaluates NLI contradiction and applies directed supersession links if confidence >= tau_contradict.
        """
        # 1. Gather all active nodes in current cluster
        cluster_nodes = [n for n in self.tier2_store.get_nodes_by_cluster(current_cluster_id, only_active=True) if n.node_id != current_node_id]
        # 2. Also gather active nodes across other clusters in case cross-cluster revision occurs
        all_active = [n for n in self.tier2_store.get_all_nodes(only_active=True) if n.node_id != current_node_id]

        # Prioritize cluster nodes first, then other active nodes
        seen_ids = set()
        candidate_nodes = []
        for n in cluster_nodes + all_active:
            if n.node_id not in seen_ids:
                seen_ids.add(n.node_id)
                is_assistant = False
                if n.metadata and str(n.metadata.get("speaker", "")).lower() == "assistant":
                    is_assistant = True
                elif n.turn_references:
                    t1_cand = self.tier1_store.get_turn(n.turn_references[-1])
                    if t1_cand:
                        role_str = getattr(t1_cand.speaker_role, "value", str(t1_cand.speaker_role))
                        if role_str.lower() == "assistant":
                            is_assistant = True
                if not is_assistant:
                    candidate_nodes.append(n)

        invalidations: List[ContradictionResult] = []
        if not candidate_nodes:
            return invalidations

        # 0. Check for Conversational Retraction speech acts ("just kidding", "nevermind", "scratch that", etc.)
        retraction_pattern = r"\b(?:just\s+kidding|jk|kidding|nevermind|never\s+mind|scratch\s+that|ignore\s+that|disregard|take\s+(?:it|that)\s+back|i\s+was\s+joking|i'm\s+joking|retract\s+that|forget\s+that)\b"
        if re.search(retraction_pattern, current_text.lower()):
            # Sort candidate nodes by turn_references descending to locate the most recent active user premise
            sorted_candidates = sorted(
                candidate_nodes,
                key=lambda n: n.turn_references[-1] if n.turn_references else 0,
                reverse=True
            )
            target_node = None
            for cand in sorted_candidates:
                if cand.turn_references:
                    t1_rec = self.tier1_store.get_turn(cand.turn_references[-1])
                    if t1_rec:
                        role_str = t1_rec.speaker_role.value if hasattr(t1_rec.speaker_role, 'value') else str(t1_rec.speaker_role)
                        if role_str.upper() == "USER":
                            target_node = cand
                            break
            if not target_node and sorted_candidates:
                target_node = sorted_candidates[0]

            if target_node and target_node.turn_references:
                target_turn_id = target_node.turn_references[-1]
                t1_target = self.tier1_store.get_turn(target_turn_id)
                premise_text = t1_target.raw_text if t1_target else ""

                self.tier2_store.mark_superseded(target_node.node_id, current_node_id)
                self.tier1_store.update_status(target_turn_id, ActiveStatus.SUPERSEDED)
                target_node.surprise_salience = 0.0
                target_node.status = ActiveStatus.SUPERSEDED
                target_node.superseded_by_node = current_node_id
                self.tier2_store.insert_node(target_node)

                res = ContradictionResult(
                    is_contradiction=True,
                    confidence=0.98,
                    target_node_id=target_node.node_id,
                    target_turn_id=target_turn_id,
                    target_text=premise_text,
                    explanation=f"Retracted via conversational speech act: '{current_text}'"
                )
                invalidations.append(res)
                return invalidations

        for node in candidate_nodes:
            if node.node_id == current_node_id:
                continue

            # Prioritize evaluating high-salience nodes or semantically close premises
            # Fetch raw text of target node from Tier 1
            if not node.turn_references:
                continue
            
            target_turn_id = node.turn_references[-1]
            t1_record = self.tier1_store.get_turn(target_turn_id)
            if not t1_record:
                continue
            
            premise_text = t1_record.raw_text

            # Run NLI Contradiction Evaluation
            res = self.nli_detector.check_contradiction(
                premise=premise_text,
                hypothesis=current_text,
                target_node_id=node.node_id,
                target_turn_id=target_turn_id
            )

            if res.is_contradiction:
                # 1. State Transition: ACTIVE -> SUPERSEDED
                self.tier2_store.mark_superseded(node.node_id, current_node_id)
                # 2. Update Tier 1 record status
                self.tier1_store.update_status(target_turn_id, ActiveStatus.SUPERSEDED)
                # 3. Suppress salience in target node record
                node.surprise_salience = 0.0
                node.status = ActiveStatus.SUPERSEDED
                node.superseded_by_node = current_node_id
                self.tier2_store.insert_node(node)

                invalidations.append(res)

        return invalidations
