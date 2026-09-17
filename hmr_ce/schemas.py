"""
Data contract schemas for HMR-CE according to the Technical Whitepaper.
All records and events are defined as typed Pydantic models.
"""

from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np


class SpeakerRole(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"


class ActiveStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    MUTED = "MUTED"


class SurpriseQuadrant(str, Enum):
    DOMAIN_PIVOT = "DOMAIN_PIVOT"               # Q1: High S_topic, High P_entropy
    CONCEPTUAL_NOVELTY = "CONCEPTUAL_NOVELTY"   # Q2: Low S_topic, High P_entropy (Unconventional Thesis)
    EXPECTED_PROGRESS = "EXPECTED_PROGRESS"     # Q3: Low S_topic, Low P_entropy (Routine Continuation)
    ROUTINE_INTERRUPT = "ROUTINE_INTERRUPT"     # Q4: High S_topic, Low P_entropy (Pleasantry/Interruption)


class Tier1Record(BaseModel):
    """
    Tier 1 Pointer Store Record:
    Immutable, append-only disk storage containing raw unmodified transcript payloads.
    Indexed monotonically by turn_id.
    """
    turn_id: int = Field(..., description="Monotonic primary key sequence index")
    session_id: str = Field(..., description="Conversation session UUID or string")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO-8601 UTC timestamp")
    speaker_role: SpeakerRole = Field(..., description="Speaker role enum")
    raw_text: str = Field(..., description="Unmodified verbatim payload")
    token_count: int = Field(default=0, description="Tokens calculated by tokenizer")
    active_status: ActiveStatus = Field(default=ActiveStatus.ACTIVE, description="State of validity")


class Tier2EpisodicNode(BaseModel):
    """
    Tier 2 Episodic Memory Store Record:
    Vector-indexed record representing an atomic assertion, subtask, or conversational episode.
    """
    node_id: str = Field(..., description="Unique UUID for episodic node")
    turn_references: List[int] = Field(default_factory=list, description="Foreign keys mapping to Tier 1 turn_id records")
    embedding_fine: Optional[List[float]] = Field(None, description="MRL embedding vector (d=768/1024/1536)")
    embedding_coarse: Optional[List[float]] = Field(None, description="MRL prefix slice vector (d=64)")
    surprise_salience: float = Field(..., description="Test-time salience score [0.0, 1.0]")
    momentum_inherited: bool = Field(default=False, description="Whether salience was assigned via Titans decay window")
    centroid_cluster_id: str = Field(..., description="Reference to parent Tier 3 cluster")
    superseded_by_node: Optional[str] = Field(None, description="Reference to succeeding node that invalidates this node")
    status: ActiveStatus = Field(default=ActiveStatus.ACTIVE, description="ACTIVE or SUPERSEDED")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata including abstract summary")


class Tier3MacroCentroid(BaseModel):
    """
    Tier 3 Topological Macro-Centroid:
    A compact index of conversational themes, tasks, or temporal phases.
    """
    cluster_id: str = Field(..., description="Unique cluster identifier")
    centroid_coarse: List[float] = Field(..., description="Unit-normalized prefix slice (d=64)")
    member_count: int = Field(default=1, description="Number of constituent Tier 2 nodes")
    bounding_turn_range: Tuple[int, int] = Field(..., description="Span of [min_turn_id, max_turn_id]")


class SurpriseGateResult(BaseModel):
    """Result of the Dual-Signal Surprise Gate classification for an incoming turn."""
    turn_id: int
    s_topic: float
    p_entropy: float
    quadrant: SurpriseQuadrant
    salience: float
    active_momentum: float
    is_isolated: bool
    label: str
    explanation: str


class ContradictionResult(BaseModel):
    """Result of NLI belief invalidation check."""
    is_contradiction: bool
    confidence: float
    target_node_id: Optional[str] = None
    target_turn_id: Optional[int] = None
    target_text: Optional[str] = None
    explanation: str = ""


class ScoredEpisodicNode(BaseModel):
    """Scored Tier 2 node during Phase 2 retrieval."""
    node_id: str
    centroid_cluster_id: str
    turn_references: List[int]
    sim_fine: float
    delta_t: int
    time_decay: float
    surprise_salience: float
    composite_score: float
    status: ActiveStatus
    hydrated_verbatim: Optional[str] = None
    abstract: str = ""


class TraversalResult(BaseModel):
    """Output of the 3-Phase Hierarchical Retrieval & Hydration lifecycle."""
    query: str
    candidate_clusters: List[str]
    scored_nodes: List[ScoredEpisodicNode]
    context_blocks: List[str]
    tier0_working_context: List[str]
    assembled_prompt_context: str
    requires_verbatim: bool
    is_retrospective: bool


class TurnIngestResult(BaseModel):
    """Comprehensive telemetry returned when a new turn is ingested."""
    tier1_record: Tier1Record
    gate_result: SurpriseGateResult
    tier2_node: Optional[Tier2EpisodicNode] = None
    tier3_cluster_id: Optional[str] = None
    invalidations: List[ContradictionResult] = Field(default_factory=list)
    tier0_size: int
