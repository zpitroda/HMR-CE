"""
Dual-Signal Surprise Gate & Titans Momentum Engine
Operationalizes prediction error at inference time by decoupling surprise into:
1. Topical Semantic Drift: S_topic(x_t) = 1 - cos(e_t, C_{t-1})
2. Context-Conditioned Token Perplexity: P_entropy(x_t | C)

Evaluates the 4-Quadrant Surprise Classification Matrix and propagates Titans-style
exponential forward momentum: alpha_{t+m} = alpha_t * gamma^m.
"""

from typing import Optional, Tuple
import re
import numpy as np

from hmr_ce.config import HMRCEConfig
from hmr_ce.schemas import SurpriseQuadrant, SurpriseGateResult
from hmr_ce.models.perplexity import PerplexityScorer


class DualSignalSurpriseGate:
    def __init__(self, config: HMRCEConfig, perplexity_scorer: Optional[PerplexityScorer] = None):
        self.config = config
        self.perplexity_scorer = perplexity_scorer or PerplexityScorer(use_mock=config.use_mock_models)
        
        # Internal state
        self.c_rolling: Optional[np.ndarray] = None  # Exponentially smoothed topic centroid
        self.last_e: Optional[np.ndarray] = None     # Prior turn embedding for discourse anaphora
        self.active_momentum: float = 0.0           # Running Titans surprise momentum
        self.momentum_steps_remaining: int = 0      # Horizon M counter

    def evaluate_turn(
        self,
        turn_id: int,
        text: str,
        embedding_fine: np.ndarray,
        working_context: str = ""
    ) -> SurpriseGateResult:
        """
        Executes Ingestion Lifecycle surprise gating (Section 3 & Algorithm 1).
        Computes S_topic, P_entropy, classifies into one of 4 quadrants,
        and applies Titans momentum propagation.
        """
        # Ensure unit norm
        e_norm = np.linalg.norm(embedding_fine)
        e_t = (embedding_fine / e_norm).astype(np.float32) if e_norm > 1e-8 else embedding_fine.astype(np.float32)

        # 1. Compute Context-Conditioned Token Perplexity P_entropy
        p_entropy = self.perplexity_scorer.calculate_perplexity(text, context=working_context)

        # 2. Initial Turn Handling
        if self.c_rolling is None:
            self.c_rolling = e_t.copy()
            if p_entropy > self.config.theta_ppl:
                self.active_momentum = 1.0
                self.momentum_steps_remaining = self.config.momentum_horizon_M
                quadrant = SurpriseQuadrant.CONCEPTUAL_NOVELTY
                label = "INITIAL_NOVELTY"
                is_isolated = True
            else:
                self.active_momentum = 0.0
                self.momentum_steps_remaining = 0
                quadrant = SurpriseQuadrant.EXPECTED_PROGRESS
                label = "INITIAL_TURN"
                is_isolated = False
            
            return SurpriseGateResult(
                turn_id=turn_id,
                s_topic=0.0,
                p_entropy=p_entropy,
                quadrant=quadrant,
                salience=1.0,
                active_momentum=self.active_momentum,
                is_isolated=is_isolated,
                label=label,
                explanation="Initial conversation turn: instantiated primary topic centroid."
            )

        # 3. Compute Normalized Cosine Topical Distance S_topic
        c_norm = np.linalg.norm(self.c_rolling)
        if c_norm > 1e-8:
            cos_c = float(np.dot(e_t, self.c_rolling) / c_norm)
        else:
            cos_c = 1.0

        # Account for conversational anaphora and immediate follow-ups
        if self.last_e is not None:
            l_norm = np.linalg.norm(self.last_e)
            cos_l = float(np.dot(e_t, self.last_e) / l_norm) if l_norm > 1e-8 else cos_c
        else:
            cos_l = cos_c

        # Effective cosine similarity incorporates both macro centroid and discourse referent
        cosine_sim = max(cos_c, cos_l)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        s_topic = float(max(0.0, min(1.0, 1.0 - cosine_sim)))

        # Check for conversational retraction speech acts ("just kidding", "nevermind", "scratch that", etc.)
        retraction_pattern = r"\b(?:just\s+kidding|jk|kidding|nevermind|never\s+mind|scratch\s+that|ignore\s+that|disregard|take\s+(?:it|that)\s+back|i\s+was\s+joking|i'm\s+joking|retract\s+that|forget\s+that)\b"
        is_retraction = bool(re.search(retraction_pattern, text.lower()))
        if is_retraction:
            # Conversational retractions inherently target the active discourse premise
            s_topic = min(s_topic, 0.35)

        # 4. Decay active momentum from previous turns (Titans rule)
        if self.momentum_steps_remaining > 0:
            self.active_momentum *= self.config.momentum_decay_gamma
            self.momentum_steps_remaining -= 1
        else:
            self.active_momentum = 0.0

        # 5. Evaluate Gating Matrix
        theta_d = self.config.theta_dist
        theta_p = self.config.theta_ppl

        if s_topic <= theta_d and p_entropy > theta_p:
            # Quadrant 2: Conceptual Novelty / Unconventional Thesis
            quadrant = SurpriseQuadrant.CONCEPTUAL_NOVELTY
            label = "CONCEPTUAL_NOVELTY"
            salience = 1.0
            self.active_momentum = 1.0
            self.momentum_steps_remaining = self.config.momentum_horizon_M
            is_isolated = True  # Surprise Isolation: bypass Tier 3 centroid aggregation
            explanation = "On-topic vocabulary but high contextual perplexity; flagged high-value novelty and triggered Titans momentum window."

            # Do NOT update c_rolling here (prevents semantic smearing of unconventional thesis)

        elif s_topic > theta_d and p_entropy > theta_p:
            # Quadrant 1: Domain / Topic Pivot
            quadrant = SurpriseQuadrant.DOMAIN_PIVOT
            label = "TOPIC_PIVOT"
            salience = 0.8
            # Abrupt topic pivot: reset running centroid to current turn
            self.c_rolling = e_t.copy()
            self.active_momentum = 0.0  # Reset momentum on topic pivot
            self.momentum_steps_remaining = 0
            is_isolated = False
            explanation = "High topical drift and high perplexity; finalized prior cluster and re-initialized running centroid."

        elif s_topic <= theta_d and p_entropy <= theta_p:
            # Quadrant 3: Expected Continuation
            quadrant = SurpriseQuadrant.EXPECTED_PROGRESS
            label = "EXPECTED_PROGRESS"
            salience = float(max(0.2, self.active_momentum))
            is_isolated = False
            explanation = "Expected conversational progression; updated running centroid with baseline exponential smoothing."

            # Update running centroid
            beta = self.config.beta_drift
            updated_c = beta * self.c_rolling + (1.0 - beta) * e_t
            u_norm = np.linalg.norm(updated_c)
            self.c_rolling = (updated_c / u_norm).astype(np.float32) if u_norm > 1e-8 else updated_c

        else:
            # Quadrant 4: Routine Interruption (Miras Coping Mechanism)
            quadrant = SurpriseQuadrant.ROUTINE_INTERRUPT
            label = "ROUTINE_INTERRUPT"
            salience = 0.1
            is_isolated = True  # Do NOT mutate active semantic centroids or spawn Tier 2 nodes
            explanation = "Predictable syntactic chatter or interruption without conceptual shift; isolated from memory mutations."

        if quadrant != SurpriseQuadrant.ROUTINE_INTERRUPT:
            self.last_e = e_t.copy()

        return SurpriseGateResult(
            turn_id=turn_id,
            s_topic=s_topic,
            p_entropy=p_entropy,
            quadrant=quadrant,
            salience=salience,
            active_momentum=self.active_momentum,
            is_isolated=is_isolated,
            label=label,
            explanation=explanation
        )

    def reset(self) -> None:
        """Reset gating state."""
        self.c_rolling = None
        self.last_e = None
        self.active_momentum = 0.0
        self.momentum_steps_remaining = 0
