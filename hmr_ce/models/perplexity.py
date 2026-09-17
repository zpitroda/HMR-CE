"""
Context-Conditioned Token Entropy & Perplexity Scorer
Evaluates autoregressive sequence perplexity:
L(x_t | C) = - 1/N * sum(log P(w_i | w_<i, C))
P_entropy(x_t) = exp(L(x_t | C))
"""

import os
import math
from typing import Optional
import numpy as np

# Offline HuggingFace mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class PerplexityScorer:
    def __init__(
        self,
        use_mock: bool = False,
        llama_server_url: Optional[str] = "http://127.0.0.1:8080"
    ):
        self.use_mock = use_mock
        self.llama_server_url = llama_server_url
        self._model = None
        self._tokenizer = None

        if not self.use_mock:
            self._try_init_local_model()

    def _try_init_local_model(self) -> None:
        """Attempt to load local cached Qwen 0.6B model."""
        snapshot_dir = os.path.expanduser(
            r"~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/6130ef31402718485ca4d80a6234f70d9a4cf362"
        )
        if os.path.isdir(snapshot_dir):
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self._tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    snapshot_dir,
                    local_files_only=True,
                    torch_dtype=torch.float32
                )
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                return
            except Exception as e:
                print(f"[PerplexityScorer] Local causal LM load error: {e}")

        # If not loaded, mock or fallback to server
        self.use_mock = True

    def calculate_perplexity(self, text: str, context: str = "") -> float:
        """
        Compute context-conditioned token perplexity P_entropy(text | context).
        Returns a float >= 1.0.
        """
        text = text.strip()
        if not text:
            return 1.0

        if not self.use_mock and self._model is not None and self._tokenizer is not None:
            return self._score_with_hf_model(text, context)
        
        # Fallback to deterministic pseudo-perplexity
        return self._mock_score(text, context)

    def _score_with_hf_model(self, text: str, context: str) -> float:
        """Compute exact cross-entropy loss over text conditioned on context."""
        import torch
        device = next(self._model.parameters()).device

        full_prompt = f"{context}\n{text}" if context.strip() else text
        enc_full = self._tokenizer(full_prompt, return_tensors="pt")
        input_ids = enc_full["input_ids"].to(device)

        if context.strip():
            enc_ctx = self._tokenizer(context + "\n", return_tensors="pt")
            ctx_len = enc_ctx["input_ids"].shape[1]
        else:
            ctx_len = 0

        # Labels: mask context tokens with -100 so loss is computed ONLY on text tokens
        labels = input_ids.clone()
        if ctx_len > 0 and ctx_len < labels.shape[1]:
            labels[:, :ctx_len] = -100

        with torch.no_grad():
            outputs = self._model(input_ids, labels=labels)
            loss = outputs.loss.item()
            if math.isnan(loss) or math.isinf(loss):
                return 1.0
            # Guard against exp overflow
            loss = min(loss, 15.0)
            return float(math.exp(loss))

    def _mock_score(self, text: str, context: str) -> float:
        """
        Deterministic heuristic perplexity for testing:
        - Out-of-vocabulary or novel terms produce higher perplexity.
        - Repeated or expected continuation terms produce lower perplexity.
        """
        words = text.lower().split()
        ctx_words = set(context.lower().split()) if context else set()
        
        # Base perplexity
        base_ppl = 15.0
        
        # Novel words increase perplexity
        novel_count = sum(1 for w in words if w not in ctx_words)
        ratio = novel_count / max(len(words), 1)
        
        # Indicator keywords for unconventional/novel hypotheses
        unconventional_signals = ["fail", "failed", "switch", "replace", "contradict", "instead", "novel", "hack", "unconventional"]
        signal_boost = sum(30.0 for w in words if any(sig in w for sig in unconventional_signals))
        
        # Routine indicators
        routine_signals = ["hello", "hi", "ok", "okay", "thanks", "sure", "yes", "confirm", "proceed", "got it"]
        if any(w in routine_signals for w in words):
            return 8.0 + (len(words) % 5)

        computed = base_ppl + (ratio * 20.0) + signal_boost
        return float(max(computed, 1.0))
