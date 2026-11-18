# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 on ROCm opts in to batch-sharded sampling with `compute_logits_local`,
which must be `compute_logits` up to the logits gather: the final norm and the
specialized decode's per-step hook both live in `compute_logits`."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM
from vllm.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration

HIDDEN, VOCAB = 16, 128


class _LogitsProcessor(LogitsProcessor):
    """Records whether each call asked to skip the TP gather."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.skip_gather: list[bool] = []

    def forward(self, lm_head, hidden_states, embedding_bias=None, skip_gather=False):
        self.skip_gather.append(skip_gather)
        return super().forward(lm_head, hidden_states, embedding_bias, skip_gather)


class _SpecializedDecode:
    steps = 0

    def after_step(self) -> None:
        self.steps += 1


def _bare(cls: type) -> Any:
    """The model without its decoder: only the output head is under test."""
    model: Any = object.__new__(cls)
    nn.Module.__init__(model)
    return model


def _text_model() -> Any:
    model = _bare(KimiLinearForCausalLM)
    model.model = nn.Module()
    model.model.norm = RMSNorm(HIDDEN, eps=1e-6)
    model.lm_head = ParallelLMHead(VOCAB, HIDDEN)
    nn.init.normal_(model.lm_head.weight)
    nn.init.normal_(model.model.norm.weight, mean=1.0, std=0.1)
    model.logits_processor = _LogitsProcessor(VOCAB)
    model.specialized_decode = _SpecializedDecode()
    return model


def test_compute_logits_local_is_compute_logits_without_the_gather(
    default_vllm_config, dist_init
) -> None:
    torch.manual_seed(0)
    text = _text_model()
    hidden_states = torch.randn(5, HIDDEN)
    expected = F.linear(text.model.norm(hidden_states, None), text.lm_head.weight)

    # One rank holds the whole vocabulary, so the two differ only in the gather.
    torch.testing.assert_close(text.compute_logits_local(hidden_states), expected)
    torch.testing.assert_close(text.compute_logits(hidden_states), expected)

    assert text.logits_processor.skip_gather == [True, False]
    # The model runner calls one of the two per step: the hook runs on either.
    assert text.specialized_decode.steps == 2


def test_vision_language_model_forwards_compute_logits_local(
    default_vllm_config, dist_init
) -> None:
    torch.manual_seed(0)
    model = _bare(KimiK3ForConditionalGeneration)
    model.language_model = _text_model()
    hidden_states = torch.randn(3, HIDDEN)

    torch.testing.assert_close(
        model.compute_logits_local(hidden_states),
        model.language_model.compute_logits_local(hidden_states),
    )
