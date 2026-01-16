# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Callable
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    # The tokens include the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    multi_modal_data: Optional["MultiModalDataDict"] = None
    mm_processor_kwargs: dict[str, Any] | None = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]


class BeamSearchInstance:
    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key


def is_done_heuristic(
    instance: BeamSearchInstance,
    beam_width: int,
    early_stopping: bool | str,
    length_penalty: float,
    cur_len: int,
    prompt_len: int = 0,
    min_length: int = 0,
    max_length: int | None = None,
    tokenizer_eos_token_id: int | None = None,
    sort_beams_key: Callable[[BeamSearchSequence], float] | None = None,
) -> bool:
    """
    判断是否应停止 beam search。

    实现原理：
    - 确保至少生成 min_length 个 token 才能停止；
    - 根据 early_stopping 模式不同使用不同的判断逻辑；
    - 对于 early_stopping=True，会在“数量满足 + 质量差距明显”后停止；
    - 对于 early_stopping="never"，仅在所有 beam 均结束或达到最大长度时停止；
    - 对于 early_stopping=False，使用分数启发式判断。
    """

    # ---------- 基础检查 ----------
    if not instance.beams:
        # 没有活跃 beam时，如果完成序列够多就可以停止
        return len(instance.completed) >= beam_width

    # ---------- 长度逻辑 ----------
    generated_len = cur_len - prompt_len
    if generated_len < min_length:
        # 生成长度未达阈值，不可停止
        return False

    # ---------- 辅助函数 ----------
    def normalized_score(seq_or_beam) -> float:
        """计算长度惩罚后的归一化分数"""
        seq_len = len(seq_or_beam.tokens)
        if length_penalty == 0.0:
            return seq_or_beam.cum_logprob
        return seq_or_beam.cum_logprob / (seq_len ** length_penalty)

    # ---------- 模式：early_stopping=True ----------
    if early_stopping is True:
        # 若未达到所需完成序列数，则继续
        if len(instance.completed) < beam_width:
            return False

        # 达到最大长度 -> 停止
        if max_length is not None and cur_len >= max_length:
            return True

        # 有活跃beam时，需比较质量差距
        if instance.completed and instance.beams:
            best_completed = max(normalized_score(seq) for seq in instance.completed)
            best_active = max(normalized_score(beam) for beam in instance.beams)
            # 若活跃beam即使最乐观情况下也难超过完成序列，则终止
            if best_active <= best_completed:
                return True
        return False

    # ---------- 模式：early_stopping="never" ----------
    if early_stopping == "never":
        # 达到最大长度 -> 停止
        if max_length is not None and cur_len >= max_length:
            return True

        # 检查是否还有活跃beam未遇到EOS
        active_beams = [
            beam for beam in instance.beams
            if not beam.tokens or beam.tokens[-1] != tokenizer_eos_token_id
        ]
        if active_beams:
            # 仍有活跃beam -> 不停止
            return False

        # 所有beam已结束，比较分数是否还能找到更好结果
        if not instance.completed or tokenizer_eos_token_id is None:
            return False

        best_active = max(normalized_score(beam) for beam in instance.beams)
        worst_completed = min(normalized_score(seq) for seq in instance.completed)
        return best_active <= worst_completed

    # ---------- 模式：early_stopping=False ----------
    # 启发式停止，当活跃beam不太可能超过当前完成序列时终止
    if not instance.completed or tokenizer_eos_token_id is None:
        return False

    worst_completed = min(normalized_score(seq) for seq in instance.completed)
    best_active = max(normalized_score(beam) for beam in instance.beams)
    return worst_completed >= best_active
