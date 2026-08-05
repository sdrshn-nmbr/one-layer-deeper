"""Typed model and optimizer factory for controlled architecture experiments."""

from dataclasses import dataclass, replace
from functools import partial
import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    BatchReuseContext,
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "transformer"
    d_model: int = 128
    num_heads: int = 4
    mlp_ratio: int = 4
    num_loops: int = 1
    training_loop_cap: int | None = None
    step_layers: int = 1
    state_feedback: str = "none"
    final_hidden_mode: str = "per_example"
    structured_features: bool = False
    gated_update: bool = False
    state_tokens: int = 0
    scratch_tokens: int = 0
    digit_slots: int = 16
    entropy_weight: float = 0.0
    entropy_active_only: bool = True
    entropy_mask_padding: bool = True
    random_detach_prefix: bool = True
    initialization_std: float | None = None
    linear_initialization_scale: float | None = None

    def __post_init__(self) -> None:
        if self.architecture not in {"transformer", "register", "causal_state"}:
            raise ValueError(
                "architecture must be transformer, register, or causal_state"
            )
        if self.d_model < 1 or self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if (
            self.mlp_ratio < 1
            or self.num_loops < 1
            or self.step_layers < 1
            or self.state_tokens < 0
            or self.scratch_tokens < 0
            or self.digit_slots < 1
        ):
            raise ValueError(
                "mlp_ratio, num_loops, and step_layers must be positive; "
                "state_tokens and scratch_tokens cannot be negative; "
                "digit_slots must be positive"
            )
        if self.training_loop_cap is not None and self.training_loop_cap < 1:
            raise ValueError("training_loop_cap must be positive when provided")
        if (
            self.training_loop_cap is not None
            and self.training_loop_cap > self.num_loops
        ):
            raise ValueError("training_loop_cap cannot exceed num_loops")
        if self.state_feedback not in {"none", "continuous", "straight_through"}:
            raise ValueError(
                "state_feedback must be none, continuous, or straight_through"
            )
        if self.final_hidden_mode not in {"per_example", "batch_max"}:
            raise ValueError("final_hidden_mode must be per_example or batch_max")
        if self.architecture == "register" and self.state_feedback == "none":
            raise ValueError("register architecture requires state feedback")
        if self.architecture == "transformer" and self.state_feedback != "none":
            raise ValueError("transformer architecture cannot use state feedback")
        if self.architecture == "causal_state":
            if self.state_feedback != "none":
                raise ValueError("causal_state uses continuous hidden state directly")
            if self.final_hidden_mode != "per_example":
                raise ValueError("causal_state requires per_example final state")
            if self.state_tokens:
                raise ValueError("causal_state uses scratch_tokens, not state_tokens")
            if not self.structured_features:
                raise ValueError("causal_state requires structured_features")
        if self.entropy_weight < 0:
            raise ValueError("entropy_weight cannot be negative")
        if self.initialization_std is not None and self.initialization_std <= 0:
            raise ValueError("initialization_std must be positive when provided")
        if (
            self.linear_initialization_scale is not None
            and self.linear_initialization_scale <= 0
        ):
            raise ValueError(
                "linear_initialization_scale must be positive when provided"
            )
        if (
            self.initialization_std is not None
            and self.linear_initialization_scale is not None
        ):
            raise ValueError(
                "initialization_std and linear_initialization_scale are mutually exclusive"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    implementation: str = "default"
    max_batch_uses: int = 1
    wall_clock_schedule: bool = False

    def __post_init__(self) -> None:
        if self.implementation not in {"default", "foreach", "fused"}:
            raise ValueError("implementation must be default, foreach, or fused")
        if not 1 <= self.max_batch_uses <= 8:
            raise ValueError("max_batch_uses must be between 1 and 8")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    compile_model: bool = False


@dataclass(frozen=True)
class ForwardTrace:
    logits: Tensor
    prompt_states: tuple[Tensor, ...]
    memory_states: tuple[Tensor, ...]
    register_states: tuple[Tensor, ...] = ()
    residue_states: tuple[Tensor, ...] = ()
    scratch_states: tuple[Tensor, ...] = ()
    active_masks: tuple[Tensor, ...] = ()
    static_memory: Tensor | None = None


EXPERIMENTS = {
    "baseline": ExperimentConfig(model=ModelConfig()),
    "scaled_init": ExperimentConfig(
        model=ModelConfig(initialization_std=0.02)
    ),
    "wide_scaled": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02)
    ),
    "wide_scaled_foreach": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02),
        optimizer=OptimizerConfig(implementation="foreach"),
    ),
    "wide_scaled_fused": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02),
        optimizer=OptimizerConfig(implementation="fused"),
    ),
    "wide_scaled_fused_compile": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02),
        optimizer=OptimizerConfig(implementation="fused"),
        compile_model=True,
    ),
    "wide_scaled_reuse2": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02),
        optimizer=OptimizerConfig(max_batch_uses=2),
    ),
    "wide_scaled_reuse4": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02),
        optimizer=OptimizerConfig(max_batch_uses=4),
    ),
    "wide_scaled_reuse8": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02),
        optimizer=OptimizerConfig(max_batch_uses=8),
    ),
    "wide_scaled_fused_reuse8": ExperimentConfig(
        model=ModelConfig(d_model=512, initialization_std=0.02),
        optimizer=OptimizerConfig(implementation="fused", max_batch_uses=8),
    ),
    "register_continuous": ExperimentConfig(
        model=ModelConfig(
            architecture="register",
            d_model=256,
            num_loops=8,
            step_layers=2,
            state_feedback="continuous",
            entropy_weight=0.01,
            initialization_std=0.02,
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-3,
            wall_clock_schedule=True,
        ),
    ),
    "register_discrete_structured": ExperimentConfig(
        model=ModelConfig(
            architecture="register",
            d_model=256,
            num_loops=8,
            step_layers=2,
            state_feedback="straight_through",
            structured_features=True,
            entropy_weight=0.01,
            initialization_std=0.02,
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-3,
            wall_clock_schedule=True,
        ),
    ),
    "register_frontier_reproduction": ExperimentConfig(
        model=ModelConfig(
            architecture="register",
            d_model=256,
            num_loops=64,
            training_loop_cap=16,
            step_layers=2,
            state_feedback="straight_through",
            final_hidden_mode="batch_max",
            structured_features=True,
            entropy_weight=0.01,
            entropy_active_only=False,
            entropy_mask_padding=False,
            linear_initialization_scale=0.4,
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-3,
            wall_clock_schedule=True,
        ),
    ),
    "causal_state_contract": ExperimentConfig(
        model=ModelConfig(
            architecture="causal_state",
            d_model=64,
            num_loops=4,
            training_loop_cap=4,
            scratch_tokens=2,
            digit_slots=16,
            structured_features=True,
            initialization_std=0.02,
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-3,
            wall_clock_schedule=True,
        ),
    ),
    "wide_scaled_tied_k2": ExperimentConfig(
        model=ModelConfig(
            d_model=512,
            num_loops=2,
            initialization_std=0.02,
        )
    ),
    "tied_k4": ExperimentConfig(model=ModelConfig(num_loops=4)),
    "gated_k4": ExperimentConfig(
        model=ModelConfig(num_loops=4, gated_update=True)
    ),
    "state_k4": ExperimentConfig(
        model=ModelConfig(num_loops=4, gated_update=True, state_tokens=4)
    ),
}
_FRONTIER = EXPERIMENTS["register_frontier_reproduction"]
EXPERIMENTS.update(
    {
        "register_frontier_per_example": replace(
            _FRONTIER,
            model=replace(_FRONTIER.model, final_hidden_mode="per_example"),
        ),
        "register_frontier_init_002": replace(
            _FRONTIER,
            model=replace(
                _FRONTIER.model,
                initialization_std=0.02,
                linear_initialization_scale=None,
            ),
        ),
        "register_frontier_masked_entropy": replace(
            _FRONTIER,
            model=replace(
                _FRONTIER.model,
                entropy_active_only=True,
                entropy_mask_padding=True,
            ),
        ),
        "register_frontier_full_bptt": replace(
            _FRONTIER,
            model=replace(_FRONTIER.model, random_detach_prefix=False),
        ),
    }
)
SELECTED_EXPERIMENT = "baseline"


class RuntimeConfig:
    def __init__(self, spec: ModelSpec, architecture: ModelConfig) -> None:
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len
        self.architecture = architecture


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.width = config.d_model
        self.num_heads = config.num_heads
        hidden_width = config.mlp_ratio * config.d_model
        self.attention_norm = RMSNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.mixer_norm = RMSNorm(config.d_model)
        self.up = nn.Linear(config.d_model, hidden_width)
        self.down = nn.Linear(hidden_width, config.d_model)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, self.width)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class RegisterStep(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            Block(config) for _ in range(config.step_layers)
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return hidden


class Model(nn.Module):
    def __init__(self, spec: ModelSpec, architecture: ModelConfig) -> None:
        super().__init__()
        self.config = RuntimeConfig(spec, architecture)
        self.num_loops = architecture.num_loops
        self.state_token_count = architecture.state_tokens
        self.token_embedding = nn.Embedding(spec.vocab_size, architecture.d_model)
        self.position_embedding = nn.Embedding(
            spec.max_seq_len, architecture.d_model
        )
        self.block = Block(architecture)
        self.state_embedding = (
            nn.Parameter(
                torch.empty(architecture.state_tokens, architecture.d_model)
            )
            if architecture.state_tokens
            else None
        )
        if self.state_embedding is not None:
            nn.init.normal_(self.state_embedding, mean=0.0, std=0.02)
        self.update_gate = (
            nn.Parameter(torch.zeros(architecture.d_model))
            if architecture.gated_update
            else None
        )
        self.final_norm = RMSNorm(architecture.d_model)
        self.head = nn.Linear(
            architecture.d_model, spec.vocab_size, bias=False
        )
        self.head.weight = self.token_embedding.weight
        if architecture.initialization_std is not None:
            self._reset_parameters(architecture.initialization_std)
        elif architecture.linear_initialization_scale is not None:
            self._reset_linear_parameters(
                architecture.linear_initialization_scale
            )

    def _reset_parameters(self, standard_deviation: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=standard_deviation,
                )
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _reset_linear_parameters(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=scale * module.weight.shape[1] ** -0.5,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _attention_mask(
        self,
        attention_mask: Tensor | None,
        *,
        batch: int,
        prompt_length: int,
        device: torch.device,
    ) -> Tensor | None:
        if self.state_token_count == 0:
            return attention_mask
        if attention_mask is None:
            return torch.ones(
                batch,
                prompt_length + self.state_token_count,
                dtype=torch.bool,
                device=device,
            )
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        if attention_mask.shape == (batch, prompt_length):
            state_mask = torch.ones(
                batch,
                self.state_token_count,
                dtype=torch.bool,
                device=device,
            )
            return torch.cat((attention_mask, state_mask), dim=1)
        if attention_mask.shape != (batch, prompt_length, prompt_length):
            raise ValueError("invalid attention_mask shape")
        total_length = prompt_length + self.state_token_count
        extended = torch.zeros(
            batch,
            total_length,
            total_length,
            dtype=torch.bool,
            device=device,
        )
        extended[:, :prompt_length, :prompt_length] = attention_mask
        valid_queries = attention_mask.any(dim=-1)
        valid_keys = attention_mask.any(dim=1)
        extended[:, :prompt_length, prompt_length:] = valid_queries[:, :, None]
        extended[:, prompt_length:, :prompt_length] = valid_keys[:, None, :]
        extended[:, prompt_length:, prompt_length:] = True
        return extended

    def _hidden_states(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        *,
        capture: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        batch, prompt_length = input_ids.shape
        positions = torch.arange(prompt_length, device=input_ids.device)
        prompt = self.token_embedding(input_ids) + self.position_embedding(positions)
        if self.state_embedding is None:
            memory = prompt.new_empty(batch, 0, prompt.shape[-1])
        else:
            memory = self.state_embedding.unsqueeze(0).expand(batch, -1, -1)
        hidden = torch.cat((prompt, memory), dim=1)
        mask = self._attention_mask(
            attention_mask,
            batch=batch,
            prompt_length=prompt_length,
            device=input_ids.device,
        )
        prompt_states = [hidden[:, :prompt_length]] if capture else []
        memory_states = [hidden[:, prompt_length:]] if capture else []
        for _ in range(self.num_loops):
            candidate = self.block(hidden, mask)
            if self.update_gate is None:
                hidden = candidate
            else:
                hidden = hidden + torch.sigmoid(self.update_gate) * (
                    candidate - hidden
                )
            if capture:
                prompt_states.append(hidden[:, :prompt_length])
                memory_states.append(hidden[:, prompt_length:])
        return hidden, tuple(prompt_states), tuple(memory_states)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        hidden, _, _ = self._hidden_states(
            input_ids, attention_mask, capture=False
        )
        prompt = hidden[:, : input_ids.shape[1]]
        return self.head(self.final_norm(prompt)), None

    def forward_with_trace(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> ForwardTrace:
        hidden, prompt_states, memory_states = self._hidden_states(
            input_ids, attention_mask, capture=True
        )
        prompt = hidden[:, : input_ids.shape[1]]
        logits = self.head(self.final_norm(prompt))
        return ForwardTrace(logits, prompt_states, memory_states)


N_MARKER = 2
X_MARKER = 3
T_MARKER = 4
DIGIT_OFFSET = 7
MAX_PLACE = 15


def derived_features(input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    field = (
        (input_ids == N_MARKER).cumsum(dim=-1)
        + (input_ids == X_MARKER).cumsum(dim=-1)
        + (input_ids == T_MARKER).cumsum(dim=-1)
    ).clamp(max=3)
    is_digit = input_ids >= DIGIT_OFFSET
    place = torch.zeros_like(input_ids)
    for field_index in (1, 2, 3):
        field_digits = (field == field_index) & is_digit
        digits_to_right = torch.flip(
            torch.flip(field_digits.long(), dims=(-1,)).cumsum(dim=-1),
            dims=(-1,),
        )
        place = place + torch.where(
            field_digits,
            digits_to_right - 1,
            torch.zeros_like(digits_to_right),
        )
    place = place.clamp(max=MAX_PLACE)

    time_digits = torch.where(
        (field == 3) & is_digit,
        input_ids - DIGIT_OFFSET,
        torch.full_like(input_ids, -1),
    )
    time_steps = torch.zeros(
        input_ids.shape[0],
        dtype=torch.long,
        device=input_ids.device,
    )
    for position in range(input_ids.shape[1]):
        digit = time_digits[:, position]
        present = digit >= 0
        time_steps = torch.where(
            present,
            time_steps * 10 + digit.clamp_min(0),
            time_steps,
        )
    return field, place, time_steps.clamp_min(1)


class CausalStateStep(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.d_model
        self.scratch_tokens = config.scratch_tokens
        self.scratch_input = (
            nn.Linear(2 * width, width) if self.scratch_tokens else None
        )
        self.scratch_cell = (
            nn.GRUCell(width, width) if self.scratch_tokens else None
        )
        self.residue_input = nn.Linear(2 * width, width)
        self.residue_cell = nn.GRUCell(width, width)

    def forward(
        self,
        residue: Tensor,
        modulus: Tensor,
        scratch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, slots, width = residue.shape
        if self.scratch_input is None or self.scratch_cell is None:
            scratch_context = residue.new_zeros(batch, width)
            next_scratch = scratch
        else:
            scratch_input = self.scratch_input(
                torch.cat((residue.mean(dim=1), modulus.mean(dim=1)), dim=-1)
            )
            repeated_input = scratch_input[:, None, :].expand(
                batch,
                self.scratch_tokens,
                width,
            )
            next_scratch = self.scratch_cell(
                repeated_input.reshape(-1, width),
                scratch.reshape(-1, width),
            ).view(batch, self.scratch_tokens, width)
            scratch_context = next_scratch.mean(dim=1)

        residue_input = self.residue_input(
            torch.cat(
                (
                    modulus,
                    scratch_context[:, None, :].expand(batch, slots, width),
                ),
                dim=-1,
            )
        )
        next_residue = self.residue_cell(
            residue_input.reshape(-1, width),
            residue.reshape(-1, width),
        ).view(batch, slots, width)
        return next_residue, next_scratch


class CausalStateModel(nn.Module):
    def __init__(self, spec: ModelSpec, architecture: ModelConfig) -> None:
        super().__init__()
        self.config = RuntimeConfig(spec, architecture)
        self.architecture = architecture
        self.token_embedding = nn.Embedding(spec.vocab_size, architecture.d_model)
        self.place_embedding = nn.Embedding(
            architecture.digit_slots,
            architecture.d_model,
        )
        self.scratch_embedding = (
            nn.Parameter(
                torch.empty(architecture.scratch_tokens, architecture.d_model)
            )
            if architecture.scratch_tokens
            else None
        )
        self.step = CausalStateStep(architecture)
        self.final_norm = RMSNorm(architecture.d_model)
        self.head = nn.Linear(
            architecture.d_model,
            spec.vocab_size,
            bias=False,
        )
        self.head.weight = self.token_embedding.weight
        if architecture.initialization_std is not None:
            self._reset_parameters(architecture.initialization_std)
        elif architecture.linear_initialization_scale is not None:
            self._reset_linear_parameters(
                architecture.linear_initialization_scale
            )

    def _reset_parameters(self, standard_deviation: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=standard_deviation,
                )
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.scratch_embedding is not None:
            nn.init.normal_(
                self.scratch_embedding,
                mean=0.0,
                std=standard_deviation,
            )

    def _reset_linear_parameters(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=scale * module.weight.shape[1] ** -0.5,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_slots(
        self,
        input_ids: Tensor,
        field: Tensor,
        place: Tensor,
        attention_mask: Tensor | None,
        field_index: int,
    ) -> Tensor:
        batch = input_ids.shape[0]
        slot_count = self.architecture.digit_slots
        slot_ids = place.clamp(max=slot_count - 1)
        valid = (field == field_index) & (input_ids >= DIGIT_OFFSET)
        if attention_mask is not None:
            valid = valid & attention_mask.to(dtype=torch.bool)
        digit_state = self.token_embedding(input_ids) + self.place_embedding(slot_ids)
        assignments = F.one_hot(slot_ids, slot_count).to(digit_state.dtype)
        assignments = assignments * valid[:, :, None].to(digit_state.dtype)
        return torch.einsum("bls,bld->bsd", assignments, digit_state).view(
            batch,
            slot_count,
            self.architecture.d_model,
        )

    def _initial_states(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        field, place, time_steps = derived_features(input_ids)
        modulus = self._encode_slots(
            input_ids,
            field,
            place,
            attention_mask,
            1,
        )
        residue = self._encode_slots(
            input_ids,
            field,
            place,
            attention_mask,
            2,
        )
        batch = input_ids.shape[0]
        if self.scratch_embedding is None:
            scratch = residue.new_empty(batch, 0, residue.shape[-1])
        else:
            scratch = self.scratch_embedding[None, :, :].expand(batch, -1, -1)
        return modulus, residue, scratch, time_steps

    def _active_rows(
        self,
        step_index: int,
        time_steps: Tensor,
    ) -> Tensor:
        return step_index < time_steps

    def decode_residue(
        self,
        residue: Tensor,
        attention_mask: Tensor | None,
        prompt_length: int,
    ) -> Tensor:
        batch = residue.shape[0]
        if attention_mask is None:
            valid_lengths = torch.full(
                (batch,),
                prompt_length,
                dtype=torch.long,
                device=residue.device,
            )
        else:
            if attention_mask.shape != (batch, prompt_length):
                raise ValueError("causal_state requires a two-dimensional padding mask")
            valid_lengths = attention_mask.long().sum(dim=-1)
        positions = torch.arange(prompt_length, device=residue.device)
        output_places = (
            valid_lengths[:, None] - 1 - positions[None, :]
        ).clamp(min=0, max=self.architecture.digit_slots - 1)
        selected = residue.gather(
            1,
            output_places[:, :, None].expand(-1, -1, residue.shape[-1]),
        )
        return self.head(self.final_norm(selected))

    def _run(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        *,
        capture: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        modulus, residue, scratch, time_steps = self._initial_states(
            input_ids,
            attention_mask,
        )
        execution_loops = (
            self.architecture.training_loop_cap
            if self.training and self.architecture.training_loop_cap is not None
            else self.architecture.num_loops
        )
        time_steps = time_steps.clamp(max=execution_loops)
        residue_states = [residue] if capture else []
        scratch_states = [scratch] if capture else []
        active_masks: list[Tensor] = []
        for step_index in range(execution_loops):
            candidate_residue, candidate_scratch = self.step(
                residue,
                modulus,
                scratch,
            )
            active = self._active_rows(step_index, time_steps)
            residue = torch.where(
                active[:, None, None],
                candidate_residue,
                residue,
            )
            scratch = torch.where(
                active[:, None, None],
                candidate_scratch,
                scratch,
            )
            if capture:
                residue_states.append(residue)
                scratch_states.append(scratch)
                active_masks.append(active)
        logits = self.decode_residue(
            residue,
            attention_mask,
            input_ids.shape[1],
        )
        return (
            logits,
            tuple(residue_states),
            tuple(scratch_states),
            tuple(active_masks),
            modulus,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        logits, _, _, _, _ = self._run(
            input_ids,
            attention_mask,
            capture=False,
        )
        return logits, None

    def forward_with_trace(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> ForwardTrace:
        logits, residue_states, scratch_states, active_masks, modulus = self._run(
            input_ids,
            attention_mask,
            capture=True,
        )
        return ForwardTrace(
            logits=logits,
            prompt_states=(),
            memory_states=(),
            register_states=residue_states,
            residue_states=residue_states,
            scratch_states=scratch_states,
            active_masks=active_masks,
            static_memory=modulus,
        )


class RegisterModel(nn.Module):
    def __init__(self, spec: ModelSpec, architecture: ModelConfig) -> None:
        super().__init__()
        self.config = RuntimeConfig(spec, architecture)
        self.architecture = architecture
        self.token_embedding = nn.Embedding(spec.vocab_size, architecture.d_model)
        self.position_embedding = nn.Embedding(
            spec.max_seq_len,
            architecture.d_model,
        )
        self.field_embedding = (
            nn.Embedding(4, architecture.d_model)
            if architecture.structured_features
            else None
        )
        self.place_embedding = (
            nn.Embedding(MAX_PLACE + 1, architecture.d_model)
            if architecture.structured_features
            else None
        )
        self.step = RegisterStep(architecture)
        self.state_projection = nn.Linear(
            spec.vocab_size,
            architecture.d_model,
            bias=False,
        )
        self.final_norm = RMSNorm(architecture.d_model)
        self.head = nn.Linear(
            architecture.d_model,
            spec.vocab_size,
            bias=False,
        )
        self.head.weight = self.token_embedding.weight
        if architecture.initialization_std is not None:
            self._reset_parameters(architecture.initialization_std)
        elif architecture.linear_initialization_scale is not None:
            self._reset_linear_parameters(
                architecture.linear_initialization_scale
            )

    def _reset_parameters(self, standard_deviation: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=standard_deviation,
                )
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _reset_linear_parameters(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=scale * module.weight.shape[1] ** -0.5,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _feedback_state(self, logits: Tensor) -> Tensor:
        probabilities = logits.softmax(dim=-1)
        if self.architecture.state_feedback == "continuous":
            return probabilities
        hard = F.one_hot(
            probabilities.argmax(dim=-1),
            probabilities.shape[-1],
        ).to(probabilities.dtype)
        return hard + (probabilities - probabilities.detach())

    def _run(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        *,
        capture: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...], Tensor | None]:
        batch, length = input_ids.shape
        field, place, time_steps = derived_features(input_ids)
        loop_cap = (
            self.architecture.training_loop_cap
            if self.training and self.architecture.training_loop_cap is not None
            else self.architecture.num_loops
        )
        time_steps = time_steps.clamp(max=loop_cap)
        positions = torch.arange(length, device=input_ids.device)
        base = self.token_embedding(input_ids) + self.position_embedding(positions)
        if self.field_embedding is not None and self.place_embedding is not None:
            base = (
                base
                + self.field_embedding(field)
                + self.place_embedding(place)
            )

        state = torch.zeros(
            batch,
            length,
            self.config.vocab_size,
            dtype=base.dtype,
            device=base.device,
        )
        selected_hidden = base
        hidden_states = [base] if capture else []
        register_states = [state] if capture else []
        maximum_steps = int(time_steps.max().item())
        detach_prefix = (
            int(torch.randint(0, maximum_steps, ()).item())
            if (
                self.training
                and self.architecture.random_detach_prefix
                and maximum_steps > 1
            )
            else 0
        )
        entropy_terms: list[Tensor] = []
        valid_positions = torch.ones(
            batch,
            length,
            dtype=base.dtype,
            device=base.device,
        )
        if attention_mask is not None and self.architecture.entropy_mask_padding:
            valid_positions = attention_mask.to(
                device=base.device,
                dtype=base.dtype,
            )

        for step_index in range(maximum_steps):
            hidden = self.step(
                base + self.state_projection(state),
                attention_mask,
            )
            logits = self.head(self.final_norm(hidden))
            probabilities = logits.float().softmax(dim=-1)
            active_rows = (step_index < time_steps).to(base.dtype)
            active_state = active_rows[:, None, None]
            next_state = self._feedback_state(logits)
            state = active_state * next_state + (1 - active_state) * state
            if self.architecture.final_hidden_mode == "batch_max":
                selected_hidden = hidden
            else:
                selected_hidden = (
                    active_rows[:, None, None] * hidden
                    + (1 - active_rows[:, None, None]) * selected_hidden
                )

            if self.training and self.architecture.entropy_weight > 0:
                position_weights = valid_positions
                if self.architecture.entropy_active_only:
                    position_weights = active_rows[:, None] * position_weights
                entropy = -(
                    probabilities * (probabilities + 1e-9).log()
                ).sum(dim=-1)
                entropy_terms.append(
                    (entropy * position_weights).sum()
                    / position_weights.sum().clamp_min(1)
                )
            if self.training and step_index < detach_prefix:
                state = state.detach()
            if capture:
                hidden_states.append(selected_hidden)
                register_states.append(state)

        logits = self.head(self.final_norm(selected_hidden))
        auxiliary = (
            self.architecture.entropy_weight * torch.stack(entropy_terms).mean()
            if entropy_terms
            else None
        )
        return logits, tuple(hidden_states), tuple(register_states), auxiliary

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits, _, _, auxiliary = self._run(
            input_ids,
            attention_mask,
            capture=False,
        )
        return logits, auxiliary

    def forward_with_trace(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> ForwardTrace:
        logits, hidden_states, register_states, _ = self._run(
            input_ids,
            attention_mask,
            capture=True,
        )
        return ForwardTrace(
            logits=logits,
            prompt_states=hidden_states,
            memory_states=(),
            register_states=register_states,
        )


def build_model_from_config(spec: ModelSpec, config: ModelConfig) -> nn.Module:
    if config.architecture == "causal_state":
        model = CausalStateModel(spec, config)
    elif config.architecture == "register":
        model = RegisterModel(spec, config)
    else:
        model = Model(spec, config)
    assert_model_state(model, spec)
    return model


class WallClockSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, total_seconds: float) -> None:
        self.optimizer = optimizer
        self.total_seconds = max(1.0, float(total_seconds))
        self.base_learning_rates = [
            group["lr"] for group in optimizer.param_groups
        ]
        self.started_at = time.monotonic()

    def step(self) -> None:
        progress = min(
            max((time.monotonic() - self.started_at) / self.total_seconds, 0.0),
            1.0,
        )
        warmup_fraction = 0.05
        final_fraction = 0.01
        if progress < warmup_fraction:
            factor = final_fraction + (1 - final_fraction) * (
                progress / warmup_fraction
            )
        else:
            tail = (progress - warmup_fraction) / (1 - warmup_fraction)
            factor = final_fraction + (1 - final_fraction) * 0.5 * (
                1 + math.cos(math.pi * tail)
            )
        for group, base_learning_rate in zip(
            self.optimizer.param_groups,
            self.base_learning_rates,
            strict=True,
        ):
            group["lr"] = base_learning_rate * factor


def build_optimizer_from_config(
    model: nn.Module,
    spec: OptimizerSpec,
    config: OptimizerConfig,
) -> OptimizerBundle:
    execution = {}
    if config.implementation == "foreach":
        execution["foreach"] = True
    elif config.implementation == "fused":
        execution["fused"] = True
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            capturable=spec.device_type == "cuda",
            **execution,
        )
    scheduler = (
        WallClockSchedule(optimizer, spec.training_time_seconds)
        if config.wall_clock_schedule
        else None
    )
    return OptimizerBundle(
        optimizer,
        scheduler=scheduler,
        should_reuse_batch=(
            partial(reuse_batch, max_batch_uses=config.max_batch_uses)
            if config.max_batch_uses > 1
            else None
        ),
    )


def reuse_batch(
    context: BatchReuseContext,
    *,
    max_batch_uses: int,
) -> bool:
    return context.current_batch_uses < max_batch_uses


def training_loss(
    loss_logits: Tensor,
    loss_labels: Tensor,
    auxiliary: Tensor | None,
) -> Tensor:
    loss = F.cross_entropy(loss_logits, loss_labels)
    if auxiliary is not None:
        loss = loss + auxiliary.to(device=loss.device, dtype=loss.dtype)
    return loss


def build_model(spec: ModelSpec) -> nn.Module:
    experiment = EXPERIMENTS[SELECTED_EXPERIMENT]
    model = build_model_from_config(spec, experiment.model)
    if experiment.compile_model:
        return torch.compile(
            model,
            fullgraph=True,
            dynamic=False,
        )
    return model


def build_optimizer(
    model: nn.Module, spec: OptimizerSpec
) -> OptimizerBundle:
    experiment = EXPERIMENTS[SELECTED_EXPERIMENT]
    return build_optimizer_from_config(model, spec, experiment.optimizer)


SELECTED_CONFIG = EXPERIMENTS[SELECTED_EXPERIMENT]
SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=(
        512 if SELECTED_CONFIG.model.architecture == "register" else None
    ),
    eval_batch_size=(
        1024 if SELECTED_CONFIG.model.architecture == "register" else None
    ),
)
