import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from transformers.models.llama.modeling_llama import LlamaMLP


class TopKRouter(nn.Module):
    """
    Top-k Gating Router with auxiliary load-balancing loss.
    Projects hidden states to expert logits and computes normalized routing weights.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 1,
        jitter_noise: float = 0.01,
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.jitter_noise = jitter_noise
        self.aux_loss_coef = aux_loss_coef

        # Gating projection
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Metrics tracking for diagnostics
        self.register_buffer("expert_counts", torch.zeros(num_experts, dtype=torch.long), persistent=False)
        self.last_top_indices: Optional[torch.Tensor] = None
        self.last_routing_weights: Optional[torch.Tensor] = None

    def compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        top_indices: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes GShard / Switch Transformer auxiliary load-balancing loss:
        L_aux = N * \sum_{i=1}^N (f_i * P_i)
        where f_i is fraction of tokens dispatched to expert i,
        and P_i is the mean probability assigned to expert i.
        """
        num_tokens = router_probs.size(0)
        if num_tokens == 0:
            return torch.tensor(0.0, device=router_probs.device)

        # Average probability per expert over all tokens: P_i
        mean_probs = router_probs.mean(dim=0)  # [num_experts]

        # Fraction of times expert i was selected: f_i
        # top_indices is [num_tokens, top_k]
        mask = F.one_hot(top_indices, num_classes=self.num_experts).float()  # [num_tokens, top_k, num_experts]
        # Any rank selection counts
        token_to_expert = (mask.sum(dim=1) > 0).float()  # [num_tokens, num_experts]
        expert_fractions = token_to_expert.mean(dim=0)    # [num_experts]

        aux_loss = self.num_experts * torch.sum(expert_fractions * mean_probs)
        return aux_loss

    def forward(
        self,
        x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_flat: [num_tokens, hidden_size]
        Returns:
            top_weights: [num_tokens, top_k] (normalized)
            top_indices: [num_tokens, top_k] (expert index per slot)
            aux_loss: scalar tensor
        """
        logits = self.gate(x_flat)  # [num_tokens, num_experts]

        # Optional jitter noise during training
        if self.training and self.jitter_noise > 0.0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        # Softmax over all experts
        probs = F.softmax(logits, dim=-1)

        # Select top-k
        top_weights, top_indices = torch.topk(probs, self.top_k, dim=-1)

        # Re-normalize top-k weights so they sum to 1.0 per token
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Auxiliary load-balancing loss
        aux_loss = self.compute_aux_loss(probs, top_indices) if self.training else torch.tensor(0.0, device=x_flat.device)

        # Cache routing decisions for evaluation / debugging
        if not self.training:
            self.last_top_indices = top_indices.detach()
            self.last_routing_weights = top_weights.detach()
            for idx in top_indices.view(-1):
                self.expert_counts[idx] += 1

        return top_weights, top_indices, aux_loss

    def forward_and_dispatch(
        self,
        x: torch.Tensor,
        experts: nn.ModuleList
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatches tokens to experts using scatter-gather (index_add_).
        Args:
            x: [batch_size, seq_len, hidden_size]
            experts: nn.ModuleList of expert MLPs
        Returns:
            out: [batch_size, seq_len, hidden_size]
            aux_loss: scalar
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.size(0)

        top_weights, top_indices, aux_loss = self.forward(x_flat)

        final_output = torch.zeros_like(x_flat)

        # Dispatch tokens to each expert
        for exp_idx in range(self.num_experts):
            # token_pos: indices into x_flat; rank_pos: index into top_k (0..top_k-1)
            token_pos, rank_pos = torch.where(top_indices == exp_idx)
            if token_pos.numel() == 0:
                continue

            # Gather tokens assigned to this expert
            tokens_for_expert = x_flat[token_pos]

            # Compute expert output
            expert_out = experts[exp_idx](tokens_for_expert)

            # Weight by normalized routing coefficient
            weights = top_weights[token_pos, rank_pos].unsqueeze(-1).to(expert_out.dtype)
            weighted_out = expert_out * weights

            # Scatter-add into final output tensor
            final_output.index_add_(0, token_pos, weighted_out)

        out = final_output.view(orig_shape)
        return out, aux_loss


class SharedAndRoutedMoEBlock(nn.Module):
    """
    DeepSeek / Qwen-MoE Style MoE Block:
    1 Shared Expert (always active) + M Routed Experts (Top-K active).
    Preserves foundational DSI retrieval representations while enabling
    routed experts to specialize on query types, indexing, and semantic depths.
    """
    def __init__(
        self,
        config,
        num_routed_experts: int = 3,
        top_k: int = 1,
        jitter_noise: float = 0.01,
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        # 1. Shared Expert (always computed)
        self.shared_expert = LlamaMLP(config)

        # 2. Router & Routed Experts
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=num_routed_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
            aux_loss_coef=aux_loss_coef
        )
        self.routed_experts = nn.ModuleList([LlamaMLP(config) for _ in range(num_routed_experts)])

        # Storage for layer-wise aux loss
        self.current_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Shared expert forward pass
        shared_out = self.shared_expert(x)

        # 2. Routed experts forward pass
        routed_out, aux_loss = self.router.forward_and_dispatch(x, self.routed_experts)

        # Store weighted aux loss
        self.current_aux_loss = aux_loss * self.aux_loss_coef

        return shared_out + routed_out


class ClassicSparseMoEBlock(nn.Module):
    """
    Mixtral-Style Sparse MoE Block:
    All N experts are routed via Top-K gating.
    """
    def __init__(
        self,
        config,
        num_experts: int = 4,
        top_k: int = 2,
        jitter_noise: float = 0.01,
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
            aux_loss_coef=aux_loss_coef
        )
        self.experts = nn.ModuleList([LlamaMLP(config) for _ in range(num_experts)])

        self.current_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed_out, aux_loss = self.router.forward_and_dispatch(x, self.experts)
        self.current_aux_loss = aux_loss * self.aux_loss_coef
        return routed_out


class SubDenseSparseMoEBlock(nn.Module):
    """
    Sub-Dense Sparse MoE Block:
    Each expert is narrower (e.g. intermediate_size = 4096 instead of 8192).
    Top-1 routing ensures active parameters per token are ~1.21B (strictly LESS than the 1.71B dense model).
    """
    def __init__(
        self,
        config,
        num_experts: int = 4,
        top_k: int = 1,
        expert_intermediate_size: int = 4096,
        jitter_noise: float = 0.01,
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        import copy
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_intermediate_size = expert_intermediate_size
        self.aux_loss_coef = aux_loss_coef

        # Narrower expert configuration
        expert_config = copy.deepcopy(config)
        expert_config.intermediate_size = expert_intermediate_size

        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
            aux_loss_coef=aux_loss_coef
        )
        self.experts = nn.ModuleList([LlamaMLP(expert_config) for _ in range(num_experts)])
        self.current_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed_out, aux_loss = self.router.forward_and_dispatch(x, self.experts)
        self.current_aux_loss = aux_loss * self.aux_loss_coef
        return routed_out


def collect_moe_aux_loss(model: nn.Module) -> torch.Tensor:
    """
    Iterates through model modules and accumulates all active MoE auxiliary losses.
    """
    total_aux = None
    count = 0
    for module in model.modules():
        if isinstance(module, (SharedAndRoutedMoEBlock, ClassicSparseMoEBlock, SubDenseSparseMoEBlock)):
            if hasattr(module, "current_aux_loss") and module.current_aux_loss is not None:
                if total_aux is None:
                    total_aux = module.current_aux_loss
                else:
                    total_aux = total_aux + module.current_aux_loss
                count += 1

    if total_aux is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    # Average aux loss across MoE layers
    return total_aux / max(1, count)
