from __future__ import annotations

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_dim: int, depth: int, dropout: float = 0.05) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


class _FourBranchLANC(nn.Module):
    def __init__(
        self,
        geometry_dim: int,
        frequency_dim: int,
        antenna_dim: int,
        building_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.05,
        model_note: str = "Building-aware LANC",
    ) -> None:
        super().__init__()
        self.model_note = model_note
        # 四个分支分别学习几何传播、频率影响、天线方向性和建筑环境修正，保持简化 LANC 的解耦结构。
        self.geometry_net = build_mlp(geometry_dim, hidden_dim, depth, dropout)
        self.frequency_net = build_mlp(frequency_dim, hidden_dim, depth, dropout)
        self.antenna_net = build_mlp(antenna_dim, hidden_dim, depth, dropout)
        self.building_net = build_mlp(building_dim, hidden_dim, depth, dropout)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        geometry_x: torch.Tensor,
        frequency_x: torch.Tensor,
        antenna_x: torch.Tensor,
        building_x: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.geometry_net(geometry_x)
            + self.frequency_net(frequency_x)
            + self.antenna_net(antenna_x)
            + self.building_net(building_x)
            + self.global_bias
        )


class BuildingAwareLANC(_FourBranchLANC):
    def __init__(
        self,
        geometry_dim: int,
        frequency_dim: int,
        antenna_dim: int,
        building_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            geometry_dim,
            frequency_dim,
            antenna_dim,
            building_dim,
            hidden_dim,
            depth,
            dropout,
            model_note="Building-aware LANC V1",
        )


class BuildingAwareLANCV2(_FourBranchLANC):
    def __init__(
        self,
        geometry_dim: int,
        frequency_dim: int,
        antenna_dim: int,
        building_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            geometry_dim,
            frequency_dim,
            antenna_dim,
            building_dim,
            hidden_dim,
            depth,
            dropout,
            model_note="Building-aware LANC V2",
        )
