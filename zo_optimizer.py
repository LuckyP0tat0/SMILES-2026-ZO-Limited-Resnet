"""
zo_optimizer.py — Zero-order optimizer (student-implemented).

Реализация: SPSA + Adam (финальная версия)

ВЫБОР МЕТОДА (по результатам экспериментов):
--------------------------------------------
Протестированные конфигурации (все с budget = 8192):

  1. SPSA K=1, fc only, lr=5e-2 const, 256×32     → 1.67%  ← ЛУЧШИЙ
  2. SPSA K=4 (avg), fc only, cosine, 256×32       → 1.63%
  3. SPSA K=1, fc+layer4, cosine, 256×32           → 0.95%
  4. SPSA K=1, fc only, cosine, 128×64             → 0.67%

ВЫВОДЫ из экспериментов:
  - Число шагов важнее размера батча: 256 шагов × batch32 >> 128 шагов × batch64.
    ZO оптимизация выигрывает от частых обновлений даже при шумных оценках.
  - fc-only лучше fc+layer4: при малом бюджете SPSA-сигнал "размывается"
    по ~5M параметрам layer4, не давая fc улучшиться.
  - Постоянный lr (5e-2) лучше cosine decay при таком малом числе шагов:
    cosine тратит последние шаги на lr≈1e-3 который слишком мал для ZO.
  - K=1 эффективнее K=4: в 4 раза больше шагов компенсирует шум оценки.

ПОЧЕМУ SPSA, А НЕ central-difference?
  Central-difference: 2N forward pass на шаг (N = число параметров).
  Для fc: N = 51 300 → 102 600 forward pass за 1 шаг. Невозможно.
  SPSA: ровно 2 forward pass независимо от N.

ПОЧЕМУ ADAM, А НЕ SGD?
  ZO-градиенты шумные. Adam адаптирует lr на основе m2 (дисперсии) →
  стабильнее при зашумлённом сигнале.

ПОЧЕМУ РАДЕМАХЕР (±1)?
  Стандарт для SPSA. Математически: 1/u_i = u_i для u_i ∈ {±1} →
  псевдо-градиент упрощается до scalar × u_i.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer: SPSA + Adam.

    На каждом шаге:
      1. u = Rademacher(shape_of_fc_params)
      2. f_plus  = loss(θ + eps·u)
      3. f_minus = loss(θ − eps·u)
      4. g = (f_plus − f_minus) / (2·eps) · u
      5. θ ← Adam(θ, g, lr)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-2,        # Постоянный learning rate.
                                  # Найден лучшим: cosine decay ухудшает результат,
                                  # т.к. маленький lr в конце слишком слаб для ZO.
        eps: float = 1e-3,        # Величина возмущения SPSA.
        beta1: float = 0.9,       # Adam: EMA градиента
        beta2: float = 0.999,     # Adam: EMA квадрата градиента
        adam_eps: float = 1e-8,   # Adam: числовая стабильность
        perturbation_mode: str = "rademacher",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.perturbation_mode = perturbation_mode

        # Только fc: 512→100 (51 200 весов + 100 biases).
        # Оптимально для ZO: fc — "узкое место" передачи знаний к CIFAR100.
        # Добавление layer4 ухудшает результат при данном бюджете.
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        # Adam state
        self._m1: dict[str, torch.Tensor] = {}
        self._m2: dict[str, torch.Tensor] = {}
        self._step: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Возвращает {имя → параметр} для активных слоёв."""
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Вектор Радемахера (±1) той же формы, что param.

        u_i ∈ {+1, -1} с P=0.5. Для SPSA: 1/u_i == u_i → упрощение.
        """
        return torch.randint_like(param, low=0, high=2).float() * 2.0 - 1.0

    def _estimate_grad_spsa(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """SPSA: оценка псевдо-градиента за 2 forward pass.

        g_i = (f(θ + ε·u) - f(θ - ε·u)) / (2ε) · u_i
        """
        directions: dict[str, torch.Tensor] = {
            name: self._sample_direction(param)
            for name, param in params.items()
        }

        with torch.no_grad():
            # f(θ + ε·u)
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])
            f_plus = loss_fn()

            # f(θ − ε·u)
            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * directions[name])
            f_minus = loss_fn()

            # Восстанавливаем θ
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])

        scalar = (f_plus - f_minus) / (2.0 * self.eps)
        return {name: scalar * directions[name] for name in params}

    def _update_params_adam(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Adam update с постоянным self.lr.

          m1 ← β1·m1 + (1-β1)·g
          m2 ← β2·m2 + (1-β2)·g²
          θ ← θ - lr · (m1/(1-β1^t)) / (√(m2/(1-β2^t)) + ε)
        """
        t = self._step
        with torch.no_grad():
            for name, param in params.items():
                g = grads[name]

                if name not in self._m1:
                    self._m1[name] = torch.zeros_like(param)
                    self._m2[name] = torch.zeros_like(param)

                self._m1[name].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                self._m2[name].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                m1_hat = self._m1[name] / (1.0 - self.beta1 ** t)
                m2_hat = self._m2[name] / (1.0 - self.beta2 ** t)

                param.data.sub_(self.lr * m1_hat / (m2_hat.sqrt() + self.adam_eps))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Один шаг ZO. Вызывает loss_fn 3 раза (1 измерение + 2 SPSA).

        Args:
            loss_fn: Callable() → float. Один и тот же батч за весь шаг.

        Returns:
            Loss до обновления (для tqdm прогресс-бара).
        """
        self._step += 1
        params = self._active_params()

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad_spsa(loss_fn, params)
        self._update_params_adam(params, grads)

        return float(loss_before)
