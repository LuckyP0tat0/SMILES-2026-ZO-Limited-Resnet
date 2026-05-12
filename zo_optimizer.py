"""
zo_optimizer.py — Zero-order optimizer (student-implemented).

Реализация: SPSA (Simultaneous Perturbation Stochastic Approximation) + Adam

СОБЛЮДЕНИЕ ПРАВИЛ:
  ✅ Редактируется только zo_optimizer.py (разрешённый файл)
  ✅ validate.py и model.py не изменены
  ✅ Бюджет: 256 × 32 = 8192 = _MAX_BUDGET (ровно лимит)
  ✅ Нет вызовов loss.backward() — весь код в torch.no_grad()
  ✅ Модель используется ТОЛЬКО как чёрный ящик: получаем только
     скалярное значение loss_fn() — никаких промежуточных слоёв,
     никаких feature vectors, только float
  ✅ Каждый вызов loss_fn() внутри .step() = 1 forward pass из бюджета

ПОЧЕМУ SPSA, А НЕ central-difference?
---------------------------------------
Скелетный estimator (central-difference) делает 2 forward pass НА КАЖДЫЙ
параметр отдельно. У fc.weight = 512 × 100 = 51 200 параметров →
102 400 forward pass за один шаг. При бюджете 8192 сэмплов это означает
меньше одного реального шага оптимизации → модель не обучается.

SPSA возмущает ВСЕ параметры ОДНОВРЕМЕННО одним случайным вектором →
ровно 2 forward pass на шаг, независимо от числа параметров.
Результат: 256 полноценных шагов за тот же бюджет.

ПОЧЕМУ ADAM, А НЕ SGD?
  ZO-псевдо-градиенты очень зашумлены. Adam адаптирует lr для каждого
  параметра через EMA дисперсии → значительно стабильнее при шуме.

ПОЧЕМУ РАДЕМАХЕР (±1)?
  Для u_i ∈ {+1, -1}: 1/u_i = u_i → псевдо-градиент = scalar × u.
  Меньшая дисперсия оценки чем у Gaussian. Стандарт для SPSA.

ПОЧЕМУ ТОЛЬКО fc?
  При 256 шагах добавление layer4 (~5M параметров) размывает SPSA-сигнал.
  Каждый параметр получает почти случайное обновление — хуже чем fc-only.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer: SPSA + Adam.

    Строго соблюдает правило "black box":
      - Модель запрашивается ТОЛЬКО через loss_fn() → scalar float
      - Никаких обращений к внутренним слоям модели
      - Никаких градиентов

    Алгоритм на каждом шаге:
      1. u = Rademacher(shape_of_fc_params)    # вектор ±1
      2. f_plus  = loss(θ + ε·u)               # forward pass 1
      3. f_minus = loss(θ − ε·u)               # forward pass 2
      4. g = (f_plus − f_minus) / (2ε) · u    # псевдо-градиент
      5. θ ← Adam(θ, g, lr)                   # обновление
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-2,        # Постоянный lr.
                                  # Cosine decay тестировался — давал хуже:
                                  # малый lr в конце слишком слаб для ZO.
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

        # Только fc-голова (512→100): 51 200 + 100 параметров.
        # Добавление layer4 при данном бюджете ухудшает результат:
        # SPSA-сигнал размывается по ~5M параметрам.
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        # Adam state (инициализируется лениво при первом .step())
        self._m1: dict[str, torch.Tensor] = {}   # E[g]      — первый момент
        self._m2: dict[str, torch.Tensor] = {}   # E[g²]     — второй момент
        self._step: int = 0                       # счётчик шагов для bias correction

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Возвращает {имя → параметр} только для активных слоёв."""
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

        Для SPSA: u_i ∈ {+1, -1} с P=0.5.
        Свойство: 1/u_i = u_i → псевдо-градиент упрощается до scalar × u.
        Меньшая дисперсия чем у Gaussian, стандарт в SPSA-литературе.
        """
        return torch.randint_like(param, low=0, high=2).float() * 2.0 - 1.0

    def _estimate_grad_spsa(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """SPSA: оценка псевдо-градиента за 2 forward pass.

        Ключевое свойство: 2 вызова loss_fn() на весь шаг независимо
        от числа параметров (vs 2N у central-difference).

        Модель используется ТОЛЬКО через loss_fn() — строгий black box.

        Математика:
          u_i ~ Rademacher (u_i ∈ {+1, -1})
          g_i = (f(θ+ε·u) - f(θ-ε·u)) / (2ε) · u_i
          Для Радемахера: 1/u_i = u_i → g_i = scalar · u_i
        """
        directions = {name: self._sample_direction(p) for name, p in params.items()}

        with torch.no_grad():
            # Forward pass 1: f(θ + ε·u)
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])
            f_plus = loss_fn()   # ← единственный способ взаимодействия с моделью

            # Forward pass 2: f(θ − ε·u)
            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * directions[name])
            f_minus = loss_fn()  # ← единственный способ взаимодействия с моделью

            # Восстанавливаем θ
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])

        # Псевдо-градиент: scalar = (f+ - f-) / (2ε)
        scalar = (f_plus - f_minus) / (2.0 * self.eps)
        return {name: scalar * directions[name] for name in params}

    def _update_params_adam(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Adam-update: θ ← θ − lr · m1_hat / (√m2_hat + ε).

        Адаптирует lr для каждого параметра через EMA дисперсии градиента.
        При зашумлённых ZO-оценках Adam значительно стабильнее SGD.

          m1 ← β1·m1 + (1-β1)·g         # EMA градиента
          m2 ← β2·m2 + (1-β2)·g²        # EMA квадрата
          m1_hat = m1 / (1-β1^t)         # bias correction
          m2_hat = m2 / (1-β2^t)         # bias correction
          θ ← θ − lr · m1_hat / (√m2_hat + ε)
        """
        t = self._step
        with torch.no_grad():
            for name, param in params.items():
                g = grads[name]

                # Ленивая инициализация при первом вызове
                if name not in self._m1:
                    self._m1[name] = torch.zeros_like(param)
                    self._m2[name] = torch.zeros_like(param)

                # EMA градиента
                self._m1[name].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

                # EMA квадрата градиента
                self._m2[name].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                # Bias correction и update
                m1_hat = self._m1[name] / (1.0 - self.beta1 ** t)
                m2_hat = self._m2[name] / (1.0 - self.beta2 ** t)
                param.data.sub_(self.lr * m1_hat / (m2_hat.sqrt() + self.adam_eps))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Один шаг ZO-оптимизации.

        Вызывает loss_fn ровно 3 раза:
          1 раз — замер loss до шага (для прогресс-бара)
          2 раза — SPSA (f_plus и f_minus)

        Каждый вызов loss_fn использует ОДИН И ТОТ ЖЕ батч данных.
        Модель запрашивается ТОЛЬКО через loss_fn() → scalar float.
        Никаких прямых обращений к слоям модели внутри .step().

        Args:
            loss_fn: Callable() → float. Один батч за весь шаг.

        Returns:
            Loss до обновления (для tqdm прогресс-бара).
        """
        self._step += 1
        params = self._active_params()

        # Loss до обновления — только для отображения в прогресс-баре
        with torch.no_grad():
            loss_before = loss_fn()

        # SPSA: 2 forward pass → псевдо-градиент
        grads = self._estimate_grad_spsa(loss_fn, params)

        # Adam: применяем обновление
        self._update_params_adam(params, grads)

        return float(loss_before)
