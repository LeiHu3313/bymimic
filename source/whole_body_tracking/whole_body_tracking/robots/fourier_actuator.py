# Copyright (c) 2025, Fourier Lab Developers.
# All rights reserved.
#
# SPDX-License-Identifier: LGPL-2.1

"""Fourier actuator model with a piecewise torque-speed envelope."""

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class FourierActuatorTN(DelayedPDActuator):
    """PD actuator using the GR3Mini three-point torque-speed curve.

    Motoring torque follows ``(X1, Y1) -> (X2, Y2) -> (X3, 0)``.
    Braking torque is limited by ``Y3`` over the full speed range.
    """

    cfg: FourierActuatorTNCfg

    def __init__(self, cfg: FourierActuatorTNCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._x1 = self._parse_joint_parameter(cfg.X1, 1.0e9)
        self._x2 = self._parse_joint_parameter(cfg.X2, 1.0e9)
        self._x3 = self._parse_joint_parameter(cfg.X3, 1.0e9)
        self._y1 = self._parse_joint_parameter(cfg.Y1, 1.0e9)
        self._y2 = self._parse_joint_parameter(
            cfg.Y2 if cfg.Y2 is not None else cfg.Y1, cfg.Y1
        )
        self._y3 = self._parse_joint_parameter(
            cfg.Y3 if cfg.Y3 is not None else cfg.Y1, cfg.Y1
        )
        self._static_friction = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._dynamic_friction = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._friction_activation_velocity = self._parse_joint_parameter(cfg.Va, 0.01)
        self._validate_curve()

    def _validate_curve(self):
        if torch.any(self._x1 > self._x2) or torch.any(self._x2 > self._x3):
            raise ValueError("FourierActuatorTNCfg requires X1 <= X2 <= X3.")
        if (
            torch.any(self._y1 <= 0.0)
            or torch.any(self._y2 < 0.0)
            or torch.any(self._y3 <= 0.0)
        ):
            raise ValueError("FourierActuatorTNCfg requires Y1/Y3 > 0 and Y2 >= 0.")

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)
        friction = self._static_friction * torch.tanh(
            joint_vel / self._friction_activation_velocity
        )
        friction += self._dynamic_friction * joint_vel
        self.applied_effort -= friction
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        abs_vel = torch.abs(self._joint_vel)
        first_width = torch.clamp(self._x2 - self._x1, min=1.0e-6)
        second_width = torch.clamp(self._x3 - self._x2, min=1.0e-6)
        first_slope = (self._y2 - self._y1) / first_width
        second_slope = -self._y2 / second_width
        first_segment = self._y1 + first_slope * (abs_vel - self._x1)
        second_segment = self._y2 + second_slope * (abs_vel - self._x2)
        motor_limit = torch.where(
            abs_vel <= self._x1,
            self._y1,
            torch.where(
                abs_vel <= self._x2,
                first_segment,
                torch.where(
                    abs_vel <= self._x3, second_segment, torch.zeros_like(self._y1)
                ),
            ),
        )
        motor_limit = torch.clamp(motor_limit, min=0.0)
        torque_limit = torch.where(
            self._joint_vel * effort > 0.0, motor_limit, self._y3
        )
        torque_limit = torch.minimum(torque_limit, self.effort_limit)
        return torch.clamp(effort, min=-torque_limit, max=torque_limit)


@configclass
class FourierActuatorTNCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`FourierActuatorTN`."""

    class_type: type = FourierActuatorTN

    X1: float = 1.0e9
    X2: float = 1.0e9
    X3: float = 1.0e9
    Y1: float = MISSING
    Y2: float | None = None
    Y3: float | None = None
    Fs: float = 0.0
    Fd: float = 0.0
    Va: float = 0.01
