from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from robobase.safetyfilter.filter import Filter
from robobase.safetyfilter.h1_state_bridge import extract_h1_state


class OSCBFFilter(Filter):
    def __init__(
        self,
        enabled: bool = True,
        filter_all_except_gripper: bool = True,
        max_action_delta: Optional[float] = None,
        debug: bool = True,
        bridge=None,
    ):
        super().__init__(
            enabled=enabled,
            filter_all_except_gripper=filter_all_except_gripper,
            max_action_delta=max_action_delta,
            debug=debug,
            bridge=bridge
        )

    def filter_action(self, action: np.ndarray, env, observations=None) -> np.ndarray:
        """
        Expected input:
            action: (T, D_motion) OR (T, D_full), depending on caller
            For OSCBF prototype, we assume D_full = 16 with last 2 dims for gripper, and D_motion = 14 for the rest.
            action (T, 14+2)
        """
        h1_state = extract_h1_state(env)

        if self.debug and not self._printed_debug:
            print("[OSCBFFilter] q_full shape:", h1_state.q_full.shape)
            print("[OSCBFFilter] q_ctrl shape:", h1_state.q_ctrl.shape)
            print("[OSCBFFilter] incoming motion action shape:", action.shape)
            self._printed_debug = True
        
        action_safe = action.copy()
        
        # Split action into motion part and gripper part
        motion_chunk = action_safe[:, :-2]   # shape: (T, 14)
        gripper_chunk = action_safe[:, -2:]  # shape: (T, 2)

        # ------------------------------------------------------------------
        # PLACEHOLDER FOR REAL OSCBF
        # ------------------------------------------------------------------
        # Later you will:
        #   1. build nominal control from action_safe[0] or full chunk
        #   2. compute current robot state / Jacobians
        #   3. solve OSCBF safety correction
        #   4. replace action_safe[:, :] or action_safe[0, :] accordingly
        # ------------------------------------------------------------------
        motion_chunk_safe = motion_chunk  # TODO: replace with real OSCBF output
        # Dummy safety correction for testing wiring: just scale down the action
        motion_chunk_safe *= 0.5
        
        if self.max_action_delta is not None:
            action_safe = np.clip(
                action_safe,
                -self.max_action_delta,
                self.max_action_delta,
            )

        # Reassemble full action: filtered motion + original grippers
        action_safe[:, :-2] = motion_chunk_safe
        action_safe[:, -2:] = gripper_chunk

        return action_safe