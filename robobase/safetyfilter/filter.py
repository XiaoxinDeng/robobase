import numpy as np
from typing import Optional

class Filter:
    def __init__(
        self,
        enabled: bool = True,
        filter_all_except_gripper: bool = True,
        max_action_delta: Optional[float] = None,
        debug: bool = True,
        bridge=None,
    ):  
        self.enabled = enabled
        self.filter_all_except_gripper = filter_all_except_gripper
        self.max_action_delta = max_action_delta
        self.debug = debug
        self.bridge = bridge
        self._printed_debug = False

    def filter_action(self, action: np.ndarray, env, observations=None) -> np.ndarray:
        raise NotImplementedError("filter_action method must be implemented by subclasses.")
