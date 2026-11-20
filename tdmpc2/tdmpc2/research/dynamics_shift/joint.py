"""Joint RL execution policy; the original TD-MPC2 agent owns all learning."""

from typing import Any, Mapping


class JointAdaptation:
    def __init__(self, agent: Any, replay: Any, configuration: Mapping[str, Any]):
        self.agent, self.replay, self.configuration = agent, replay, configuration
        self.transitions = self.updates = 0

    def observe(self) -> Mapping[str, Any]:
        """Call only after the caller has committed a real transition/replay."""
        self.transitions += 1
        first = self.configuration['first_update']
        every = self.configuration.get('update_every', 1)
        if not self.replay.num_eps or self.transitions < first or (
                every != 1 and self.transitions % every):
            return {}
        result = self.agent.update(self.replay)
        self.updates += 1
        return result

    def expected_updates(self, budget: int) -> int:
        first = self.configuration['first_update']
        if budget < first:
            return 0
        every = self.configuration.get('update_every', 1)
        return budget // every - (first - 1) // every
