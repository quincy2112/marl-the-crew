from gym.spaces import Discrete
import numpy as np


class Environment(object):
    """Abstract Environment interface.

    All concrete implementations of an environment should derive from this
    interface and implement the method stubs.
    """

    def seed(self, seed):
        raise NotImplementedError("Not implemented in Abstract Base class")

    def reset(self, config):
        r"""Reset the environment with a new config.

        Signals environment handlers to reset and restart the environment using
        a config dict.

        Args:
          config: dict, specifying the parameters of the environment to be
            generated.

        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to an action taken by an agent.

        Returns:
          observation: dict, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def close(self):
        """Take one step in the game.

        Raises:
          AssertionError: abnormal close.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


class CrewEnv(Environment):
    def __init__(self, args, seed):
        self._seed = seed
        if args.crew_name == "TheCrew-small-no_rocket":
            config = {
                "colors": 4,
                "ranks": 4,
                "players": args.num_agents,
                "hints": 0,
                "rockets": 0,
                "seed": self._seed,
            }

            self.action_space = []
            self.observation_space = []
            self.share_observation_space = []
            for i in range(self.players):
                self.action_space.append(Discrete(self.num_moves()))
                self.observation_space.append(
                    [self.vectorized_observation_shape()[0] + self.players]
                )
                self.share_observation_space.append(
                    [self.vectorized_share_observation_shape()[0] + self.players]
                )
