from gym.spaces import Discrete
import numpy as np
from gymnasium.spaces import Space, Sequence, Discrete
from typing import Deque
from bidict import bidict
import random
from collections import Counter, deque

#TODO: remove pettingzoo rec, factor out agent_selector
from pettingzoo.utils import agent_selector

REWARD_MAP = {
    "task_complete": 1,
    "win": 10,
    "lose": -10,
}

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
                "rockets": 0,
                "seed": self._seed,
            }
        elif args.crew_name == "TheCrew-standard":
            config = {
                "colors": 4,
                "ranks": 9,
                "players": args.num_agents,
                "rockets": 9,
                "seed": self._seed,
            }
        config['hints'] = args.num_hints
        config['tasks'] = args.num_tasks
        self.config = config
        self.playing_cards, self.task_cards = self.generateAllCards()
        self.possible_agents: list[str] = [f"player_{i}" for i in range(self.config['players'])]
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.config['players']):
            self.action_space.append(Discrete(self.deck_shape()))
            self.observation_space.append(
                [self.observation_shape()]
            )
            self.share_observation_space.append(
                [self.shared_observation_shape()]
            )


    def reset(self, choose=True) -> None:
        if choose:
            self.agents = self.possible_agents[:]
            self.terminations = {agent: False for agent in self.agents}
            self.truncations = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            self._cumulative_rewards = {agent: 0 for agent in self.agents}
            self.agent_selector = agent_selector(self.agents)
            self.hands: dict[str, list[tuple[str, int]]] = {}
            self.tasks: dict[str, list[tuple[str, int]]] = {}
            self.suit_counters: dict[str, Counter] = {}
            self.tasks_owner: dict[tuple[str, int], str] = {}
            self.current_trick: list[tuple[str, tuple[str, int]]] = []
            self.discards = []
            for agent in self.agents:
                self.hands[agent] = []
                self.tasks[agent] = []
                self.suit_counters[agent] = Counter()
            self.deal_playing_cards()
            if self.config["rockets"] == 0:
                self.reinit_agents_order(random.choice(self.agents))
            else:
                for agent in self.agents:
                    if ("R", self.config["rockets"]) in self.hands[agent]:
                        self.reinit_agents_order(agent)
                        break
            self.deal_task_cards()
            for agent in self.agents:
                self.hands[agent].sort()

            #current agent
            self.agent_selection = self.agent_selector.reset()


            obs = self.get_observation(self.agent_selection)
            share_obs = self.get_shared_observation()
            available_actions = self.legal_moves(self.agent_selection)
        else:
            obs = np.zeros(self.observation_shape())
            share_obs = np.zeros(
                self.shared_observation_shape()
            )
            available_actions = np.zeros(self.deck_shape())



        return obs, share_obs, available_actions
    def get_observation(self, agent):
        #TODO: order observations to be consistent. IE, first player index corresponds to self,
        # next to next player in order, etc
        hand = [self.playing_cards_bidict[card] for card in self.hands[agent]]
        vectorized_hands = np.zeros(self.deck_shape())
        vectorized_hands[hand] = 1

        discards = [self.playing_cards_bidict[card] for card in self.discards]
        vectorized_discards = np.zeros(self.deck_shape())
        vectorized_discards[discards] = 1

        vectorized_current_trick = np.zeros(self.config['players'] * self.deck_shape())
        for agent, card in self.current_trick:
            vectorized_current_trick[self.agents.index(agent) * self.deck_shape() + self.playing_cards_bidict[card]] = 1
        
        vectorized_tasks = np.zeros(self.config['players'] * self.deck_shape())
        for agent in self.agents:
            for task in self.tasks[agent]:
                vectorized_tasks[self.agents.index(agent) * self.deck_shape() + self.task_cards_bidict[task]] = 1

        return np.concatenate([vectorized_hands, vectorized_discards, vectorized_current_trick, vectorized_tasks])
    
    def get_shared_observation(self):
        """
        When using centralized value function, value function takes in entire
        state of the game, not just agent's observation.
        """
        hands = []
        for agent in self.agents:
            hands += [self.playing_cards_bidict[card] for card in self.hands[agent]]
        vectorized_hands = np.zeros(self.config['players'] * self.deck_shape())
        vectorized_hands[hands] = 1

        discards = [self.playing_cards_bidict[card] for card in self.discards]
        vectorized_discards = np.zeros(self.deck_shape())
        vectorized_discards[discards] = 1

        vectorized_current_trick = np.zeros(self.config['players'] * self.deck_shape())
        for agent, card in self.current_trick:
            vectorized_current_trick[self.agents.index(agent) * self.deck_shape() + self.playing_cards_bidict[card]] = 1
        
        vectorized_tasks = np.zeros(self.config['players'] * self.deck_shape())
        for agent in self.agents:
            for task in self.tasks[agent]:
                vectorized_tasks[self.agents.index(agent) * self.deck_shape() + self.task_cards_bidict[task]] = 1

        return np.concatenate([vectorized_hands, vectorized_discards, vectorized_current_trick, vectorized_tasks])
    

    def step(self, action):
        """
        Action is the index of card to play
        """
        action = action[0]
        reward = 0
        done = False
        assert self.playing_cards_bidict.inverse[action] in self.hands[self.agent_selection], (self.playing_cards_bidict.inverse[action], self.hands[self.agent_selection])
        self.hands[self.agent_selection].remove(self.playing_cards_bidict.inverse[action])
        self.current_trick.append((self.agent_selection, self.playing_cards_bidict.inverse[action]))
        self.suit_counters[self.agent_selection][self.playing_cards_bidict.inverse[action][0]] -= 1
        
        # check if trick is over
        if self.agent_selector.is_last() and len(self.current_trick) == len(
            self.agents
        ):
            trick_suit, trick_value = self.current_trick[0][1]
            trick_owner = self.current_trick[0][0]
            for card_player, (card_suit, card_value) in self.current_trick[1:]:
                if card_suit == trick_suit and card_value > trick_value:
                    trick_value = card_value
                    trick_owner = card_player
                elif card_suit == "R" and trick_suit != "R":
                    trick_suit = "R"
                    trick_value = card[1]
                    trick_owner = card_player

            # check if any task is completed
            for _, card in self.current_trick:
                if card in self.tasks_owner.keys():
                    task_owner = self.tasks_owner[card]
                    if task_owner == trick_owner:
                        self.tasks[task_owner].remove(card)
                        self.tasks[task_owner].sort()
                        self.tasks_owner.pop(card)
                        # currently only reward trick_owner(task_owner)
                        reward += REWARD_MAP["task_complete"]
                        # Terminate if all tasks are completed
                        if len(self.tasks_owner.keys()) == 0:
                            reward += REWARD_MAP["win"]
                            done = True
                            break
                    else:
                        # currently only punish trick_owner(task_owner)
                        reward += REWARD_MAP["lose"]
                        # Terminate if task_owner != trick_owner
                        done = True
                        break

            self.reinit_agents_order(trick_owner)
            self.current_trick = []
        self.agent_selection = self.agent_selector.next()



        obs = self.get_observation(self.agent_selection)
        share_obs = self.get_shared_observation()
        infos = {'scores': self.config['tasks'] -  len(self.tasks_owner.keys())}
        available_actions = self.legal_moves(self.agent_selection)
        rewards = [[reward]] * self.config['players']
        return obs, share_obs, rewards, done, infos, available_actions

    def deck_shape(self):
        return self.config["colors"] * self.config["ranks"] + self.config["rockets"]

    def shared_observation_shape(self):
        hands = self.config['players'] * self.deck_shape()
        
        # TODO: hints 
        discards = self.deck_shape()

        # TODO: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.deck_shape()
        tasks = self.config['players'] * self.deck_shape()

        return hands + discards + current_trick + tasks

    def observation_shape(self):
        hand = self.deck_shape()
        
        # TODO: hints = 
        discards = self.deck_shape()

        # TODO: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.deck_shape()
        tasks = self.config['players'] * self.deck_shape()

        return hand + discards + current_trick + tasks

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)


    def deal_playing_cards(self):
        to_deal = random.sample(self.playing_cards, len(self.playing_cards))     
        for card in to_deal:
            agent = self.agent_selector.next()
            self.hands[agent].append(card)
            self.suit_counters[agent][card[0]] += 1

    def reinit_agents_order(self, start_agent):
        """Reset order given a start agent"""
        idx = self.agents.index(start_agent)
        new_order = self.agents[idx:] + self.agents[0:idx]
        self.agent_selector.reinit(new_order)

    # Random for now
    def deal_task_cards(self):
        to_deal = random.sample(self.task_cards, self.config["tasks"])     
        for _ in range(len(to_deal)):
            agent = self.agent_selector.next()
            self.tasks[agent].append(task := to_deal.pop())
            self.tasks_owner[task] = agent


    def generateAllCards(self):
        playing_cards: list[tuple[str, int]] = []
        task_cards: Deque[tuple[str, int]] = deque()
        # Color Cards & Task Cards
        for suit in ["B", "P", "G", "Y"][: self.config["colors"]]:
            for rank in range(1, self.config["ranks"] + 1):
                playing_cards.append((suit, rank))
                task_cards.append((suit, rank))
        # Rocket Cards
        for value in range(1, self.config["rockets"] + 1):
            playing_cards.append(("R", value))
        self.playing_cards_bidict = bidict(
            {k: idx for idx, k in enumerate(playing_cards)}
        )
        self.task_cards_bidict = bidict({k: idx for idx, k in enumerate(task_cards)})
        return playing_cards, task_cards
    

    def close(self):
        pass

    def legal_moves(self, agent):
        """
        Legal Moves of a agent:
        1. if trick_basecard exist, play cards with color same as trick_basecard
        2. if trick_basecard exist, play any card if player don't have cards with color same as the trick_basecard
        3. if trick_basecard not exist(first player in turn), play any cards
        """

        mask = np.zeros(len(self.playing_cards_bidict), dtype=np.int8)
        # condition 3
        if self.agent_selector.is_first() and agent == self.agent_selection:
            for card in self.hands[self.agent_selection]:
                mask[self.playing_cards_bidict[card]] = 1
            return mask
        # condition 2
        if (
            self.suit_counters[agent][
                (trick_basecard := self.current_trick[0][1])[0]
            ]
            == 0
        ):
            for card in self.hands[self.agent_selection]:
                mask[self.playing_cards_bidict[card]] = 1
            return mask
        # condition 1
        elif self.suit_counters[agent][trick_basecard[0]] > 0:
            for card in self.hands[agent]:
                if card[0] == trick_basecard[0]:
                    mask[self.playing_cards_bidict[card]] = 1
            return mask
