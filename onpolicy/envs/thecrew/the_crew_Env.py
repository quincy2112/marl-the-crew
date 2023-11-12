from gym.spaces import Discrete
import numpy as np
from gymnasium.spaces import Space, Sequence, Discrete
from typing import Deque
from bidict import bidict
import random
from collections import Counter, deque, defaultdict

#TODO: remove pettingzoo rec, factor out agent_selector
from pettingzoo.utils import agent_selector

REWARD_MAP = {
    "task_complete": 1,
    "win": 10,
    "lose": -10,
}

REWARD_MAP_HINT_PENALTY = {
    "task_complete": 1,
    "win": 10,
    "lose": -10,
    "hint_remaining": -2
}

HINT_TYPE_DICT = {
    'lowest': 0,
    'highest': 1,
    'only': 2,
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
        elif args.crew_name == "TheCrew-small-rockets":
            config = {
                "colors": 4,
                "ranks": 4,
                "players": args.num_agents,
                "rockets": 2,
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
        else:
            raise ValueError("Unknown environment name: " + args.crew_name)
        config['hints'] = args.num_hints 
        config['tasks'] = args.num_tasks
        self.config = config
        self.playing_cards, self.task_cards = self.generateAllCards()
        self.possible_agents: list[str] = [f"player_{i}" for i in range(self.config['players'])]
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for _ in range(self.config['players']):
            self.action_space.append(Discrete(self.action_shape()))
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
            self.agent_selector_hint = agent_selector(self.agents)
            self.hands: dict[str, list[tuple[str, int]]] = {}
            self.tasks: dict[str, list[tuple[str, int]]] = {}
            self.suit_counters: dict[str, Counter] = {}
            self.tasks_owner: dict[tuple[str, int], str] = {}
            # self.current_trick: list[tuple[str, tuple[str, int]]] = []
            self.current_trick: dict[str, tuple[str, int]] = {}
            self.discards = []
            self.hint_step = self.config['hints'] > 0 # True if next action to be taken is hint, False otherwise
            self.trick_suit= None
            # TODO: skip the call to step if no hints left. Querying policy when only one action is available is not efficient!
            self.remaining_hints = {agent: self.config['hints'] for agent in self.agents} 
            self.hinted_card: dict[str, tuple[str, tuple[str, int]]] = {} # agent: (hint_type, card) 
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

            #current agent to play next
            self.agent_to_play = self.agent_selector.reset()
            
            # TODO: current turn as feature. For hints

            # current agent to hint next
            self.agent_to_hint = self.agent_selector_hint.reset()


            obs = self.get_observation(self.agent_to_play)
            share_obs = self.get_shared_observation()
            available_actions = self.legal_moves(self.agent_to_play)
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
        vectorized_hand = np.zeros(self.deck_shape())
        vectorized_hand[hand] = 1

        discards = [self.playing_cards_bidict[card] for card in self.discards]
        vectorized_discards = np.zeros(self.deck_shape())
        vectorized_discards[discards] = 1

        vectorized_current_trick = np.zeros(self.config['players'] * self.deck_shape())
        ordered_agents = self.ordered_obs_players(agent)
        for agent in ordered_agents:
            if agent not in self.current_trick:
                continue
            card = self.current_trick[agent]
            vectorized_current_trick[self.agents.index(agent) * self.deck_shape() + self.playing_cards_bidict[card]] = 1
        
        vectorized_tasks = np.zeros(self.config['players'] * self.deck_shape())
        for agent in ordered_agents:
            for task in self.tasks[agent]:
                vectorized_tasks[self.agents.index(agent) * self.deck_shape() + self.task_cards_bidict[task]] = 1
        vectorized_trick_suit = np.zeros(len(self.suits))
        if self.trick_suit is not None:
            vectorized_trick_suit[self.suits.index(self.trick_suit)] = 1
        if self.config['hints'] > 0:
            all_hints = []
            for agent in ordered_agents:
                agent_hints = np.zeros(self.deck_shape()+4) # 3 for type of hint, 1 for hint unused
                if self.remaining_hints[agent] > 0:
                    agent_hints[-1] = 1
                elif agent in self.hinted_card.keys():
                    hint_type, card = self.hinted_card[agent]
                    agent_hints[self.playing_cards_bidict[card]] = 1
                    agent_hints[self.deck_shape() + HINT_TYPE_DICT[hint_type]] = 1
                all_hints.append(agent_hints)
            hints = np.concatenate(all_hints)
            return np.concatenate([vectorized_hand, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit, hints])
        

        return np.concatenate([vectorized_hand, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit])
    
    def get_hints_vector(self, agent):

        agent_hints = np.zeros(self.deck_shape()+3)
        hint_type, card = self.hinted_card[agent]
        agent_hints[self.playing_cards_bidict[card]] = 1
        agent_hints[self.deck_shape() + HINT_TYPE_DICT[hint_type]] = 1

        return agent_hints

    def get_shared_observation(self):
        """
        When using centralized value function, value function takes in entire
        state of the game, not just agent's observation.
        """
        ordered_agents = self.ordered_obs_players(self.agent_to_play)

        hands = []
        for agent in ordered_agents:
            hands += [self.playing_cards_bidict[card] for card in self.hands[agent]]
        vectorized_hands = np.zeros(self.config['players'] * self.deck_shape())
        vectorized_hands[hands] = 1

        discards = [self.playing_cards_bidict[card] for card in self.discards]
        vectorized_discards = np.zeros(self.deck_shape())
        vectorized_discards[discards] = 1

        vectorized_current_trick = np.zeros(self.config['players'] * self.deck_shape())
        for agent in ordered_agents:
            if agent not in self.current_trick:
                continue
            card = self.current_trick[agent]
            vectorized_current_trick[self.agents.index(agent) * self.deck_shape() + self.playing_cards_bidict[card]] = 1
        
        vectorized_tasks = np.zeros(self.config['players'] * self.deck_shape())
        for agent in ordered_agents:
            for task in self.tasks[agent]:
                vectorized_tasks[self.agents.index(agent) * self.deck_shape() + self.task_cards_bidict[task]] = 1

        vectorized_trick_suit = np.zeros(len(self.suits))

        if self.trick_suit is not None:
            vectorized_trick_suit[self.suits.index(self.trick_suit)] = 1


        if self.config['hints'] > 0:
            all_hints = []
            for agent in ordered_agents:
                agent_hints = np.zeros(self.deck_shape()+4) # 3 for type of hint, 1 for hint unused
                if self.remaining_hints[agent] > 0:
                    agent_hints[-1] = 1
                elif agent in self.hinted_card.keys():
                    hint_type, card = self.hinted_card[agent]
                    agent_hints[self.playing_cards_bidict[card]] = 1
                    agent_hints[self.deck_shape() + HINT_TYPE_DICT[hint_type]] = 1
                all_hints.append(agent_hints)
            hints = np.concatenate(all_hints)
            return np.concatenate([vectorized_hands, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit, hints])
        
        return np.concatenate([vectorized_hands, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit])
    

    def step(self, action):
        """
        Action is tuple of "play" or "hint" and index of card to play or hint
        """
        action = action[0]
        reward = 0
        done = False
        assert self.playing_cards_bidict.inverse[action] in self.hands[self.agent_to_play], (self.playing_cards_bidict.inverse[action], self.hands[self.agent_to_play])
        self.hands[self.agent_to_play].remove(self.playing_cards_bidict.inverse[action])
        if self.trick_suit is None:
            self.trick_suit = self.playing_cards_bidict.inverse[action][0]

        self.current_trick[self.agent_to_play]= self.playing_cards_bidict.inverse[action]
        self.suit_counters[self.agent_to_play][self.playing_cards_bidict.inverse[action][0]] -= 1
        
        # check if trick is over
        if self.agent_selector.is_last() and len(self.current_trick.keys()) == len(
            self.agents
        ):
            trick_suit = self.trick_suit
            cur_trick_list = list(self.current_trick.items())
            trick_value = cur_trick_list[0][1][1] # TODO: Could track these two values live, include as observation.
            trick_owner = cur_trick_list[0][0]
            for card_player, (card_suit, card_value) in cur_trick_list[1:]:
                if card_suit == trick_suit and card_value > trick_value:
                    trick_value = card_value
                    trick_owner = card_player
                elif card_suit == "R" and trick_suit != "R":
                    trick_suit = "R"
                    trick_value = card_value
                    trick_owner = card_player

            # check if any task is completed
            for _, card in cur_trick_list:
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
            self.trick_suit = None
            self.current_trick = {}
        self.agent_to_play = self.agent_selector.next()



        obs = self.get_observation(self.agent_to_play)
        share_obs = self.get_shared_observation()
        infos = {'scores': self.config['tasks'] -  len(self.tasks_owner.keys())}
        available_actions = self.legal_moves(self.agent_to_play)
        rewards = [[reward]] * self.config['players']
        return obs, share_obs, rewards, done, infos, available_actions

    def deck_shape(self):
        return self.config["colors"] * self.config["ranks"] + self.config["rockets"]

    def shared_observation_shape(self):
        hands = self.config['players'] * self.deck_shape()
        
        # additional 3 for hint type. Note that history of when hints were played is lost!
        if self.config['hints'] > 0:
            hints = self.config['players'] * (self.deck_shape() + 3) 
        else:
            hints = 0
        discards = self.deck_shape()

        discards = self.deck_shape()

        # TODO: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.deck_shape()
        tasks = self.config['players'] * self.deck_shape()


        trick_suit = len(self.suits)
        return hands + discards + current_trick + tasks + hints + trick_suit

    def observation_shape(self):
        hand = self.deck_shape()
        
        # additional 3 for hint type. Note that history of when hints were played is lost!
        if self.config['hints'] > 0:
            hints = self.config['players'] * (self.deck_shape() + 3) 
        else:
            hints = 0
        discards = self.deck_shape()

        # TODO: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.deck_shape()
        tasks = self.config['players'] * self.deck_shape()

        trick_suit = len(self.suits)

        return hand + discards + current_trick + tasks + hints + trick_suit
    
    def action_shape(self):
        if self.config['hints'] > 0:
            return self.deck_shape() + 1
        else:
            return self.deck_shape()
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


        self.suits = ["B", "P", "G", "Y"][: self.config["colors"]]
        if self.config["rockets"] > 0:
            self.suits.append("R")


        return playing_cards, task_cards
    
    def ordered_obs_players(self, agent):
        """
        Returns list of players in order from current player clockwise.

        Used to ensure observations are consistent with relative player positions,
        not absolute player positions.
        """
        # TODO: should probably store agent as index, not string 
        agent_ind = int(agent.split('_')[1])
        l = self.agents[:]
        return l[-agent_ind:] + l[:-agent_ind]

    

    def close(self):
        pass

    def legal_moves(self, agent):
        """
        Legal Moves of a agent:
        1. if trick_basecard exist, play cards with color same as trick_basecard
        2. if trick_basecard exist, play any card if player don't have cards with color same as the trick_basecard
        3. if trick_basecard not exist(first player in turn), play any cards

        Hints: Can hint if hint is available and card is highest, lowest, or only card of that suit in hand. No rockets
        """
        mask = np.zeros(self.action_shape(), dtype=np.int8)
        if not self.hint_step:
            # condition 3
            if self.agent_selector.is_first() and agent == self.agent_to_play:
                for card in self.hands[self.agent_to_play]:
                    mask[self.playing_cards_bidict[card]] = 1
                return mask
            # condition 2
            if (
                self.suit_counters[agent][
                    self.trick_suit
                ]
                == 0
            ):
                for card in self.hands[self.agent_to_play]:
                    mask[self.playing_cards_bidict[card]] = 1
                return mask
            # condition 1
            elif self.suit_counters[agent][self.trick_suit] > 0:
                for card in self.hands[agent]:
                    if card[0] == self.trick_suit:
                        mask[self.playing_cards_bidict[card]] = 1
                return mask

        else:
            mask[-1] = 1 # action representing no hint
            if self.remaining_hints[agent] == 0:
                return mask

            # TODO: more efficient method?
            bucketed_cards = defaultdict(list)
            for card in self.hands[agent]:
                if card[0] != "R":
                    bucketed_cards[card[0]].append(card)
            

            for _, cards in bucketed_cards.items():
                if len(cards) == 1:
                    mask[self.playing_cards_bidict[cards[0]]] = 1
                elif len(cards) > 1:
                    mask[self.playing_cards_bidict[max(cards, key=lambda card: card[1])]] = 1
                    mask[self.playing_cards_bidict[min(cards, key=lambda card: card[1])]] = 1

