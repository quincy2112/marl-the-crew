from gym.spaces import Discrete
import numpy as np
from gymnasium.spaces import Space, Sequence, Discrete
from typing import Deque
from bidict import bidict
import random
from collections import Counter, deque, defaultdict

#TODO: remove pettingzoo req, factor out agent_selector
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
                "rockets": 4,
                "seed": self._seed,
            }
        else:
            raise ValueError("Unknown environment name: " + args.crew_name)
        self.log = args.log
        config['hints'] = args.num_hints 
        config['tasks'] = args.num_tasks
        config['unified_action_space'] = args.unified_action_space
        config['bidirectional_rep'] = args.use_bidirectional_rep
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
            # self.hint_step = self.config['hints'] > 0 # True if next action to be taken is hint, False otherwise
            self.stage_hint_counter = 0 # number of agents given chance to hint this round. When hits num_agents, next action is playing
            self.trick_suit= None
            # TODO: skip the call to step if no hints left. Querying policy when only one action is available is not efficient!
            self.remaining_hints = {agent: self.config['hints'] for agent in self.agents} 
            self.hinted_cards: dict[str, tuple[str, tuple[str, int]]] = {} # agent: (hint_type, card) 
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
            self.agent_selector.reset()
            
            # TODO: current turn as feature. For hints

            self.agent_selector_hint.reset()

            
            if self.config['bidirectional_rep']:
                self.card_to_bidir_reps_high_to_low = {}
                self.card_to_bidir_reps_low_to_high = {}

                for suit in self.suits:
                    self.card_to_bidir_reps_low_to_high[suit] = np.arange(self.config['ranks'])
                    self.card_to_bidir_reps_high_to_low[suit] = np.arange(self.config['ranks'])[::-1]
                pass    

            # current agent to hint next
            obs = self.get_observation(self.agent_selector.selected_agent)
            share_obs = self.get_shared_observation()
            available_actions = self.get_legal_moves(self.agent_selector.selected_agent)
        else:
            obs = np.zeros(self.observation_shape())
            share_obs = np.zeros(
                self.shared_observation_shape()
            )
            available_actions = np.zeros(self.action_shape())
    
        return obs, share_obs, available_actions
    


    def get_observation(self, agent):
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
                elif agent in self.hinted_cards.keys():
                    hint_type, card = self.hinted_cards[agent]
                    agent_hints[self.playing_cards_bidict[card]] = 1
                    agent_hints[self.deck_shape() + HINT_TYPE_DICT[hint_type]] = 1
                all_hints.append(agent_hints)
            hints = np.concatenate(all_hints)

            # which player, relative to current (hinting) player, is leading the trick # TODO: verify this is done correctly
            # This is done so that the current leader is correctly related to the corresponding hand, tasks, etc in observation.
            current_leader = (int(agent.split("_")[-1])- self.stage_hint_counter) % len(self.agents)
            leader = np.zeros(self.config['players'])
            leader[current_leader] = 1
            return np.concatenate([vectorized_hand, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit, hints, leader])
        

        return np.concatenate([vectorized_hand, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit])
    
    def get_hints_vector(self, agent):

        agent_hints = np.zeros(self.deck_shape()+3)
        hint_type, card = self.hinted_cards[agent]
        agent_hints[self.playing_cards_bidict[card]] = 1
        agent_hints[self.deck_shape() + HINT_TYPE_DICT[hint_type]] = 1

        return agent_hints


    # TODO: try without centralized V
    def get_shared_observation(self):
        """
        When using centralized value function, value function takes in entire
        state of the game, not just agent's observation.
        """
        # TODO: ordering tricks in here SHOULDN'T be used. Order should be objective for shared. 
        ordered_agents = self.ordered_obs_players(self.agent_selector.selected_agent)

        hands = []
        for agent in ordered_agents:
            hands += [self.playing_cards_bidict[card] + self.deck_shape() * int(agent.split('_')[-1]) for card in self.hands[agent]]
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
                elif agent in self.hinted_cards.keys():
                    hint_type, card = self.hinted_cards[agent]
                    agent_hints[self.playing_cards_bidict[card]] = 1
                    agent_hints[self.deck_shape() + HINT_TYPE_DICT[hint_type]] = 1
                all_hints.append(agent_hints)
            hints = np.concatenate(all_hints)
            # NOTE: shared obs is based on agent to play, NOT agent to hint. Therefore, current leader not tracked in shared obs.
            return np.concatenate([vectorized_hands, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit, hints])
        
        return np.concatenate([vectorized_hands, vectorized_discards, vectorized_current_trick, vectorized_tasks, vectorized_trick_suit])
    

    def step(self, action):
        """
        Action is a list of length 1, containing the index of the action. With hints, the index could correspond to a hinting action or a playing action.
        """
        if self.log:
            self.render()
        action = action[0]
        reward = 0
        done = False

        # hint action
        if self.config['hints'] > 0 and self.stage_hint_counter < len(self.agents): # TODO TODO TODO: agent to hint not bumping or not being used!
            assert action >= self.deck_shape(), (action, self.deck_shape())
            player = self.agent_selector_hint.selected_agent
            card = action - self.deck_shape() 
            # if card == self.deck_shape(): # no hint
            #     print('no hint by ', player)
            # else:
            #     print('hinting ', self.playing_cards_bidict.inverse[card], ' by ', player)
            self.stage_hint_counter += 1
            self.agent_selector_hint.next()
            if card == self.deck_shape(): # no hint
                obs = self.get_observation(self.agent_selector.selected_agent)
            else:
                hint_type = self.identify_hint_type(player, self.playing_cards_bidict.inverse[card])
                self.hinted_cards[player] = (hint_type, self.playing_cards_bidict.inverse[card])
                self.remaining_hints[player] -= 1

        # play action
        else:
            assert action < self.deck_shape(), (action, self.deck_shape())
            assert self.playing_cards_bidict.inverse[action] in self.hands[self.agent_selector.selected_agent], (self.playing_cards_bidict.inverse[action], self.hands[self.agent_selector.selected_agent])
            self.hands[self.agent_selector.selected_agent].remove(self.playing_cards_bidict.inverse[action])
            if self.trick_suit is None:
                self.trick_suit = self.playing_cards_bidict.inverse[action][0]
            # print('playing ', self.playing_cards_bidict.inverse[action], ' by ', self.agent_selector.selected_agent)
            self.current_trick[self.agent_selector.selected_agent]= self.playing_cards_bidict.inverse[action]
            self.suit_counters[self.agent_selector.selected_agent][self.playing_cards_bidict.inverse[action][0]] -= 1
            self.agent_selector.next()

            # check if trick is over
            if len(self.current_trick.keys()) == len(self.agents):
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
                        # print('game over')
                        if task_owner == trick_owner:
                            self.tasks[task_owner].remove(card)
                            self.tasks[task_owner].sort()
                            self.tasks_owner.pop(card)

                            # Terminate if all tasks are completed
                            if len(self.tasks_owner.keys()) == 0:
                                reward += REWARD_MAP["win"]
                                done = True
                                break
                        else:
                            # Terminate if task_owner != trick_owner
                            reward += REWARD_MAP["lose"] - REWARD_MAP['task_complete'] * (self.config['tasks'] -  len(self.tasks_owner.keys()))
                            done = True
                            break
                # print('trick won by ', trick_owner)
                self.reinit_agents_order(trick_owner)
                self.reinit_agents_order(trick_owner)
                self.discards += list(self.current_trick.values())
                if self.config['bidirectional_rep']:
                    for suit, rank in self.current_trick.values():
                        rank -=1 # rank is 1 indexed
                        if suit != 'R': # TODO: Instead of maintaining both, maintain one and reverse to create the other.
                            self.card_to_bidir_reps_low_to_high[suit][rank:] -= 1 # low to high
                            self.card_to_bidir_reps_high_to_low[suit][:rank] -= 1 # high to low
                            pass
                self.trick_suit = None
                self.current_trick = {}
                self.stage_hint_counter = 0



        if self.config['hints'] > 0 and self.stage_hint_counter < len(self.agents):
            next_agent = self.agent_selector_hint.selected_agent
        else:
            next_agent = self.agent_selector.selected_agent
        obs = self.get_observation(next_agent)
        share_obs = self.get_shared_observation()
        infos = {'score': self.config['tasks'] -  len(self.tasks_owner.keys())}
        # print(infos)
        available_actions = self.get_legal_moves(next_agent)
        rewards = [[reward]] * self.config['players']
        
        # reset hinting stage. So next actions will be hints. NOTE: which player starts off hinting is just based on previous stage. Arbitrary but should be as good as anything?
        # TODO: implement different hinting timings
        if done and self.log:
            print('GAME OVER', self.config['tasks'] -  len(self.tasks_owner.keys()), '\n\n\n\n\n\n\n\n\n\n\n')
        return obs, share_obs, rewards, done, infos, available_actions

    def deck_shape(self):
        return self.config["colors"] * self.config["ranks"] + self.config["rockets"]

    def shared_observation_shape(self):
        hands = self.config['players'] * self.deck_shape()
        
        # additional 3 for hint type. Note that history of when hints were played is lost!
        if self.config['hints'] > 0:
            hints = self.config['players'] * (self.deck_shape() + 4)
        else:
            hints = 0
        discards = self.deck_shape()

        discards = self.deck_shape()

        # NOTE: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.deck_shape()
        tasks = self.config['players'] * self.deck_shape()


        trick_suit = len(self.suits)


        # no more discards, double all other card reps
        # if self.config['bidirectional_rep']:
        #     return 2 * hands + 2 * current_trick + 2 * tasks + 2 * hints + trick_suit
        
        return hands + discards + current_trick + tasks + hints + trick_suit

    def observation_shape(self):
        hand = self.deck_shape()
        
        # additional 3 for hint type. Note that history of when hints were played is lost!
        if self.config['hints'] > 0:
            hints = self.config['players'] * (self.deck_shape() + 4)
            hints  += self.config['players'] # for trick leader
        else:
            hints = 0
        discards = self.deck_shape()

        # TODO: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.deck_shape()
        tasks = self.config['players'] * self.deck_shape()

        trick_suit = len(self.suits)

        # no more discards, double all other card reps
        # if self.config['bidirectional_rep']:
        #     return 2 * hand + 2 * current_trick + 2 * tasks + 2 * hints + trick_suit
        
        #TODO: feature for commander. 
        return hand + discards + current_trick + tasks + hints + trick_suit
    

    #TODO: Different action space for hinting and not (double the action space)
    def action_shape(self):
        if self.config['hints'] > 0:
            return 2 * self.deck_shape() + 1
        else:
            return self.deck_shape()
        
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)


    def deal_playing_cards(self):
        if self.config["rockets"] == 0:
            to_deal = random.sample(self.playing_cards, len(self.playing_cards))     
        else:
            # deal the highest rocket to first player. Guarantees first player has highest rocket
            to_deal = random.sample(self.playing_cards[:-1], len(self.playing_cards) - 1)
            to_deal.insert(0, self.playing_cards[-1]) 
        for card in to_deal:
            agent = self.agent_selector.next()
            self.hands[agent].append(card)
            self.suit_counters[agent][card[0]] += 1

    def reinit_agents_order(self, start_agent):
        """Reset order given a start agent"""
        idx = self.agents.index(start_agent)
        new_order = self.agents[idx:] + self.agents[0:idx]
        self.agent_selector.reinit(new_order)
        self.agent_selector.next() # NOTE: Agent selector is really stupid. This is necessary. Why????
        self.agent_selector_hint.reinit(new_order)
        self.agent_selector_hint.next()


    # since player 0 is guaranteed to be commander, this ensures that the commander is always dealt the first task
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
    
    def cardlist_to_vector(self, cards):
        """
        Converts list of cards to vector representation. Used for tasks and hands.
        
        Works with both normal and bidir rep.
        """
        if self.config['bidirectional_rep']:
            raise NotImplementedError
        else:
            vector = np.zeros(self.deck_shape())
            for card in cards:
                vector[self.playing_cards_bidict[card]] = 1
        return vector


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

    def get_legal_moves(self, agent):
        """
        """
        if self.config['hints'] == 0:
            return self.get_action_mask(agent)

        if self.config['unified_action_space']:  # TODO: this
            if self.stage_hint_counter < len(self.agents):
                return self.get_hinting_mask(agent)
            else:
                return np.concatenate(self.get_action_mask(agent), np.zeros(1))
            
        mask = np.zeros(1+2*self.deck_shape(), dtype=np.int8)
        if self.stage_hint_counter < len(self.agents):
            mask[self.deck_shape():] = self.get_hinting_mask(agent)
        else:
            mask[:self.deck_shape()] = self.get_action_mask(agent)
        return mask

    def render(self):
        print("Current Trick: ", self.current_trick)
        for agent, hand in self.hands.items():
            print(f"{agent}: {hand}")
        # for agent, tasks in self.tasks.items():
        #     print(f"{agent}: {tasks}")
        print("Current Discards: ", self.discards)
        print("Current Trick Suit: ", self.trick_suit)
        print("Current Tasks Owner: ", self.tasks_owner)
        # print("Current Suit Counters: ", self.suit_counters)
        print("Current Remaining Hints: ", self.remaining_hints)
        print("Current Hinted Cards: ", self.hinted_cards)
        print("Current Agent Selector: ", self.agent_selector.selected_agent)
        print("Current Agent Selector Hint: ", self.agent_selector_hint.selected_agent)
        print("Current Stage Hint Counter: ", self.stage_hint_counter)
        print("\n\n")

    def get_action_mask(self, agent):
        """
        Legal Moves of a agent:
        1. if trick_basecard exist, play cards with color same as trick_basecard
        2. if trick_basecard exist, play any card if player don't have cards with color same as the trick_basecard
        3. if trick_basecard not exist(first player in turn), play any cards
        """
        mask = np.zeros(self.deck_shape(), dtype=np.int8)
        # condition 3
        # if self.agent_selector.is_first():
        if len(self.current_trick.keys()) == 0:
            for card in self.hands[agent]:
                mask[self.playing_cards_bidict[card]] = 1
        # condition 2
        elif (
            self.suit_counters[agent][
                self.trick_suit
            ]
            == 0
        ):
            for card in self.hands[agent]:
                mask[self.playing_cards_bidict[card]] = 1
        # condition 1
        elif self.suit_counters[agent][self.trick_suit] > 0:
            for card in self.hands[agent]:
                if card[0] == self.trick_suit:
                    mask[self.playing_cards_bidict[card]] = 1

        # this assert can break if the game is over. We would be resetting the env before the agent has a chance to play.
        # assert np.sum(mask) > 0, (mask, self.hands[agent], self.trick_suit, self.suit_counters[agent])
        return mask
    def get_hinting_mask(self, agent):
        """
        Hints: Can hint if hint is available and card is highest, lowest, or 
        only card of that suit in hand. No rockets
        """
        mask = np.zeros(self.deck_shape()+1, dtype=np.int8)
        mask[-1] = 1 # action representing no hint
        # print(self.remaining_hints)
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
        return mask

    def identify_hint_type(self, agent, card):
        try:
            suit, _ = card
            same_suit = [card for card in self.hands[agent] if card[0] == suit]
            if len(same_suit) == 1:
                return 'only'
            if card == max(same_suit, key=lambda card: card[1]):
                return 'highest'
            elif card == min(same_suit, key=lambda card: card[1]):
                return 'lowest'
        except:
            pass
        raise ValueError(f"Something went wrong. Card {card} tried to be hinted in hand {self.hands[agent]}")
    

    def get_bidir_bidicts(self):
        """
        Equivalent to playing_cards_bidict, but for bidirectional representations.
        Creates the bidicts as needed. That way, we can use self.card_to_bidir_reps_low_to_high
        and self.card_to_bidir_reps_high_to_low which are easier to track for debugging.

        Returns low_to_high, high_to_low bidicts.
        """
        low_to_high = bidict()
        high_to_low = bidict()
        suit_count = 0
        for suit in self.suits:
            for rank in range(1, self.config['ranks']+1):
                if (suit, rank) in self.discards:
                    continue
                low_to_high[(suit, rank)] = self.card_to_bidir_reps_low_to_high[suit][rank-1] + suit_count * self.config['ranks']
                high_to_low[(suit, rank)] =  self.card_to_bidir_reps_high_to_low[suit][rank-1] + suit_count * self.config['ranks'] + self.config['ranks'] * self.config['colors']
            suit_count += 1
        return low_to_high, high_to_low