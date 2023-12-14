from gym.spaces import Discrete
import numpy as np
from gymnasium.spaces import Space, Sequence, Discrete
from typing import Deque
from bidict import bidict
import random
from collections import Counter, deque, defaultdict

from pettingzoo.utils import agent_selector

# TODO: train with multiple configurations.

# TODO: feature for player known to be out of suit

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

TRICK_WIN_REWARD_MAP = { # task holder should have control over trick
    "task_holder": 0.5,
    "task_not_holder": -0.5,
}


# multiply reward by card value. Task holder wants high cards, others don't
# double these rewards if task suit
# these rewards NOT used if task played
CARD_VALUE_REWARD_MAP = {
    "task_holder": -0.05, 
    "task_not_holder": 0.05,
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
        elif args.crew_name == "TheCrew-small-rigged":
            config = {
                "colors": 4,
                "ranks": 4,
                "players": args.num_agents,
                "rockets": 0,
                "seed": self._seed,
                "rigged": True
            }
        elif args.crew_name == "TheCrew-small-rockets":
            config = {
                "colors": 4,
                "ranks": 4,
                "players": args.num_agents,
                "rockets": 4,
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
        self.rigged = config.get('rigged', False)
        self.perfect_information = args.perfect_info
        config['hints'] = args.num_hints 
        config['tasks'] = args.num_tasks
        config['unified_action_space'] = args.unified_action_space
        config['bidirectional_rep'] = args.use_bidirectional_rep
        config['use_trick_win_rewards'] = args.use_trick_win_rewards
        config['use_card_value_rewards'] = args.use_card_value_rewards
        self.config = config
        self.playing_cards, self.task_cards = self.generateAllCards()
        self.possible_agents: list[str] = [f"player_{i}" for i in range(self.config['players'])]
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.observation_shape = self.get_observation_shape()
        self.shared_observation_shape = self.get_shared_observation_shape()
        for _ in range(self.config['players']):
            self.action_space.append(Discrete(self.action_shape()))
            self.observation_space.append(
                [self.observation_shape]
            )
            self.share_observation_space.append(
                [self.shared_observation_shape]
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

            self.game_history = []

            #current agent to play next
            self.agent_selector.reset()
            self.reset_needed=False

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
            if self.log == 1:
                print('New game started')
                print('Tasks: ', self.tasks)
                print('Hands: ', self.hands)
        else:
            obs = np.zeros(self.observation_shape)
            share_obs = np.zeros(
                self.shared_observation_shape
            )
            available_actions = np.zeros(self.action_shape())

        return obs, share_obs, available_actions
    


    def get_observation(self, agent):
        
        if self.perfect_information:
            return self.get_shared_observation()

        # orders based on current agent, meaning agent_to_hint or agent_to_play
        ordered_agents = self.ordered_obs_players(agent)
        
        vectorized_hand = self.cardlist_to_vector(self.hands[agent])
        cumulative_vector = vectorized_hand

        if self.config['bidirectional_rep']:
            vectorized_discards = np.zeros(self.config['rockets'])
            for suit, rank in self.discards: # TODO: optimize by tracking discarded rockets
                if suit == 'R':
                    vectorized_discards[rank-1] = 1
        else:
            vectorized_discards = self.cardlist_to_vector(self.discards)
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_discards])

        current_trick = []
        for agent in ordered_agents:
            current_trick.append(self.cardlist_to_vector([self.current_trick[agent]] if agent in self.current_trick else []))
        vectorized_current_trick = np.concatenate(current_trick)
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_current_trick])

        tasks = []
        for agent in ordered_agents:
            tasks.append(self.cardlist_to_vector(self.tasks[agent]))
        vectorized_tasks = np.concatenate(tasks)
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_tasks])

        vectorized_trick_suit = np.zeros(len(self.suits))
        if self.trick_suit is not None:
            vectorized_trick_suit[self.suits.index(self.trick_suit)] = 1
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_trick_suit])

        if self.config['hints'] > 0:
            all_hints = []
            for agent in ordered_agents:
                agent_hints = np.zeros(self.card_repr_shape()+4) # 3 for type of hint, 1 for hint unused
                if self.remaining_hints[agent] > 0:
                    agent_hints[-1] = 1 # TODO: reverse this. 0 if used, 1 if not
                elif agent in self.hinted_cards.keys():
                    hint_type, card = self.hinted_cards[agent]
                    agent_hints[:-4] = self.cardlist_to_vector([card])
                    agent_hints[self.card_repr_shape() + HINT_TYPE_DICT[hint_type]] = 1
                all_hints.append(agent_hints)
            hints = np.concatenate(all_hints)
            cumulative_vector = np.concatenate([cumulative_vector, hints])

            # which player, relative to current (hinting) player, is leading the trick 
            # This is done so that the current leader is correctly related to the corresponding hand, tasks, etc in observation.
            leader_vec = np.zeros(self.config['players'])
            leader_vec[ordered_agents.index(self.agent_selector.selected_agent)] = 1
            cumulative_vector = np.concatenate([cumulative_vector, leader_vec])        

        is_commander = np.zeros(self.config['players'])
        is_commander[ordered_agents.index("player_0")] = 1
        cumulative_vector = np.concatenate([cumulative_vector, is_commander])

        return cumulative_vector
    
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
        
        ordered_agents = self.ordered_obs_players(self.agent_selector.selected_agent)


        vectorized_hands = np.concatenate([self.cardlist_to_vector(self.hands[agent]) for agent in ordered_agents])

        cumulative_vector = vectorized_hands

        if self.config['bidirectional_rep']:
            vectorized_discards = np.zeros(self.config['rockets'])
            for suit, rank in self.discards: # TODO: optimize by tracking discarded rockets
                if suit == 'R':
                    vectorized_discards[rank-1] = 1
        else:
            vectorized_discards = self.cardlist_to_vector(self.discards)
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_discards])


        current_trick = []
        for agent in ordered_agents:
            current_trick.append(self.cardlist_to_vector([self.current_trick[agent]] if agent in self.current_trick else []))
        vectorized_current_trick = np.concatenate(current_trick)
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_current_trick])

        tasks = []
        for agent in ordered_agents:
            tasks.append(self.cardlist_to_vector(self.tasks[agent]))

        vectorized_tasks = np.concatenate(tasks)
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_tasks])
        vectorized_trick_suit = np.zeros(len(self.suits))

        if self.trick_suit is not None:
            vectorized_trick_suit[self.suits.index(self.trick_suit)] = 1
        cumulative_vector = np.concatenate([cumulative_vector, vectorized_trick_suit])

        if self.config['hints'] > 0:
            all_hints = []
            for agent in ordered_agents:
                agent_hints = np.zeros(self.card_repr_shape()+4) # 3 for type of hint, 1 for hint unused
                if self.remaining_hints[agent] > 0:
                    agent_hints[-1] = 1
                elif agent in self.hinted_cards.keys():
                    hint_type, card = self.hinted_cards[agent]
                    agent_hints[:-4] = self.cardlist_to_vector([card])
                    agent_hints[self.card_repr_shape() + HINT_TYPE_DICT[hint_type]] = 1
                all_hints.append(agent_hints)
            hints = np.concatenate(all_hints)
            cumulative_vector = np.concatenate([cumulative_vector, hints])

            # which player, relative to current (hinting) player, is leading the trick 
            # This is done so that the current leader is correctly related to the corresponding hand, tasks, etc in observation.
            leader_vec = np.zeros(self.config['players'])
            leader_vec[ordered_agents.index(self.agent_selector.selected_agent)] = 1
            cumulative_vector = np.concatenate([cumulative_vector, leader_vec])              
  
        is_commander = np.zeros(self.config['players'])
        is_commander[ordered_agents.index("player_0")] = 1
        cumulative_vector = np.concatenate([cumulative_vector, is_commander])

        return cumulative_vector
    

    def step(self, action):
        """
        Action is a list of length 1, containing the index of the action. With hints, the index could correspond to a hinting action or a playing action.
        """
        
        if self.log ==2 :
            self.render()
        # if action == -1:  # invalid action
        #     obs = np.zeros(self.observation_shape)
        #     share_obs = np.zeros(
        #         self.shared_observation_shape
        #     )
        #     rewards = np.zeros((len(self.agents), 1))
        #     done = False
        #     infos = {}
        #     available_actions = np.zeros(self.action_shape())
        #     return obs, share_obs, rewards, done, infos, available_actions
        action = action[0]
        done = False
        rewards = [[0]] * self.config['players']

        # hint action
        if self.config['hints'] > 0 and self.stage_hint_counter < len(self.agents):
            assert action >= self.card_repr_shape(), (action, self.card_repr_shape())
            player = self.agent_selector_hint.selected_agent
            card_index = action - self.card_repr_shape() 

            if card_index != self.card_repr_shape(): # if "no hint" not chosen
                card = self.index_to_card(card_index)
                hint_type = self.identify_hint_type(player,card)
                self.hinted_cards[player] = (hint_type, card)
                self.remaining_hints[player] -= 1

            self.stage_hint_counter += 1
            if self.log == 1:
                if card_index == self.card_repr_shape():
                    print('No hint given by ', player)
                else:
                    print('Hint given by ', player, ' to ', card, ' of type ', hint_type)
            self.agent_selector_hint.next()

        # play action
        else:
            assert action < self.card_repr_shape(), (action, self.card_repr_shape())
            player = self.agent_selector.selected_agent
            card = self.index_to_card(action)

            assert card in self.hands[player], (card, self.hands[player])
            self.hands[player].remove(card)
            if self.trick_suit is None:
                self.trick_suit = card[0]
            self.current_trick[player]= card
            self.suit_counters[player][card[0]] -= 1
            if self.log == 1:
                print(player, ' played ', card)
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
                if self.log == 1:
                    print('trick won by ', trick_owner)
                # check if any task is completed
                any_task_completed = False
                for _, card in cur_trick_list:
                    if card in self.tasks_owner.keys():
                        task_owner = self.tasks_owner[card]
                        # print('game over')
                        if task_owner == trick_owner:
                            self.tasks[task_owner].remove(card)
                            self.tasks[task_owner].sort()
                            self.tasks_owner.pop(card)
                            for reward in rewards:
                                reward[0] += REWARD_MAP["task_complete"]
                            any_task_completed = True

                            # Terminate if all tasks are completed
                            if len(self.tasks_owner.keys()) == 0:
                                for reward in rewards:
                                    reward[0] += REWARD_MAP["win"]
                                done = True
                                break
                        else:
                            # Terminate if task_owner != trick_owner
                            for reward in rewards:
                                reward[0] += REWARD_MAP["lose"] - REWARD_MAP['task_complete'] * (self.config['tasks'] -  len(self.tasks_owner.keys()))
                            done = True
                            break
                
                
                if not any_task_completed:
                    if self.config['use_trick_win_rewards']: # trick win rewards shared
                        for reward in rewards:
                            if trick_owner == list(self.tasks_owner.values())[0]:
                                reward[0] += TRICK_WIN_REWARD_MAP['task_holder']
                            else:
                                reward[0] += TRICK_WIN_REWARD_MAP['task_not_holder']
                    if self.config['use_card_value_rewards']:
                        for reward, (card_player, card) in zip(rewards, cur_trick_list):
                            task_suits = [card[0] for card in self.tasks[card_player]]
                            if card[0] in task_suits:
                                multiplier = 2
                            else:
                                multiplier = 1
                            if card_player == list(self.tasks_owner.values())[0]:
                                reward[0] += CARD_VALUE_REWARD_MAP['task_holder'] * card[1] * multiplier
                            else:
                                reward[0] += CARD_VALUE_REWARD_MAP['task_not_holder'] * card[1] * multiplier

                # print('trick won by ', trick_owner)
                self.game_history.append((trick_owner, self.current_trick.copy()))
                self.reinit_agents_order(trick_owner)
                self.reinit_agents_order(trick_owner)
                self.discards += list(self.current_trick.values())
                if self.config['bidirectional_rep']:
                    for suit, rank in self.current_trick.values():
                        rank -=1 # rank is 1 indexed
                        if suit != 'R': # TODO: Instead of maintaining both, maintain one and reverse to create the other? More efficient. Do we care?
                            self.card_to_bidir_reps_low_to_high[suit][rank:] -= 1 # low to high
                            self.card_to_bidir_reps_high_to_low[suit][:rank] -= 1 # high to low
                            pass
                self.trick_suit = None
                self.current_trick = {}
                self.stage_hint_counter = 0
        

        if sum(len(hand) for hand in self.hands.values()) == 0 and not done:
            pass

        if self.config['hints'] > 0 and self.stage_hint_counter < len(self.agents):
            next_agent = self.agent_selector_hint.selected_agent
        else:
            next_agent = self.agent_selector.selected_agent


        # THE BELOW BREAKS WHEN GREATER THAN 1 ROLLOUT. WHY????????????
        # if self.config['hints'] > 0 and self.stage_hint_counter < len(self.agents):
        #     next_agent = self.agent_selector_hint.selected_agent
        #     while self.remaining_hints[next_agent] == 0 and self.stage_hint_counter < len(self.agents):
        #         self.agent_selector_hint.next()
        #         next_agent = self.agent_selector_hint.selected_agent
        #         self.stage_hint_counter += 1
        # if self.config['hints'] == 0 or self.stage_hint_counter == len(self.agents):
        #     next_agent = self.agent_selector.selected_agent


        obs = self.get_observation(next_agent)
        share_obs = self.get_shared_observation()
        infos = {'score': self.config['tasks'] -  len(self.tasks_owner.keys())}
        # print(infos)
        available_actions = self.get_legal_moves(next_agent)
        if sum(available_actions) == 0:
            self.reset_needed=True
        
        # reset hinting stage. So next actions will be hints. NOTE: which player starts off hinting is just based on previous stage. Arbitrary but should be as good as anything?
        # TODO: implement different hinting timings
        if done and self.log > 0:
            print('GAME OVER. Tasks remaining: ', len(self.tasks), '\n\n\n\n\n\n\n\n\n\n\n')
        return obs, share_obs, rewards, done, infos, available_actions

    def deck_shape(self):
        return self.config["colors"] * self.config["ranks"] + self.config["rockets"]

    def card_repr_shape(self):
        """shape of card representations"""
        if self.config['bidirectional_rep']:
            return self.config["colors"] * self.config["ranks"] * 2 + self.config["rockets"] 
        else:
            return self.deck_shape()
        
    def get_shared_observation_shape(self):
        total = 0

        hands = self.config['players'] * self.card_repr_shape()
        total += hands
        # additional 3 for hint type. Note that history of when hints were played is lost!
        if self.config['hints'] > 0:
            hints = self.config['players'] * (self.card_repr_shape() + 4)
            hints  += self.config['players'] # for trick leader
            total += hints
    
        if self.config['bidirectional_rep']:
            # only keep track of rockets as they don't use bidirectional rep
            discards = self.config['rockets']
        else:
            discards = self.card_repr_shape()
        total += discards

        # NOTE: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.card_repr_shape()
        total += current_trick

        tasks = self.config['players'] * self.card_repr_shape()
        total += tasks

        trick_suit = len(self.suits)
        total += trick_suit

        is_commander = self.config['players']
        total += is_commander

        return total

    def get_observation_shape(self):
        if self.perfect_information:
            return self.get_shared_observation_shape()
        total = 0

        hand = self.card_repr_shape()
        total += hand

        if self.config['bidirectional_rep']:
            # only keep track of rockets as they don't use bidirectional rep
            discards = self.config['rockets']
        else:
            discards = self.card_repr_shape()
        total += discards


        


        # TODO: current_trick does NOT include order cards were played. 
        current_trick = self.config['players'] * self.card_repr_shape()
        total += current_trick

        tasks = self.config['players'] * self.card_repr_shape()
        total += tasks

        trick_suit = len(self.suits)
        total += trick_suit
        
        # additional 3 for hint type. Note that history of when hints were played is lost!
        if self.config['hints'] > 0:
            hints = self.config['players'] * (self.card_repr_shape() + 4)
            hints += self.config['players'] # for trick leader
            total += hints


        is_commander = self.config['players']
        total += is_commander

        return total
    
    def action_shape(self):
        if self.config['hints'] > 0:
            return 2 * self.card_repr_shape() + 1
        else:
            return self.card_repr_shape()
        
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
        
        # if rigged, no task cards of rank 4
        while self.rigged and to_deal[0][1] == 4:
            to_deal = random.sample(self.task_cards, self.config["tasks"])

        # give task to whoever has the 4 of task suit
        if self.rigged:
            for agent in self.agents:
                if (to_deal[0][0], 4) in self.hands[agent]:
                    self.tasks[agent].append(task := to_deal.pop(0))
                    self.tasks_owner[task] = agent
                    break
            
            # if agent has 4, trade the task card to another agent
            if task in self.hands[agent]:
                # choose agent other than agent
                other_agent = random.choice([a for a in self.agents if a != agent])
                # choose random card from other_agent's hand
                other_card = random.choice(self.hands[other_agent])
                # remove other_card from other_agent's hand
                self.hands[other_agent].remove(other_card)
                # add other_card to agent's hand
                self.hands[agent].append(other_card)
                # add task to other_agent's hand
                self.hands[other_agent].append(task)
                # remove task from agent's hand
                self.hands[agent].remove(task)
                # update suit counters
                self.suit_counters[agent][task[0]] -= 1
                self.suit_counters[other_agent][task[0]] += 1
                self.suit_counters[agent][other_card[0]] += 1
                self.suit_counters[other_agent][other_card[0]] -= 1

        else:
            agent = self.agent_selector.selected_agent
            for _ in range(len(to_deal)):
                self.tasks[agent].append(task := to_deal.pop())
                self.tasks_owner[task] = agent
                agent = self.agent_selector.next()
            self.reinit_agents_order(self.agents[0])


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
            vector = np.zeros(self.card_repr_shape())
            for card in cards:
                suit, rank = card
                if suit != 'R':
                    local_idx_low = self.card_to_bidir_reps_low_to_high[suit][rank-1]
                    local_idx_high = self.card_to_bidir_reps_high_to_low[suit][rank-1]
                    idx_low = local_idx_low + self.config['ranks'] * self.suits.index(suit)
                    idx_high = local_idx_high + self.config['ranks'] * self.suits.index(suit) + self.config['ranks'] * self.config['colors']
                    vector[idx_low] = 1
                    vector[idx_high] = 1
                else:
                    local_idx = rank-1
                    idx = local_idx + 2 * self.config['ranks'] * self.config['colors']
                    vector[idx] = 1
        else:
            vector = np.zeros(self.deck_shape())
            for card in cards:
                vector[self.playing_cards_bidict[card]] = 1
        return vector.astype(int)


    def ordered_obs_players(self, agent):
        """
        Returns list of players in order from current player clockwise.

        Used to ensure observations are consistent with relative player positions,
        not absolute player positions.
        """
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

        mask = np.zeros(1+2*self.card_repr_shape(), dtype=np.int8)
        if self.stage_hint_counter < len(self.agents):
            mask[self.card_repr_shape():] = self.get_hinting_mask(agent)
        else:
            mask[:self.card_repr_shape()] = self.get_action_mask(agent)
        if sum(mask) == 0:
            pass
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

        # condition 3
        # if self.agent_selector.is_first():
        if len(self.current_trick.keys()) == 0:
            mask = self.cardlist_to_vector(self.hands[agent])
        # condition 2
        elif (
            self.suit_counters[agent][
                self.trick_suit
            ]
            == 0
        ):
            mask = self.cardlist_to_vector(self.hands[agent])
        # condition 1
        elif self.suit_counters[agent][self.trick_suit] > 0:
            mask = self.cardlist_to_vector([card for card in self.hands[agent] if card[0] == self.trick_suit])
        else:
            print(self.hands, self.trick_suit, self.suit_counters)
        if sum(mask) == 0:
            pass
        return mask
    
    def get_hinting_mask(self, agent):
        """
        Hints: Can hint if hint is available and card is highest, lowest, or 
        only card of that suit in hand. No rockets
        """
        mask = np.zeros(self.card_repr_shape()+1, dtype=np.int8)
        mask[-1] = 1 # action representing no hint
        if self.remaining_hints[agent] == 0:
            #TODO: assert false here when hint skipping completed
            return mask

        # TODO: more efficient method?
        bucketed_cards = defaultdict(list)
        for card in self.hands[agent]:
            if card[0] != "R":
                bucketed_cards[card[0]].append(card)
        

        for _, cards in bucketed_cards.items():
            if len(cards) == 1:
                indices = self.cardlist_to_vector([cards[0]])
                mask[:-1] = np.logical_or(mask[:-1], indices).astype(int)
            elif len(cards) > 1:
                indices = self.cardlist_to_vector([max(cards, key=lambda card: card[1])])
                mask[:-1] = np.logical_or(mask[:-1], indices).astype(int)
                indices = self.cardlist_to_vector([min(cards, key=lambda card: card[1])])
                mask[:-1] = np.logical_or(mask[:-1], indices).astype(int)
        if sum(mask) == 0:
            pass
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
            if suit == "R":
                continue
            for rank in range(1, self.config['ranks']+1):
                if (suit, rank) in self.discards:
                    continue
                low_to_high[(suit, rank)] = self.card_to_bidir_reps_low_to_high[suit][rank-1] + suit_count * self.config['ranks']
                high_to_low[(suit, rank)] =  self.card_to_bidir_reps_high_to_low[suit][rank-1] + suit_count * self.config['ranks'] + self.config['ranks'] * self.config['colors']
            suit_count += 1
        return low_to_high, high_to_low

    def index_to_card(self, index):
        if self.config['bidirectional_rep']:
            low_to_high, high_to_low = self.get_bidir_bidicts()
            if index < self.config['ranks'] * self.config['colors']:
                if index < 0:
                    pass
                return low_to_high.inverse[index]
            elif index < 2 * self.config['ranks'] * self.config['colors']:
                return high_to_low.inverse[index]
            else:
                return ('R', int(index - 2 * self.config['ranks'] * self.config['colors'] + 1))
        else:
            if index < 0:
                pass
            return self.playing_cards_bidict.inverse[index]
        