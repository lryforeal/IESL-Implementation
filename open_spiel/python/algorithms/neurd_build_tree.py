# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Neural Replicator Dynamics [Omidshafiei et al, 2019].

A policy gradient-like extension to replicator dynamics and the hedge algorithm
that incorporates function approximation.

# References

Shayegan Omidshafiei, Daniel Hennes, Dustin Morrill, Remi Munos,
  Julien Perolat, Marc Lanctot, Audrunas Gruslys, Jean-Baptiste Lespiau,
  Karl Tuyls. Neural Replicator Dynamics. https://arxiv.org/abs/1906.00190.
  2019.
"""

import numpy as np
import torch
import random

class IndexedState(object):
  def __init__(self, openspiel_state, index):
    self._index = index
    self._child_state_index = []
    self._is_terminal = openspiel_state.is_terminal()
    self._is_chance_node = openspiel_state.is_chance_node()
    self._returns = np.array(openspiel_state.returns())
    if self._is_chance_node:
      self._chance_outcomes = openspiel_state.chance_outcomes()
    elif not self._is_terminal:
      self._current_player = openspiel_state.current_player()
      self._legal_actions = openspiel_state.legal_actions()
      self._info_state_index = -1
      self._info_state_str = openspiel_state.information_state_string()
  def index(self):
    return self._index
  def legal_actions(self):
    return self._legal_actions
  def is_terminal(self):
    return self._is_terminal
  def is_chance_node(self):
    return self._is_chance_node
  def returns(self):
    # note, the original returns() returns a list, but here we return a np.array
    return self._returns
  def child(self, action):
    i = -1
    if self._is_chance_node:
      for j in range(len(self._chance_outcomes)):
        if self._chance_outcomes[j][0] == action:
          i = j
          break
    elif not self._is_terminal:
      i = self._legal_actions.index(action)
    return self._child_state_index[i]
  def chance_outcomes(self):
    if self._is_chance_node:
      return self._chance_outcomes
  def current_player(self):
    if not (self._is_chance_node or self._is_terminal):
      return self._current_player
  def legal_actions(self):
    if not (self._is_chance_node or self._is_terminal):
      return self._legal_actions
  def information_state_index(self):
    if not (self._is_chance_node or self._is_terminal):
      return self._info_state_index

class CounterfactualNeurdSolver(object):
  """All-actions, strong NeuRD on counterfactual regrets.

  No regularization bonus is applied, so the current policy likely will not
  converge. The average policy profile is updated and stored in a full
  game-size table and may converge to an approximate Nash equilibrium in
  two-player, zero-sum games.
  """

  def __init__(self, game, epsilon):
    """Creates a new `CounterfactualNeurdSolver`.

    Args:
      game: An OpenSpiel `Game`.
      models: Current policy models (optimizable array-like -> `torch.Tensor`
      callables) for both players.
    """
    # self._game = game
    # used to control the relation between policy and y
    self._epsilon = epsilon

    self._action_num = game.num_distinct_actions()
    self._num_players = game.num_players()

    # build game tree
    self._game_state_list = []
    self._info_state_to_index = dict()
    self._info_state_list = []
    self._info_state_num = 0

    # _info_states_to_actions is a list
    # _info_states_to_actions contains a list mapping from info_state_index to action
    self._info_states_to_actions = []

    def build_game_tree(current_state):
      # return the index of current state
      current_state_indexed = IndexedState(current_state, len(self._game_state_list))
      self._game_state_list.append(current_state_indexed)
      if current_state.is_terminal():
        return current_state_indexed.index()
      child_state_index = []
      if current_state.is_chance_node():
        for action, _ in current_state.chance_outcomes():
          child_state = current_state.child(action)
          child_state_index.append(build_game_tree(child_state))
        current_state_indexed._child_state_index = child_state_index
        return current_state_indexed.index()
      info_state_str = current_state.information_state_string()
      if not info_state_str in self._info_state_to_index:
        self._info_state_to_index[info_state_str] = self._info_state_num
        self._info_state_num = self._info_state_num +1
        self._info_state_list.append(info_state_str)
        self._info_states_to_actions.append(current_state.legal_actions())
      current_state_indexed._info_state_index = self._info_state_to_index[info_state_str]
      for action in current_state.legal_actions():
        child_state = current_state.child(action)
        child_state_index.append(build_game_tree(child_state))
      current_state_indexed._child_state_index = child_state_index
      return current_state_indexed.index()
    build_game_tree(game.new_initial_state())

    self._sa_pair_max_index = self._info_state_num * self._action_num

    # initialize ys and pis

    # ys is a list
    # ys is indexed with (info_state, action), and contains the corresponding score
    self._ys = [0 for _ in range(self._sa_pair_max_index)]
    # self._ys = [[random.random() for _ in range(self._sa_pair_max_index)] for _ in range(self._num_players)]

    # pis is also a list
    # pis is indexed with (info_state, action), and contains the corresponding action probability
    self._pis = [0 for _ in range(self._sa_pair_max_index)]

    # calculate pis
    self.calculate_pi()

    # initialize ws
    # ws is a list
    # ws is indexed with (info_state, action), and contains the corresponding w value
    self._ws = [0 for _ in range(self._sa_pair_max_index)]

  def info_state_action_pair_to_num(self, info_state_index, action):
    # use this to substitude 'sequence feature'
    return info_state_index * self._action_num + action

  def calculate_pi(self):
    # calculate policy pi based on self._y
    # result will be stored in self._pi
    for info_state_index in range(self._info_state_num):
      current_y = [self._ys[self.info_state_action_pair_to_num(info_state_index, action)] for action in self._info_states_to_actions[info_state_index] ]
      if len(current_y) == 0:
        continue

      # calculate action probability using softmax(1/epsilon*y)
      probability = [y / self._epsilon for y in current_y]
      probability = [p - max(probability) for p in probability]
      probability = [np.exp(p) for p in probability]
      probability = [p / sum(probability) for p in probability]


      for (index, action) in enumerate(self._info_states_to_actions[info_state_index]):
        self._pis[self.info_state_action_pair_to_num(info_state_index, action)] = probability[index]

  def current_policy(self):
    """Returns the current policy profile.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to `Action`-probability pairs describing each player's policy.
    """

    # we want the policy of the form `dict<info state str, list<Action, probability>>`

    policy = dict()
    for info_state_index in range(self._info_state_num):
      info_state_str = self._info_state_list[info_state_index]
      if len(self._info_states_to_actions[info_state_index]) == 0:
        continue
      policy[info_state_str] = []
      for action in self._info_states_to_actions[info_state_index]:
        policy[info_state_str].append((action, self._pis[self.info_state_action_pair_to_num(info_state_index, action)]))
    return policy

  def evaluate_and_update_policy(self):
    """Performs a single step of policy evaluation and policy improvement.
    """
    # compute w(pi), store in self._ws
    self.calculate_w()

    eta = 1e-6
    for info_state_index in range(self._info_state_num):
      for action in self._info_states_to_actions[info_state_index]:
        # update y[(info_state, action)]
        # y(x, a) = y(x, a) + eta_m * (w(x, a) - y(x, a))
        self._ys[self.info_state_action_pair_to_num(info_state_index, action)] = \
                self._ys[self.info_state_action_pair_to_num(info_state_index, action)] + \
                eta * \
                (\
                self._ws[self.info_state_action_pair_to_num(info_state_index, action)] - \
                self._ys[self.info_state_action_pair_to_num(info_state_index, action)]\
                )

    # update policy pi
    self.calculate_pi()


  def calculate_w(self):
    # calculate w by travelling through the game tree
    # the result will be stored in self._w

    # w(x, a) = sum(Pr(h)A(h, a))/sum(Pr(h))

    # we will use the following two lists to record the numerators and denominators of w(x, a)
    numerators_w = [0 for _ in range(self._sa_pair_max_index)]
    denominators_w = [0 for _ in range(self._sa_pair_max_index)]

    # we will try to calculate w in one pass
    def travel_game_tree(current_state_index, probability):
      # `probability` is the occurence probability of `current_state`
      # returns the numpy array of values of `current_state`

      current_state = self._game_state_list[current_state_index]

      if current_state.is_terminal():
        return np.zeros(self._num_players)

      current_state_value = np.zeros(self._num_players)

      # if current state is chance node, the only thing we need to do is to calculate state value
      if current_state.is_chance_node():
        for action, action_prob in current_state.chance_outcomes():
          # calculate current state's value
          new_state_index = current_state.child(action)
          new_state_value = travel_game_tree(new_state_index, probability * action_prob)
          reward = self._game_state_list[new_state_index].returns()
          current_state_value = current_state_value + action_prob * (reward + new_state_value)
        return current_state_value

      # else, we need to calculate state value, q value, advantage, contribution to w
      player = current_state.current_player()
      info_state_index = current_state.information_state_index()
      current_state_q_value = [] # current_state_q_value[i] is the q value of `player`'s ith action at current state
      for (index, action) in enumerate(current_state.legal_actions()):
        # calculate current state's value and current player's q value
        action_prob = self._pis[self.info_state_action_pair_to_num(info_state_index, action)]
        new_state_index = current_state.child(action)
        new_state_value = travel_game_tree(new_state_index, probability * action_prob)
        reward = self._game_state_list[new_state_index].returns()

        current_state_value = current_state_value + action_prob * (reward + new_state_value)
        current_state_q_value.append(reward[player] + new_state_value[player])
      for (index, action) in enumerate(current_state.legal_actions()):
        # calculate current player's advantage
        advantage = current_state_q_value[index] - current_state_value[player]

        numerators_w[self.info_state_action_pair_to_num(info_state_index, action)] = \
                numerators_w[self.info_state_action_pair_to_num(info_state_index, action)] + probability * advantage

        denominators_w[self.info_state_action_pair_to_num(info_state_index, action)] = \
                denominators_w[self.info_state_action_pair_to_num(info_state_index, action)] + probability

      return current_state_value

    new_initial_state_index = 0
    travel_game_tree(new_initial_state_index, 1.0)

    # calculate w
    for info_state_index in range(self._info_state_num):
      for action in self._info_states_to_actions[info_state_index]:
        if denominators_w[self.info_state_action_pair_to_num(info_state_index, action)] != 0:
          self._ws[self.info_state_action_pair_to_num(info_state_index, action)] = \
            numerators_w[self.info_state_action_pair_to_num(info_state_index, action)] / \
            denominators_w[self.info_state_action_pair_to_num(info_state_index, action)]
        else:
          self._ws[self.info_state_action_pair_to_num(info_state_index, action)] = 0.0
