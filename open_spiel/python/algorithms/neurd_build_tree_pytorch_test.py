# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
import torch
import torch.nn.functional as F

import pyspiel
import neurd_build_tree as neurd

import time

_GAME = pyspiel.load_game('kuhn_poker')

def time_test():
  def find_all_str(current_state):
    if current_state.is_terminal():
      return
    if current_state.is_chance_node():
      for action, _ in current_state.chance_outcomes():
        new_state = current_state.child(action)
        find_all_str(new_state)
      return
    # current_state.information_state_tensor()
    current_state.information_state_string()
    for action in current_state.legal_actions():
      new_state = current_state.child(action)
      find_all_str(new_state)
  new_state = _GAME.new_initial_state()
  find_all_str(new_state)


class NeurdTest(absltest.TestCase):

  def setUp(self):
    super(NeurdTest, self).setUp()
    torch.manual_seed(42)

  def test_neurd(self):
    num_iterations = int(10000)

    solver = neurd.CounterfactualNeurdSolver(_GAME, epsilon=0.001)

    current_policy = solver.current_policy()
    nash_conv = pyspiel.nash_conv(_GAME, current_policy)
    print("The initial NashConv is " + str(nash_conv))
    self.assertGreater(nash_conv, 0.91)

    for i in range(num_iterations):
      solver.evaluate_and_update_policy()

    current_policy = solver.current_policy()
    nash_conv = pyspiel.nash_conv(_GAME, current_policy)
    print("The final NashConv is " + str(nash_conv))
    # self.assertLess(nash_conv, 0.91)

#    beg_time = time.time()
#    for i in range(1000):
#      solver.calculate_w()
#    end_time = time.time()
#    print(end_time - beg_time)
#    beg_time = time.time()
#    for i in range(1000):
#      solver.calculate_pi()
#    end_time = time.time()
#    print(end_time - beg_time)


if __name__ == '__main__':
#  beg_time = time.time()
#  for _ in range(10000):
#      time_test()
#  end_time = time.time()
#  print(end_time - beg_time)
  # solver = neurd.CounterfactualNeurdSolver(_GAME, epsilon=0.001)
  # solver.evaluate_and_update_policy()
  absltest.main()
