#
# Copyright (c) 2017 Intel Corporation
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
#


import tensorflow as tf
from tensorflow import keras

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace


class QHead(Head):
    def __init__(self,
                 agent_parameters: AgentParameters,
                 spaces: SpacesDefinition,
                 network_name: str,
                 head_idx: int = 0,
                 loss_weight: float = 1.,
                 is_local: bool = True,
                 activation_function: str = 'relu',
                 **kwargs):
        super().__init__(**kwargs)
        #self.name = 'q_values_head'
        self.spaces = spaces
        if isinstance(self.spaces.action, BoxActionSpace):
            self.num_actions = 1
        elif isinstance(self.spaces.action, DiscreteActionSpace):
            self.num_actions = len(self.spaces.action.actions)
        else:
            raise ValueError(
                'QHead does not support action spaces of type: {class_name}'.format(
                    class_name=self.spaces.action.__class__.__name__,))
        self.return_type = QActionStateValue
        self.q_head = keras.layers.Dense(self.num_actions, activation=keras.activations.get(activation_function))

    def call(self, inputs, **kwargs):
        q_value = self.q_head(inputs)
        return q_value

    def __str__(self):
        result = [
            "Dense (num outputs = {})".format(self.num_actions)
        ]
        return '\n'.join(result)



