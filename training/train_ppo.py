"""RLLib PPO model based

You can visualize experiment results in ~/logs using TensorBoard.
"""
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import inspect
import sys

# Ray imports
import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.schedulers import ASHAScheduler # https://openreview.net/forum?id=S1Y7OOlRZ algo for early stopping
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved


# CybORG imports
from CybORG import CybORG
# from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgentProperly import BlueLoadAgent
from CybORG.Agents.SimpleAgents.MainAgent import MainAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper

from ray.rllib.models.torch.misc import SlimFC

from typing import Any
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


tf1, tf, tfv = try_import_tf()


class CybORGAgent(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {
        'Blue': BlueLoadAgent  # , #B_lineAgent, 'Green': GreenAgent
    }

    """The CybORGAgent env"""
    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents={'Blue': BlueLoadAgent})
        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Red')
        self.steps = 0
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()

    def step(self, action=None):
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=None):
        random.seed(seed)

class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)

        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

def normc_initializer(std: float = 1.0) -> Any:
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))

    return initializer

class TorchModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                 name)
        torch.nn.Module.__init__(self)

        self.model = TorchFC(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    ray.init()

    relative_path = os.path.abspath(os.getcwd())

    log_dir = 'logs/'
    if len(sys.argv[1:]) != 1:
        print('No log directory specified, defaulting to: {}'.format(log_dir))
    else:   
        log_dir = sys.argv[1]
        print('Log directory specified: {}'.format(log_dir))

    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_PPO_tf_Model", TorchModel)


    config = Trainer.merge_trainer_configs(
        DEFAULT_PPO_CONFIG,{
        "env": CybORGAgent,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_PPO_tf_Model",
            "vf_share_layers": False,
        },
        "lr": 0.0005,
        #"momentum": tune.uniform(0, 1),
        "num_workers": 2,  # parallelism
        "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        "clip_param": 0.5,
        "vf_clip_param": 5.0,
    })

    stop = {
        "training_iteration": 400,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,     # Total number of timesteps
        #"episode_reward_mean": -0.1,     # When to stop.. it would be great if we could define this in terms
                                         # of a more complex expression which incorporates the episode reward min too
                                         # There is a lot of variance in the episode reward min
    }

    

    analysis = tune.run(ppo.PPOTrainer, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config,
                        local_dir=log_dir,
                        stop=stop,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        keep_checkpoints_num=3,
                        checkpoint_score_attr="episode_reward_mean")

    checkpoint_pointer = open("checkpoint_pointer.txt", "w")
    last_checkpoint = analysis.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )

    checkpoint_pointer.write(last_checkpoint)
    print("Best model checkpoint written to: {}".format(last_checkpoint))

    # If you want to throw an error
    #if True:
    #    check_learning_achieved(analysis, 0.1)

    checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=log_dir/PPO... to visualise the learning processs during and after training

