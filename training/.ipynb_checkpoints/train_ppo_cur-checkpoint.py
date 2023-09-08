"""Alternative RLLib model based on local training

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import inspect
import sys
os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'
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
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import DEFAULT_CONFIG
import ray.rllib.agents.impala as impala
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

# CybORG imports
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
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
    
    # the fixed agent
    agents = {
        'Blue': BlueLoadAgent  # , #MainAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents={'Blue': BlueLoadAgent})

        # self.env = OpenAIGymWrapper('Blue',
        #                            EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(self.cyborg))))
        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Red')
        self.steps = 0
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # print("RED OBS SPACE:", self.observation_space)
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()

    def step(self, action=None):
        # training_agent_name = str(self.agent_name)
        # actions_file = open(training_agent_name+"_training_actions.txt", "w")
        # write_action = str(action)
        # actions_file.write(write_action)
        #actions_file.close()
        
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=2):
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
    ModelCatalog.register_custom_model("CybORG_PPO_tf_Model", CustomModel)


    config = Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,{
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
        "num_workers": 0,  # parallelism
        "framework": "tf2", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        "clip_param": 0.5,
        "vf_clip_param": 5.0,
        "exploration_config": {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 53,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
            },
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }
    })
    

    stop = {
        "training_iteration": 400,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,   # Total number of timesteps
        #"episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    # checkpoint = log_dir + 'PPO_CUR_2022-02-24_MEANDER_3M/PPO_CybORGAgent_3ec94_00000_0_2022-02-24_18-04-44/checkpoint_000750/checkpoint-750'
    # local_dir_resume = log_dir + 'PPO_CUR_2022-02-24_MEANDER_3M/PPO_CybORGAgent_3ec94_00000_0_2022-02-24_18-04-44/'
    #agent = ppo.PPOTrainer(config=config, env=CybORGAgent)
    #agent.restore(checkpoint)
    checkpoint = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-ppo_cur_300itersMainAgent\PPOTrainer_CybORGAgent_a3dad_00000_0_2023-08-29_16-13-42\checkpoint_000300\checkpoint-300"

    analysis = tune.run(ppo.PPOTrainer, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config,
                        local_dir=log_dir,
                        stop=stop,
                        #restore=checkpoint,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        keep_checkpoints_num=3,
                        checkpoint_score_attr="episode_reward_mean")

    checkpoint_pointer = open("checkpoint_pointer.txt", "w")
    last_checkpoint = analysis.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )
    last_checkpoint_str = str(last_checkpoint)

    checkpoint_pointer.write(last_checkpoint_str)
    print("Best model checkpoint written to: {}".format(last_checkpoint))

    # If you want to throw an error
    #if True:
    #    check_learning_achieved(analysis, 0.1)

    checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=log_dir/PPO... to visualise the learning processs during and after training

# python c:\users\shrey\appdata\roaming\python\python38\site-packages\tensorboard\main.py --logdir="C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-08-18_17-49-55"
