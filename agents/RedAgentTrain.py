import inspect
import gym
import os, sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRestoreAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgentProperly import BlueLoadAgent
from CybORG.Agents.SimpleAgents.MainAgent import MainAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper

from CybORG.Agents.SimpleAgents.CybORGRedAgent import CybORGMultiAgent

import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from typing import Any
tf1, tf, tfv = try_import_tf()

def wrap(env):
    return OpenAIGymWrapper(agent_name='Red', env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))



# for loading in restored PPO model trained using CybORGAgent env created in below class
# make sure these params match those given in the CybORGAgent class
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
cyborg = CybORG(path, 'sim', agents={'Blue': BlueLoadAgent})
env = ChallengeWrapper(env=cyborg, agent_name='Red')

class CybORGAgentMindrake(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    
    # the fixed agent
    agents = {
        'Blue': BlueLoadAgent  # , #B_lineAgent, 'Green': GreenAgent
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
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=117):
        random.seed(seed)
        
class CybORGAgentCUABv2(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    
    # the fixed agent
    agents = {
        'Blue': MainAgent  # , #B_lineAgent, 'Green': GreenAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents={'Blue': MainAgent})

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
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=117):
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

ModelCatalog.register_custom_model("CybORG_tf_Model", CustomModel)

PPO_cur_config = Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,{
        "env": CybORGAgentCUABv2,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_tf_Model",
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

PPO_config = Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,{
        "env": CybORGAgentCUABv2,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_tf_Model",
            "vf_share_layers": False,
        },
        "lr": 0.0005,
        #"momentum": tune.uniform(0, 1),
        "num_workers": 0,  # parallelism
        "framework": "tf2", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
        "clip_param": 0.5,
        "vf_clip_param": 5.0, # originally 5.0
    })

DQN_config = Trainer.merge_trainer_configs(
        DQN_DEFAULT_CONFIG,{
        "env": CybORGAgentCUABv2,
        "env_config": {
            "null": 0,
        },
        "model": {
            "custom_model": "CybORG_tf_Model",
            "vf_share_layers": False,
        },
            "lr": 0.0001, # 0.1 for DQN 
            "gamma": 0.9,
            #"epsilon": 0.99,
            "double_q": True, # use DDQN not just DQN
            "v_min": -10.0,
            "v_max": 10.0,
            "train_batch_size": 32, 
            "framework": "tf2",
            "explore": True,
            "exploration_config": {
               # Exploration sub-class by name or full path to module+class
               # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
               "type": "EpsilonGreedy",
               # Parameters for the Exploration class' constructor:
               "initial_epsilon": 1.0,
               "final_epsilon": 0.01,
               "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.
            },
            # "replay_buffer_config": {
            #     "type": ReplayBuffer,
            #     "capacity": 1000, 
            # }
            
        })



class TestRedAgentTrain(BaseAgent):

    # agent that loads a StableBaselines3 PPO model file
    def train(self, results):
        self.model.learn(total_timesteps=10)

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, model_file: str = None):
        if model_file is not None:
            self.model = PPO.load(model_file)
        else:
            self.model = None

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        if self.model is None:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            cyborg = OpenAIGymWrapper('Red', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
            self.model = PPO('MlpPolicy', cyborg)
        action, _states = self.model.predict(observation)
        return action

class BlueLoadAgent(BaseAgent):
    # agent that loads a StableBaselines3 PPO model file
    def train(self, results):
        print("Training for 10 timesteps")
        self.model.learn(total_timesteps=10)

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, model_file: str = None):
        if model_file is not None:
            self.model = PPO.load(model_file)
        else:
            self.model = None

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        if self.model is None:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            cyborg = OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
            self.model = PPO('MlpPolicy', cyborg)
        action, _states = self.model.predict(observation)
        
        return action
    
    def get_action_2(self, observation, action_space, path, cyborg):
        
        if self.model is None:
            self.model = PPO("MlpPolicy", cyborg)
        action, _states = self.model.predict(observation)
        
        return action
    
    def save_model(self, path):
        self.model.save(path)
        print("Saved model at", path)


        
class RedAgentTrain(BaseAgent):
    # agent that loads a StableBaselines3 PPO model file
    def __init__(self, model_file=True, ray=True): # : str = None
        # ModelCatalog.register_custom_model("CybORG_PPO_Model", CustomModel)
        if ray:
            self.ray=True
        else:
            self.ray=False
        if model_file is True:
            print("Loading model from file")
            if not self.ray:
                print("Using Stable baselines model saved at", model_file)
                self.model = PPO.load(model_file)
            else:
                print("Using Ray model")
                
                RedPPO_cur_Mindrake = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_cur_Mindrake\PPOTrainer_CybORGAgent_d8486_00000_0_2023-08-15_22-08-13\checkpoint_000401\checkpoint-401"
                
                RedPPO_cur_CUABv2 = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_cur_CUABv2\PPOTrainer_CybORGAgent_43272_00000_0_2023-08-18_18-40-03\checkpoint_000385\checkpoint-385"
                
                RedPPO_cur_CUABv2_redone = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-PPO_cur_CUABv2_REDONE\PPOTrainer_CybORGAgent_83fcf_00000_0_2023-08-26_20-48-23\checkpoint_000400\checkpoint-400"
                
                RedPPO_Mindrake = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_Mindrake\PPOTrainer_CybORGAgent_9dedd_00000_0_2023-08-19_21-11-44\checkpoint_000401\checkpoint-401"
                
                RedPPO_CUABv2_highimpact = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_CUABv2\PPOTrainer_CybORGAgent_228a1_00000_0_2023-08-22_17-29-58\checkpoint_000428\checkpoint-428"
                
                RedPPO_CUABv2 = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-PPO_cuabv2_REDONE\PPOTrainer_CybORGAgent_6bacd_00000_0_2023-08-25_16-09-44\checkpoint_000400\checkpoint-400"
                
                DQN_Mindrake = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\DQNTrainer_2023-Mindrake_continued_405iters\DQNTrainer_CybORGAgent_4c484_00000_0_2023-08-21_20-24-07\checkpoint_000405\checkpoint-405"
                
                DQN_CUABv2 = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\DQNTrainer_2023-CUABv2_continued_429iters\DQNTrainer_CybORGAgent_48597_00000_0_2023-08-22_11-40-16\checkpoint_000429\checkpoint-429"
                
                PPO_cur_both = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_cur_BOTH\PPOTrainer_CybORGAgentCUABv2_28765_00000_0_2023-08-23_13-32-44\checkpoint_000800\checkpoint-800"
                
                PPO_cur_both_400iters = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-ppo_cur_400itersMindrake\PPOTrainer_CybORGAgent_7ff64_00000_0_2023-08-29_21-49-08\checkpoint_000400\checkpoint-400"
                
                ############ USE DQN TRAINER IF EVALUATING DQN ALGO ############################
                ################# HAVE YOU CHANGED THE ENV IN THE CONFIG TO SUIT ??? ################
                
                self.model = ppo.PPOTrainer(config=PPO_config, env=CybORGAgentMindrake)
                self.model.restore(RedPPO_Mindrake)
                
                # self.model = dqn.DQNTrainer(config=DQN_config, env=CybORGAgentMindrake)
                # self.model.restore(DQN_Mindrake)
        else:
            self.model = None
            # self.path = str(inspect.getfile(CybORG))
            # self.path = self.path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            # self.agent_name="Red"
            # self.cyborg = OpenAIGymWrapper(self.agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(self.path, 'sim')))))
            # observation = self.cyborg.reset(agent=self.agent_name)
            # action_space = self.cyborg.get_action_space(self.agent_name)
            # print(f"Observation size {len(observation)}, Action Size {action_space}")
            self.env = None
            self.callback = None
     
    def train(self, results):
        if self.ray:
            print("Training using ray.")
            pass
        else:
            timesteps = 10
            print("Training for {} timesteps".format(timesteps))
            self.model.learn(total_timesteps=timesteps, callback = self.callback)
            self.model.save("PPO_zip")
            
    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    # def __init__(self, model_file, ray=True): # model_file: str = None
    #     if model_file is not None:
    #         print("Loading model from file")
    #         if not ray:
    #             print("Using Stable baselines model saved at", model_file)
    #             self.model = PPO.load(model_file)
    #         else:
    #             print("Using Ray model")
    #             self.model = ppo.PPOTrainer(config=config, env=CybORGMultiAgent)
    #             self.model.restore(model_file)
    #     else:
    #         self.model = None
    #         # self.path = str(inspect.getfile(CybORG))
    #         # self.path = self.path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    #         # self.agent_name="Red"
    #         # self.cyborg = OpenAIGymWrapper(self.agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(self.path, 'sim')))))
    #         # observation = self.cyborg.reset(agent=self.agent_name)
    #         # action_space = self.cyborg.get_action_space(self.agent_name)
    #         # print(f"Observation size {len(observation)}, Action Size {action_space}")
    #         self.env = None
    #         self.callback = None

    def get_action(self, observation, action_space, cyborg):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        if not self.ray: #if self.model is None: # or technically if model is stable baselines model
            print("self.model is None or using StableBaselines model")
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            cyborg = OpenAIGymWrapper('Red', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
            self.env = cyborg
            log_dir = "sb3_logs/"
            eval_env = Monitor(cyborg)
            self.callback = EvalCallback(eval_env=eval_env, log_path=log_dir, eval_freq = 2, verbose=1)
            self.model = PPO('MlpPolicy', cyborg, verbose=1, tensorboard_log="./sb3_logs/tensorboard/") # only called if no model is loaded
        
            action, _states = self.model.predict(observation)
        else:
            #print("Computing action using ray command compute single action")
            action = self.model.compute_single_action(observation)
        #print("ACTION:", action)
        return action
    
    
    def save_model(self, path):
        self.model.save(path)
        print("Saved model at", path)

class RedAgentTrainBlue(BaseAgent):
    # agent that loads a StableBaselines3 PPO model file
    def __init__(self, model_file=True, ray=True): # : str = None
        # ModelCatalog.register_custom_model("CybORG_PPO_Model", CustomModel)
        if ray:
            self.ray=True
        else:
            self.ray=False
        if model_file is True:
            print("Loading model from file")
            if not self.ray:
                print("Using Stable baselines model saved at", model_file)
                self.model = PPO.load(model_file)
            else:
                print("Using Ray model")
                
                RedPPO_cur_Mindrake = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_cur_Mindrake\PPOTrainer_CybORGAgent_d8486_00000_0_2023-08-15_22-08-13\checkpoint_000401\checkpoint-401"
                
                #redPPO_Mindrake = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-08-15_14-23-39_250iter\PPOTrainer_CybORGAgent_f2528_00000_0_2023-08-15_14-23-39\checkpoint_000250\checkpoint-250"
                
                # oldRedPPO_cur_CUAB_v2 = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-CUABV2_371iters_OLD\PPOTrainer_CybORGAgent_1518c_00000_0_2023-08-17_16-09-38\checkpoint_000371\checkpoint-371"
                
                RedPPO_cur_CUABv2_redone = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-PPO_cur_CUABv2_REDONE\PPOTrainer_CybORGAgent_83fcf_00000_0_2023-08-26_20-48-23\checkpoint_000400\checkpoint-400"
                
                RedPPO_cur_CUABv2 = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_cur_CUABv2\PPOTrainer_CybORGAgent_43272_00000_0_2023-08-18_18-40-03\checkpoint_000472\checkpoint-472"
                
                RedPPO_cur_BOTH = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-PPO_cur_BOTH\PPOTrainer_CybORGAgentCUABv2_28765_00000_0_2023-08-23_13-32-44\checkpoint_000800\checkpoint-800"                
                
                RedPPO_Mindrake = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_Mindrake\PPOTrainer_CybORGAgent_9dedd_00000_0_2023-08-19_21-11-44\checkpoint_000401\checkpoint-401"
                
                RedPPO_CUABv2 = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-RedPPO_CUABv2\PPOTrainer_CybORGAgent_228a1_00000_0_2023-08-22_17-29-58\checkpoint_000428\checkpoint-428"
                
                DQN_Mindrake = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\DQNTrainer_2023-Mindrake_continued_405iters\DQNTrainer_CybORGAgent_4c484_00000_0_2023-08-21_20-24-07\checkpoint_000405\checkpoint-405"
                
                DQN_CUABv2 = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\DQNTrainer_2023-CUABv2_continued_429iters\DQNTrainer_CybORGAgent_48597_00000_0_2023-08-22_11-40-16\checkpoint_000429\checkpoint-429"
                
                self.model = ppo.PPOTrainer(config=PPO_cur_config, env=CybORGAgentMindrake)
                self.model.restore(RedPPO_cur_Mindrake)
                
                # self.model = dqn.DQNTrainer(config=DQN_config, env=CybORGAgent)
                # self.model.restore(DQN_CUABv2)
        else:
            self.model = None
            # self.path = str(inspect.getfile(CybORG))
            # self.path = self.path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            # self.agent_name="Red"
            # self.cyborg = OpenAIGymWrapper(self.agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(self.path, 'sim')))))
            # observation = self.cyborg.reset(agent=self.agent_name)
            # action_space = self.cyborg.get_action_space(self.agent_name)
            # print(f"Observation size {len(observation)}, Action Size {action_space}")
            self.env = None
            self.callback = None
     
    def train(self, results):
        if self.ray:
            print("Training using ray.")
            pass
        else:
            timesteps = 10
            print("Training for {} timesteps".format(timesteps))
            self.model.learn(total_timesteps=timesteps, callback = self.callback)
            self.model.save("PPO_zip")
            
    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    # def __init__(self, model_file, ray=True): # model_file: str = None
    #     if model_file is not None:
    #         print("Loading model from file")
    #         if not ray:
    #             print("Using Stable baselines model saved at", model_file)
    #             self.model = PPO.load(model_file)
    #         else:
    #             print("Using Ray model")
    #             self.model = ppo.PPOTrainer(config=config, env=CybORGMultiAgent)
    #             self.model.restore(model_file)
    #     else:
    #         self.model = None
    #         # self.path = str(inspect.getfile(CybORG))
    #         # self.path = self.path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    #         # self.agent_name="Red"
    #         # self.cyborg = OpenAIGymWrapper(self.agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(self.path, 'sim')))))
    #         # observation = self.cyborg.reset(agent=self.agent_name)
    #         # action_space = self.cyborg.get_action_space(self.agent_name)
    #         # print(f"Observation size {len(observation)}, Action Size {action_space}")
    #         self.env = None
    #         self.callback = None

    def get_action(self, observation, action_space): #, cyborg
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        if not self.ray: #if self.model is None: # or technically if model is stable baselines model
#             print("self.model is None or using StableBaselines model")
#             path = str(inspect.getfile(CybORG))
#             path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
#             cyborg = OpenAIGymWrapper('Red', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
#             self.env = cyborg
#             log_dir = "sb3_logs/"
#             eval_env = Monitor(cyborg)
#             self.callback = EvalCallback(eval_env=eval_env, log_path=log_dir, eval_freq = 2, verbose=1)
#             self.model = PPO('MlpPolicy', cyborg, verbose=1, tensorboard_log="./sb3_logs/tensorboard/") # only called if no model is loaded
        
#             action, _states = self.model.predict(observation)
            pass
        else:
            #print("Computing action using ray command compute single action")
            action = self.model.compute_single_action(observation)
        #print("ACTION:", action)
        return action
    
    
    def save_model(self, path):
        self.model.save(path)
        print("Saved model at", path)
        
# class CybORGAgent(gym.Env):
#     max_steps = 100
#     path = str(inspect.getfile(CybORG))
#     path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    
#     # the fixed agent
#     agents = {
#         'Blue': BlueReactRestoreAgent  # , #B_lineAgent, 'Green': GreenAgent
#     }

#     """The CybORGAgent env"""

#     def __init__(self, config: EnvContext):
#         self.cyborg = CybORG(self.path, 'sim', agents=self.agents)

#         # self.env = OpenAIGymWrapper('Blue',
#         #                            EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(self.cyborg))))
#         self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Red') 
#         self.steps = 0
#         self.agent_name = self.env.agent_name
#         self.action_space = self.env.action_space
#         self.observation_space = self.env.observation_space
#         self.action = None

#     def reset(self):
#         self.steps = 1
#         return self.env.reset()

#     def step(self, action=None):
#         result = self.env.step(action=action)
#         self.steps += 1
#         if self.steps == self.max_steps:
#             return result[0], result[1], True, result[3]
#         assert (self.steps <= self.max_steps)
#         return result

#     def seed(self, seed=None):
#         random.seed(seed)


stop = {
    "training_iteration": 100000,   # The number of times tune.report() has been called
    "timesteps_total": 10000000,   # Total number of timesteps
    "episode_reward_mean": -0.01, # When to stop.. it would be great if we could define this in terms
                                # of a more complex expression which incorporates the episode reward min too
                                # There is a lot of variance in the episode reward min
}


#      def get_action_2(self, observation, action_space, path, cyborg):
            
#         """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
#         if self.model is None:
#             # path = str(inspect.getfile(CybORG))
#             # path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
#             # cyborg = OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
#             self.model = PPO('MlpPolicy', cyborg)
#         action, _states = self.model.predict(observation)
#         return action


# class RedAgentTrain_confused(BaseAgent):
#     # agent that loads a StableBaselines3 PPO model file
     
#     def train(self, results):
#         print("Training for 10 timesteps")
#         self.model.learn(total_timesteps=10)

#     def end_episode(self):
#         pass

#     def set_initial_values(self, action_space, observation):
#         pass

#     def __init__(self, model_file: str = None):
#         if model_file is not None:
#             self.model = PPO.load(model_file)
#         else:
#             self.model = None
#             self.callback = None
#             self.path = str(inspect.getfile(CybORG))
#             self.path = self.path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
#             self.agent_name="Red"
#             self.cyborg = OpenAIGymWrapper(self.agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(self.path, 'sim')))))
#             observation = self.cyborg.reset(agent=self.agent_name)
#             action_space = self.cyborg.get_action_space(self.agent_name)
#             print(f"Observation size {len(observation)}, Action Size {action_space}")

#     def get_action(self, observation, action_space):
#         """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
#         if self.model is None:
#             # path = str(inspect.getfile(CybORG))
#             # path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
#             # cyborg = OpenAIGymWrapper('Red', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
#             log_dir = "logs/"
#             self.callback = EvalCallback(eval_env=self.cyborg, log_dir=log_dir)
#             self.model = PPO('MlpPolicy', self.cyborg)
#         action, _states = self.model.predict(observation)
        
#         return action
    
#     def get_action_2(self, observation, action_space):
        
#         if self.model is None:
#             self.model = PPO("MlpPolicy", self.cyborg)
#         action, _states = self.model.predict(observation)

#         return action
    
#     def save_model(self, path):
#         self.model.save(path)
#         print("Saved model at", path)