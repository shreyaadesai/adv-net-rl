#### load original Mindrake to be evaluated against B_line + Meander adversaries

import os
from pprint import pprint
import os.path as path
import numpy as np
import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from CybORG.Agents.SimpleAgents.train_hier import CustomModel, TorchModel
#from agents.hierachy_agents.scaffold_env import CybORGScaffRM, CybORGScaffBL
#`from agents.hierachy_agents.hier_env import HierEnv
import os
from CybORG.Agents import B_lineAgent, SleepAgent, RedMeanderAgent
from CybORG.Agents import BaseAgent
#from agents.hierachy_agents.sub_agents import sub_agents
# from agents.hierachy_agents.CybORGAgent import CybORGAgent
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

from CybORG.Agents.SimpleAgents.hier_env import HierEnv
from CybORG.Agents.SimpleAgents.sub_agents import sub_agents
from CybORG.Agents.SimpleAgents.CybORGAgent import CybORGAgent
from CybORG.Agents.SimpleAgents.CybORGRedAgent import CybORGMultiAgent

from CybORG.Agents.Wrappers.BlueTableWrapper import *

os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'

class LoadRedAgent:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_PPO_Model_red", TorchModel)
        #relative_path = os.path.abspath(os.getcwd())[:62] + '/cage-challenge-1'
        #print("Relative path:", relative_path)
        self.baseline = None
        self.red_checkpoint = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\multi_agent_cage\CybORG\CybORG\Agents\logs\PPOTrainer_2023-07-31_18-32-43\PPOTrainer_CybORGAgent_41c65_00000_0_2023-07-31_18-32-44\checkpoint_000413\checkpoint-413"
        

        print("Using checkpoint file (Red Agent): {}".format(self.red_checkpoint))

        config = Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,{
        "env": CybORGMultiAgent,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_PPO_Model_red",
            "vf_share_layers": False,
        },
        "lr": 0.0005,
        #"momentum": tune.uniform(0, 1),
        "num_workers": 0,  # parallelism
        "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
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

        # Restore the red agent model
        self.red_agent = ppo.PPOTrainer(config=config, env=CybORGMultiAgent)
        self.red_agent.restore(self.red_checkpoint)
        print("Restored model from checkpoint")
        self.blue_agent=-1


    def set_blue_agent(self, blue_agent):
        self.blue_agent = blue_agent

    """Compensate for the different method name"""
    def get_action(self, obs, action_space, cyborg):
        print("OBSERVATION:", obs)
        #update sliding window
        # self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        # self.observation[HierEnv.mem_len-1] = obs           # Replace what's on the rightmost position
        agent_action = self.red_agent.compute_single_action(obs)

        #self.controller_agent.compute_single_action(self.observation)
        #agent_to_select = 1#np.random.choice([0,1]) # hard-coded meander agent only
        # if agent_to_select == 0:
        #     # get action from agent trained against the B_lineAgent
        #     agent_action = self.BL_def.compute_single_action(self.observation[-1:])
        # elif agent_to_select == 1:
        #     # get action from agent trained against the RedMeanderAgent
        #     agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        return agent_action#, agent_to_select

class LoadBlueAgent_changed(BaseAgent):

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)
        #relative_path = os.path.abspath(os.getcwd())[:62] + '/cage-challenge-1'
        #print("Relative path:", relative_path)
        self.RM = True
        
        # Load checkpoint locations of each agent
        two_up = path.abspath(path.join(__file__, "../../../"))
        #self.CTRL_checkpoint_pointer = two_up + '/log_dir/rl_controller_scaff/PPO_HierEnv_1e996_00000_0_2022-01-27_13-43-33/checkpoint_000212/checkpoint-212'
        self.BL_checkpoint_pointer = two_up + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        #print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

        sub_config_BL = {
            "env": CybORGAgent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
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
        }

        sub_config_M = {
            "env": CybORGAgent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                #"vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
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
        }

        #load agent trained against RedMeanderAgent
        self.RM_def = ppo.PPOTrainer(config=sub_config_M, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        #load agent trained against B_lineAgent
        self.BL_def = ppo.PPOTrainer(config=sub_config_BL, env=CybORGAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)
    
    def get_action(self, obs, action_space):
        
#         env = BlueTableWrapper(cyborg,output_mode='vector')

#         env.reset(agent='Blue')
#         for i in range(3):
#             results = env.step(agent='Blue')
#             blue_obs = results.observation
#             print(blue_obs)
#             print(76*'-')
        print("INPUT OBS SHAPE:", len(obs))
        vectorised_obs = self.observation_change(obs, baseline=True)
        print("Vectorised_obs:", vectorised_obs)
        # observation = BlueTableWrapper(obs)s
        # print("CONVERTED OBS:", observation)#, "OBS:", obs)
        #update sliding window
        #self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        #self.observation= obs #[HierEnv.mem_len - 1]          # Replace what's on the rightmost position

        #select agent to compute action
        # if self.red_agent == B_lineAgent or self.red_agent == SleepAgent:
        #     agent_to_select = 0 # i.e. B_line defence
        # else: # select RedMeanderAgent
        #     agent_to_select = 1
        
        
        # agent_to_select = 1
        #self.controller_agent.compute_single_action(self.observation)
        #agent_to_select = 1#np.random.choice([0,1]) # hard-coded meander agent only
        if self.RM:
            # get action from agent trained against the B_lineAgent
            agent_action = self.RM_def.compute_single_action(vectorised_obs) #self.observation[-1:]
        else:
            # get action from agent trained against the RedMeanderAgent
            agent_action = self.BL_def.compute_single_action(vectorised_obs) #self.observation[-1:]
            #print("BLUE AGENT SELECTING action:", agent_action)
        return agent_action#, agent_to_select

    def train(self, results):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass
    

class LoadBlueAgent_original:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)
        #relative_path = os.path.abspath(os.getcwd())[:62] + '/cage-challenge-1'
        #print("Relative path:", relative_path)

        # Load checkpoint locations of each agent
        two_up = path.abspath(path.join(__file__, "../../../"))
        #self.CTRL_checkpoint_pointer = two_up + '/log_dir/rl_controller_scaff/PPO_HierEnv_1e996_00000_0_2022-01-27_13-43-33/checkpoint_000212/checkpoint-212'
        self.CTRL_checkpoint_pointer = two_up + '/log_dir/controller_1_step/PPO_HierEnv_ebf84_00000_0_2022-02-01_10-30-05/checkpoint_000251/checkpoint-251'
        self.BL_checkpoint_pointer = two_up + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + sub_agents['RedMeander_trained']

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

        config = Trainer.merge_trainer_configs(
            DEFAULT_CONFIG,
            {
            "env": HierEnv,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_hier_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            #"momentum": tune.uniform(0, 1),
            "num_workers": 4,  # parallelism
            "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
             "in_evaluation": True,
            'explore': False
        })

        # Restore the controller model
        self.controller_agent = ppo.PPOTrainer(config=config, env=HierEnv)
        self.controller_agent.restore(self.CTRL_checkpoint_pointer)
        self.observation = np.zeros((HierEnv.mem_len,52))
        print('loaded controller')
        sub_config_BL = {
            "env": CybORGAgent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
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
        }

        sub_config_M = {
            "env": CybORGAgent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                #"vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
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
        }

        #load agent trained against RedMeanderAgent
        self.RM_def = ppo.PPOTrainer(config=sub_config_M, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        print('loaded rm')
        #load agent trained against B_lineAgent
        self.BL_def = ppo.PPOTrainer(config=sub_config_BL, env=CybORGAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)
        print('loaded bl')
        self.red_agent=-1


    def set_red_agent(self, red_agent):
        self.red_agent = red_agent

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        #update sliding window
        self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        self.observation[HierEnv.mem_len-1] = obs           # Replace what's on the rightmost position

        #select agent to compute action
        if self.red_agent == B_lineAgent or self.red_agent == SleepAgent:
            agent_to_select = 0
        else: #RedMeanderAgent
            agent_to_select = 1

        #self.controller_agent.compute_single_action(self.observation)
        #agent_to_select = 1#np.random.choice([0,1]) # hard-coded meander agent only
        print("INPUT OBS TO COMPUTE ACTION:", self.observation[-1:])
        if agent_to_select == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BL_def.compute_single_action(self.observation[-1:])
        elif agent_to_select == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        return agent_action#, agent_to_select