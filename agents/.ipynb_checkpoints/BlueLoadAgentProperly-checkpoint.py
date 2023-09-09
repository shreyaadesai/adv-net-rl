import inspect
from prettytable import PrettyTable
from copy import deepcopy

from stable_baselines3 import PPO
import numpy as np
import yaml
from CybORG import CybORG
from CybORG.Shared import Scenario
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.ActionSpace import ActionSpace

import os
from pprint import pprint
import os.path as ppath
import numpy as np
import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo

from CybORG.Agents.SimpleAgents.train_hier import CustomModel, TorchModel
from CybORG.Agents.SimpleAgents.CybORGAgent import CybORGAgent
#from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
#from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
#from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
#from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper

class BlueLoadAgent(BaseAgent):
    # agent that loads a StableBaselines3 PPO model file
    def train(self, results):
        pass

    def end_episode(self):
        self.blue_info = {}
        self.previous_action = -1
        pass
    
    def set_red_agent(self, red_agent):
        self.red_agent = red_agent

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, model_file: str = None, RM = False):
        self.blue_info = {}
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
        scenario_dict = self._parse_scenario(path)
        scenario = Scenario(scenario_dict)
        agent_info = scenario.get_agent_info('Blue')
        self.actions = ActionSpace(agent_info.actions, "Blue", agent_info.allowed_subnets)
        self.output_mode = 'vector'
        self.previous_action = -1
        self.baseline = None
        # if model_file is not None:
        #     self.model = PPO.load(model_file)
        # else:
        #     self.model = None
            
        ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)
        #relative_path = os.path.abspath(os.getcwd())[:62] + '/cage-challenge-1'
        #print("Relative path:", relative_path)
        # if RM == True:
        #     self.RM = True
        # else:
        #     self.RM = False
        
        self.agent_to_select = random.choice([int(0), int(1)])
        
        # Load checkpoint locations of each agent
        two_up = ppath.abspath(ppath.join(__file__, "../../../"))
        #self.CTRL_checkpoint_pointer = two_up + '/log_dir/rl_controller_scaff/PPO_HierEnv_1e996_00000_0_2022-01-27_13-43-33/checkpoint_000212/checkpoint-212'
        self.BL_checkpoint_pointer = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-redo\cage-challenge-1\CybORG\CybORG\log_dir\b_line_trained\PPO_CybORGAgent_e81fb_00000_0_2022-01-29_11-23-39\checkpoint_002500\checkpoint-2500"
        # two_up + '/log_dir/b_line_trained/PPO_CybORGAgent_e81fb_00000_0_2022-01-29_11-23-39/checkpoint_002500/checkpoint-250'
        
        #"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-redo\cage-challenge-1\CybORG\CybORG\log_dir\b_line_trained\PPO_CybORGAgent_e81fb_00000_0_2022-01-29_11-23-39\checkpoint_002500\checkpoint-2500"
        
        self.RM_checkpoint_pointer = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-redo\cage-challenge-1\CybORG\CybORG\log_dir\meander_trained\PPO_CybORGAgent_3c456_00000_0_2022-01-27_20-39-34\checkpoint_001882\checkpoint-1882"
        #two_up + '/log_dir/meander_trained/PPO_CybORGAgent_3c456_00000_0_2022-01-27_20-39-34/checkpoint_001882'
        
        #"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-redo\cage-challenge-1\CybORG\CybORG\log_dir\meander_trained\PPO_CybORGAgent_3c456_00000_0_2022-01-27_20-39-34\checkpoint_001882"

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        #print("Using checkpoint file (Controller): {}".format(self.CTRL_checkpoint_pointer))
        # print("Using checkpoint file (B-line): {}".format(self.BL_checkpoint_pointer))
        # print("Using checkpoint file (Red Meander): {}".format(self.RM_checkpoint_pointer))

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
        print("Restoring blue Mindrake agents from checkpoints...")
        #load agent trained against RedMeanderAgent
        self.RM_def = ppo.PPOTrainer(config=sub_config_M, env=CybORGAgent)
        self.RM_def.restore(self.RM_checkpoint_pointer)
        #load agent trained against B_lineAgent
        self.BL_def = ppo.PPOTrainer(config=sub_config_BL, env=CybORGAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)
        print("Agents restored.")

    def action_space_change(self, action_space: dict) -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        possible_actions = []
        temp = {}
        params = ['action']
        # for action in action_space['action']:
        for i, action in enumerate(action_space['action']):
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            param_dict = {}
            param_list = [{}]
            for p in self.action_signature[action]:
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))

        self.possible_actions = possible_actions
        #print("POSSIBLE ACTIONS: (BLUEAGENTLOADPROPERLY.PY)", possible_actions)
        #print("
        return len(possible_actions)



    def _parse_scenario(self, scenario_file_path: str, scenario_mod: dict = None):
        with open(scenario_file_path) as fIn:
            scenario_dict = yaml.load(fIn, Loader=yaml.FullLoader)
        return scenario_dict

    def get_action(self, observation, action_space):
        #print("Executing get_action function")
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        #if self.model is None:
        #    path = str(inspect.getfile(CybORG))
        #    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
        #    cyborg = OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
        #    self.model = PPO('MlpPolicy', cyborg)
        if self.blue_info == {}:
            self._process_initial_obs(observation)
        #print(observation)
        
        #print(self.actions.actions)
        self.action_signature = {}
        self.possible_actions = []
        self.actions.update(observation)
        #print(self.actions.get_action_space())
        self.action_space_change(self.actions.get_action_space())
        #print(self.possible_actions)
        #exit(3)

        observation = self.observation_change(observation)
        #print(observation)

        # action, _states = self.model.predict(observation)
        #action = 3
        if self.agent_to_select == 1:
            action = self.RM_def.compute_single_action(observation)
        else:
            action = self.BL_def.compute_single_action(observation)
        #print("COMPUTED ACTION FROM MODEL:", action)
        action =  self.possible_actions[action]
        self.previous_action = action
        #print("BLUE ACTION AFTER SELECTING FROM POSSIBLE ACTIONS:", action)
        #print("PRINT STATEMENTS FROM BLUELOADAGENTPROPERLY.PY")
        return action

    def observation_change(self,observation, baseline=False):
        obs = observation if type(observation) == dict else observation.data
        obs = deepcopy(observation)
        success = obs['success']

        self._process_last_action()
        anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
        del obs['success']
        # TODO check what info is for baseline
        info = self._process_anomalies(anomaly_obs)
        if baseline:
            for host in info:
                info[host][-2] = 'None'
                info[host][-1] = 'No'
                self.blue_info[host][-1] = 'No'

        self.info = info

        return self._create_vector(success)
        
    def _process_anomalies(self,anomaly_dict):
        info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if 'Processes' in host_anomalies:
                connection_type = self._interpret_connections(host_anomalies['Processes'])
                info[hostid][-2] = connection_type
                if connection_type == 'Exploit':
                    info[hostid][-1] = 'User'
                    self.blue_info[hostid][-1] = 'User'
            if 'Files' in host_anomalies:
                malware = [f['Density'] >= 0.9 for f in host_anomalies['Files']]
                if any(malware):
                    info[hostid][-1] = 'Privileged'
                    self.blue_info[hostid][-1] = 'Privileged'

        return info

    def _detect_anomalies(self,obs):
        if self.baseline is None:
            raise TypeError('BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method.')

        anomaly_dict = {}

        for hostid,host in obs.items():
            if hostid == 'success':
                continue

            host_baseline = self.baseline[hostid]
            if host == host_baseline:
                continue

            host_anomalies = {}
            if 'Files' in host:
                baseline_files = host_baseline.get('Files',[])
                anomalous_files = []
                for f in host['Files']:
                    if f not in baseline_files:
                        anomalous_files.append(f)
                if anomalous_files:
                    host_anomalies['Files'] = anomalous_files

            if 'Processes' in host:
                baseline_processes = host_baseline.get('Processes',[])
                anomalous_processes = []
                for p in host['Processes']:
                    if p not in baseline_processes:
                        anomalous_processes.append(p)
                if anomalous_processes:
                    host_anomalies['Processes'] = anomalous_processes

            if host_anomalies:
                anomaly_dict[hostid] = host_anomalies

        return anomaly_dict

    def _interpret_connections(self,activity:list):                
        num_connections = len(activity)
        ports = set([item['Connections'][0]['local_port'] \
            for item in activity if 'Connections' in item])
        port_focus = len(ports)

        remote_ports = set([item['Connections'][0]['remote_port'] \
            for item in activity if 'Connections' in item])

        if num_connections >= 3 and port_focus >=3:
            anomaly = 'Scan'
        elif 4444 in remote_ports:
            anomaly = 'Exploit'
        elif num_connections >= 3 and port_focus == 1:
            anomaly = 'Exploit'
        else:
            anomaly = 'Scan'

        return anomaly

    def _process_initial_obs(self, obs):
        # TODO remove deepcopy replace with dict comprehension
        obs = deepcopy(obs)
        self.baseline = obs
        del self.baseline['success']
        for hostid in obs:
            if hostid == 'success':
                continue
            host = obs[hostid]
            interface = host['Interface'][0]
            subnet = interface['Subnet']
            ip = str(interface['IP Address'])
            hostname = host['System info']['Hostname']
            self.blue_info[hostname] = [str(subnet),str(ip),hostname, 'None','No']
        return self.blue_info

    def _process_last_action(self):
        action = self.previous_action
        if action is not None:
            name = action.__class__.__name__
            hostname = action.get_params()['hostname'] if name in ('Restore','Remove') else None

            if name == 'Restore':
                self.blue_info[hostname][-1] = 'No'
            elif name == 'Remove':
                compromised = self.blue_info[hostname][-1]
                if compromised != 'No':
                    self.blue_info[hostname][-1] = 'Unknown'


    def _create_blue_table(self, success):
        table = PrettyTable([
            'Subnet',
            'IP Address',
            'Hostname',
            'Activity',
            'Compromised'
            ])
        for hostid in self.info:
            table.add_row(self.info[hostid])
        
        table.sortby = 'Hostname'
        table.success = success
        return table

    def _create_vector(self, success):
        table = self._create_blue_table(success)._rows

        proto_vector = []
        for row in table:
            # Activity
            activity = row[3]
            if activity == 'None':
                value = [0,0]
            elif activity == 'Scan':
                value = [1,0]
            elif activity == 'Exploit':
                value = [1,1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

            # Compromised
            compromised = row[4]
            if compromised == 'No':
                value = [0, 0]
            elif compromised == 'Unknown':
                value = [1, 0]
            elif compromised == 'User':
                value = [0,1]
            elif compromised == 'Privileged':
                value = [1,1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

        return np.array(proto_vector)
    
    
    
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