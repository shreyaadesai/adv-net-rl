from CybORG.Agents.SimpleAgents.DQNAgent import RNNDQNAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

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

eval_agent_dir = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cyborg-submission-CUABv2\saved_best_model" 

class MainAgent(BaseAgent):
    def __init__(self, suffix=31):
        self.end_episode()

        self.agent = RNNDQNAgent(
            input_dims=(52,),
            n_actions=54,
            lookback_steps=16,
            epsilon=0, chkpt_dir=eval_agent_dir, #"/saved_best_model",
            algo=f'RNNDDQNAgent_{suffix}',
            env_name='Scenario1b')
        self.agent.load_models()
        
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
        #print("POSSIBLE ACTIONS (PRINTED FROM MAINAGENT.PY):", possible_actions)
        return len(possible_actions)



    def _parse_scenario(self, scenario_file_path: str, scenario_mod: dict = None):
        with open(scenario_file_path) as fIn:
            scenario_dict = yaml.load(fIn, Loader=yaml.FullLoader)
        return scenario_dict

    def get_action(self, observation, action_space=None):
        
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
        action = self.agent.get_action(observation, action_space=action_space)
        action =  self.possible_actions[action]
        self.previous_action = action
        #print("ACTION (PRINTED FROM MAINAGENT.PY):", action)
        
        return action #self.agent.get_action(observation, action_space=action_space)

    def train(self, results):
        pass

    def end_episode(self):
        self.blue_info = {}
        self.previous_action = -1
        pass

    def set_initial_values(self, action_space, observation):
        pass
    
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

    