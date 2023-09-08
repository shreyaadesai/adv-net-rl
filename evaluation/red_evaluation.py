import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgentProperly import BlueLoadAgent
from CybORG.Agents.SimpleAgents.MainAgent import MainAgent # amended to be loaded in for eval, MainAgentOriginal is the other one (not changed)
#from CybORG.Agents.SimpleAgents.MainAgentv1RedEval import MainAgentv1
# from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent, BlueReactRestoreAgent
# from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

#from CybORG.Agents.SimpleAgents.RedPPOAgent import RedPPOAgentClass as RedPPOAgent
from CybORG.Agents.SimpleAgents.RedAgentTrain import RedAgentTrain

from CybORG.Shared.Actions.Action import InvalidAction

MAX_EPS = 10
agent_name = 'Red'

mindrake = BlueLoadAgent
cuab_v2 = MainAgent
#cuab_v1 = MainAgentv1

def wrap(env):
    return OpenAIGymWrapper(agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))


if __name__ == "__main__":
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    # ask for a name
    name = input('Name: ') #RedPPO
    # ask for a team
    team = "adv-net-rl" #input("Team: ")
    # ask for a name for the agent
    name_of_agent = input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    #ray_file_path = r"C:\Users\shrey\OneDrive\Imperial Masters\Dissertation\cage-challenge-1-public\agents\hierachy_agents\logs\PPOTrainer_2023-08-15_14-23-39_250iter\PPOTrainer_CybORGAgent_f2528_00000_0_2023-08-15_14-23-39\checkpoint_000250\checkpoint-250"
    #agent = RedAgentTrain(model_file=ray_file_path) # RedPPOAgent()
    agent = RedAgentTrain()

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]: # 
        for blue_agent in [cuab_v2]: # cuab_v2, mindrake
            print("Evaluating {} against {}...".format(agent.__class__.__name__, blue_agent))
            cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
            wrapped_cyborg = ChallengeWrapper(env=cyborg, agent_name='Red')

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            both_rewards = []
            for i in range(MAX_EPS):
                r = []
                a = []
                both_r = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    invalid = False
                    action = agent.get_action(observation, action_space, wrapped_cyborg)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    if type(info['action']) == InvalidAction:
                        invalid=True
                        while invalid == True:
                            #print("Action Invalid:", info['action'])
                            action = agent.get_action(observation, action_space, wrapped_cyborg)
                            observation, rew, done, info = wrapped_cyborg.step(action=action)     
                            if type(info['action']) != InvalidAction:
                                invalid = False
                    
                    # result = cyborg.step(agent_name, action)
                    # r.append(rew)
                    blue_r = cyborg.get_last_reward('Blue')
                    r.append(blue_r)
                    # both_r.append(("Rew:", str(rew), "Red get last reward:", str(cyborg.get_last_reward('Red'), "Blue get last reward",  str(cyborg.get_last_reward('Blue')))))
                    both_r.append((str(rew), str(cyborg.get_last_reward('Blue'))))
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Red')), str(cyborg.get_last_action('Blue'))))
                total_reward.append(sum(r))
                actions.append(a)
                both_rewards.append(both_r)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for blue agent {str(blue_agent)} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}') #agent.__name__
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, victim: {blue_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, both_rews, sum_rew in zip(actions, both_rewards, total_reward):
                    data.write(f'actions: {act}, each agent reward: {both_rews}, TOTAL {agent.__class__.__name__} REWARD: {sum_rew}\n')

                    
# for step in range(MAX_STEPS):
#     invalid = False
#     action = agent.get_action(observation,action_space)
#     next_observation, reward, done, info = env.step(action=action)      
    
#     if type(info['action']) == InvalidAction:
#         invalid=True
#         while invalid == True:
#             #print("Action Invalid:", info['action'])
#             action = agent.get_action(observation,action_space)
#             next_observation, reward, done, info = env.step(action=action)     
#             if type(info['action']) != InvalidAction:
#                 invalid = False