import gym
import inspect
import random
from ray.rllib.env.env_context import EnvContext

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRestoreAgent, BlueReactRemoveAgent
# from CybORG.Agents.SimpleAgents.RedPPOAgent import RedPPOAgent
#from CybORG.Agents.SimpleAgents.loadagent import LoadRedAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

class CybORGAgent(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    
    # the fixed agent
    agents = {
        'Red': RedMeanderAgent  # , #B_lineAgent, 'Green': GreenAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)

        # self.env = OpenAIGymWrapper('Blue',
        #                            EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(self.cyborg))))
        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Blue') # this shouldve been blue :( 
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

    def seed(self, seed=117):
        random.seed(seed)


class CybORGAgent_old(gym.Env):
    max_steps = 100
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    
    # these are the agents that move automatically in the environmnet.
    # want this to be blue Mindrake agent
    # CHANGE THIS TO JUST THE MEANDER DEFENCE TRAINED.
    agents = {
        #'Blue': BlueLoadAgentAdapted(model_file=RedMeanderTrainedPath) #
        'Red': B_lineAgent #LoadRedAgent()#, BlueReactRemoveAgent LoadBlueAgent#_original  # , #B_lineAgent, 'Green': GreenAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)

        # self.env = OpenAIGymWrapper('Blue',
        #                            EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(self.cyborg))))
        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Blue') # what team are you on 
        self.steps = 0
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()

    def step(self, action=None):
        # print("Agent stepping in env:", self.agent_name)
        # print("Taking action:", action)
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == self.max_steps:
            return result[0], result[1], True, result[3]
        assert (self.steps <= self.max_steps)
        return result

    def seed(self, seed=2):
        random.seed(seed)