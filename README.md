# adv-net-rl
Code repository for studying the threat of adversarial policies against defensive RL agents for network security.

Running this code requires CybORG version 1.2 to be installed. 

Copying the agents included in the `agents/` directory of this repo into the `Agents/` directory of the CybORG installation will enable the existing agents to be loaded into the environment. 

Running the scripts in `training/` will train adversarial policies against RL defence agents. You must specify the victim in the ```CybORGAgent``` class in the training script. Separate scripts exist to train a red agent with PPO with Curiosity, PPO or DDQN.

The `evaluation/` directory contains the script with which you can evaluate red agents against both victims. The ```RedAgentTrain.py``` script is where you may change the red agent model checkpoint to use during evaluation. Currently this is done by specifying the file path to the model checkpoint.

