# adv-net-rl
Code repository for studying the threat of adversarial policies against defensive RL agents for network security.

Running this code requires CybORG version 1.2 to be installed. You can find the documentation for the environment code [here.](https://github.com/cage-challenge/CybORG/tree/cage-challenge-1)

Copying the agents included in the `agents/` directory of this repo into the `Agents/` directory of the CybORG installation will should provide the extended framework to load victim agents against which you may train adversarial policies as well as for evaluation.

Running the scripts in `training/` will train red agents that will learn adversarial policies against RL defence agents. You must specify the victim in the ```CybORGAgent``` class in the training script. Separate scripts exist to train a red agent with PPO with Curiosity, PPO or DDQN.

The `evaluation/` directory contains the script with which you can evaluate red agents against both victims. The ```RedAgentTrain.py``` script is where you may change the red agent model checkpoint to use during evaluation. Currently this is done by specifying the file path to the model checkpoint.

The checkpoints to the defence agents used for the experiments conducted in this research can be found at the following links:
- [Mindrake](https://github.com/alan-turing-institute/cage-challenge-1-public)
- [CUABv2](https://github.com/mprhode/cyborg-submission-CUABv2)



