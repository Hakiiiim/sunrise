# SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning

Official codebase for [SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning](https://arxiv.org/abs/2007.04938). 

This repo was forked from the original repo of sunrise by pokaxpoka.

My contribution consist of the design and implementation of a maze-like environment to test SUNRISE.


*Abstract*:

In the framework of approximate Reinforcement Learning, deep neural networks brought a true improvement to the performance of RL agents in previously human-dominated arenas (e.g. Alpha-GO). However, the optimization of these non-linear universal approximators remains a tough task that induces instability in the learning process. In order to address this issue -but not only-, SUNRISE was conceived as a unified framework that can be mounted on any off-policy algorithm to boost its performance. In this project, we first try to reproduce some of the original paper's results on Atari environments, we then design a challenging maze-like environment to further assess the performance of SUNRISE.

*Environment description*:

The choice of such an environment results from its particular difficulty towards model-free RL algorithms. Indeed, the reward is very sparse and exploration is a key ingredient to succeeding in such environment. Hence the relevance of evaluating the SUNRISE framework on this setting.

The properties of the environment are:


* Composed of an Agent (yellow), a Goal (green), and Traps (blue).
* The terminal states are the Goal and the Traps.
* A sparse reward, Moving penalty (-1), Out-of-the-maze penalty (-5), Falling into trap penalty (-10), Goal reward (+10)
* Lifetime, The game terminates if a lower bound on the total reward is reached (-100)


The experiments will be conducted using the default hyper-parameters stated by the paper for discrete control tasks: N = 5, beta = 1, T = 40, and lambda = 10. The reference algorithm is again RainbowDQN.

We tried SUNRISE on Three different configurations: (SIZE,N_TRAPS) \in {(5,3),(7,5),(10,10)}.

SUNRISE managed to find its way out throughout the maze in the three cases (with different training times though), the average reward is shown in Figure and Appendix.


The three configurations required respectively 10k, 10k, and 30k training iterations to solve the maze. It is also important to mention that training SUNRISE takes up to 3 or 4 times the time required for RainbowDQN.

On the (SIZE = 7, N_TRAPS = 5) configuration, SUNRISE managed to find the shortest path to the goal state, an illustration is in figure.

