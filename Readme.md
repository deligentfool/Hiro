# HIRO

[paper](https://arxiv.org/abs/1805.08296)

This is a try to achieve the implement of HIRO, a hierarchy reinforcement learning algorithm.

Because I use the **TD3** to be the basic structure of both high level policy and low level policy, there are **12 neural network** that need to be trained. Maybe the training speed is slow.

HIRO perhaps only fit to low dim observation space problems due to its some features, so I test it under *Pendulum* environment. The test shows HIRO performs well in this case.