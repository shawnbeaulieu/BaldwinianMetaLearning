# Implementation of Meta-Learning by the Baldwin Effect: 

Dependencies: NumPy, PyTorch

For use in Python 3.6. A comparatively simple instantiation of [Meta-Learning by the Baldwin Effect](https://arxiv.org/abs/1806.07917) (Fernando et al. 2018) in which evolution acts on a population of neural networks that learn to perform a set of tasks
according to some local update rule(s) starting from an initial state which is subject to selection. Fit individuals are those 
that can obtain high performance in relatively few local updates. By virtue of Baldwinian evolution, which is nothing more than 
an emergent process of Darwinian evolution, beneficial network states that are discovered during the local update become
progressively "preponed"; that is, they arise earlier in the "lifetime" of the agent until they're present at birth, at which
point the cost of learning them is removed.

In the same spirit, a paper by myself, Sam Kriegman, and Josh Bongard, [Combating Catastrophic Forgetting with Developmental Compression](https://arxiv.org/abs/1804.04286), was concurrently published at the same conference (GECCO 2018).

