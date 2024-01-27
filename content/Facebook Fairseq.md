## Overview 
Facebook Fairseq is an open-source toolkit for sequence modeling, primarily used for training custom models for tasks like machine translation, text summarization, and other text-generation applications. Think of it as a toolbox specifically designed for tasks that involve processing sequences of text data.

## Features
- multi-GPU training on one machine or across multiple machines
- fast generation on both CPU and GPU with multiple search algorithms implemented
	- [[Beam Search]] / [[Diverse Beam Search]]
	- Sampling (Unconstrained, Top-k and Top-p/nucleus)
	- [[Lexically Constrained Decoding]]
- [[Gradient Accumulation]] enables training with large mini-batches even on single GPU
- [[Mixed Precision Training]]
- extensible: easily register new models, criteria, tasks, optimizers and learning rate schedulers
- flexible configuration based on [[Hydra]] allowing a combination of code, command-line and file-based configurations
- full parameter and optimizer state sharding
- offloading parameters to CPU

## Notes
### Resources
- [Fairseq Documentation](https://fairseq.readthedocs.io/en/latest/)
- [Fairseq Github Repository](https://github.com/facebookresearch/fairseq)
### Further Readings
- 