# mcts-langchain

This is a fork of [BrendanGraham14/mcts-llm](https://github.com/BrendanGraham14/mcts-llm), I have integrated langchain implementations for model call and refactored the code.

## MCTSr

Based on [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394) by Zhang, et al.

At a high level, MCTSr iteratively generates solutions to a specified (math) problem.

In a MCTSr tree, nodes correspond to attempted answers, and edges correspond to attempts to improve the answer.


### Initialize
Generate an solution to the problem. This paper uses a "dummy" solution (e.g. `"I don't know"`).

### Select a node to expand
We gather a set of candidate nodes which haven't been fully expanded.

A node is fully expanded if either:
1. it has `max_children`
2. any of its children have a Q value which is greater than its own

Once we've gathered the candidates, we compute UCT scores for each candidate node.
There are a few ways we can make our selection:
1. Greedily (choose the node with the highest UCT)
2. Importance sampling (sample from the set of candidates, weighted by their UCT score)
3. Pairwise importance sampling (sample the max from a pair of nodes from the set of candidates, weighted by the difference between the pair's UCT scores)

The authors mention that they perform greedy selection in the paper. In their [repo](https://github.com/trotsky1997/MathBlackBox/blob/main/gen_mcts_dpo.py#L182), they also perform pairwise sampling and save the (question, answer1, answer2) tuples for use in DPO.

### Expand the node

Expansion involves several steps:
1. Generate a critique of the current solution.
2. Refine the solution based on the critique.
3. Add a new child, corresponding to the refined solution.
4. Self-evaluate the `reward` of the new child.
5. Backpropagate the reward from the new child through its parents, through to the root.


# Usage

### Imports
```python
from mcts_llm.mctsr import MCTSr
from mcts_llm.prompt_configs import llama_3_8b_prompt_config
```

### Instantiate the bLLM
```python
from langchain_community.llms import Ollama

model = Ollama(model="llama3:8b")
```

### Just Run It!!
```python
llama = MCTSr(model=model, problem="what are the cube roots of unity", prompt_config=llama_3_8b_prompt_config)
llama.run()
```