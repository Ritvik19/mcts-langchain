from __future__ import annotations

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

The authors' [repo](https://github.com/trotsky1997/MathBlackBox) uses critiques,
refinements, and parent nodes' answers as conversation history.
I haven't tried it yet.

"""

import math
import random
from collections import deque
from enum import Enum

import numpy as np
import tqdm
from langchain_core.language_models import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from .llm import chat_completion
from .prompt_configs import PromptConfig

ROOT_UCT_SCORE = 10_000


class MCTSNode(BaseModel):
    answer: str
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    visits: int = 0
    Q: float = 0
    reward_samples: list[int] = []

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)

        # Average worst-case and average outcomes
        self.Q = (min_reward + avg_reward) / 2


class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2


class MCTSr:
    def __init__(
        self,
        model: BaseLLM,
        prompt_config: PromptConfig,
        problem: str,
        max_rollouts: int = 2,
        exploration_constant: float = 1.0,
        max_children: int = 2,
        epsilon: float = 1e-10,
        reward_limit: int = 95,
        excess_reward_penalty: int = 5,
        selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING,
        initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT,
    ):
        self.model = model
        self.prompt_config = prompt_config
        self.problem = problem
        self.max_rollouts = max_rollouts
        self.exploration_constant = exploration_constant
        self.max_children = max_children
        self.epsilon = epsilon
        self.reward_limit = reward_limit
        self.excess_reward_penalty = excess_reward_penalty
        self.selection_policy = selection_policy
        self.initialize_strategy = initialize_strategy
        self.root = MCTSNode(answer="I don't know.")
        self.critiques = []
        self.refinements = []
        self.rewards = []
        self.selected_nodes = []

    def zero_shot(self) -> str:
        response = chat_completion(
            messages=[
                SystemMessage(self.prompt_config.zero_shot_system_prompt),
                HumanMessage(f"<problem>\n{self.problem}\n</problem>"),
            ],
            model=self.model,
        )
        assert response is not None
        return response

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        message4critic = "\n".join(
            [f"<problem>\n{self.problem}\n</problem>", f"<current_answer>\n{node.answer}\n</current_answer>"]
        )
        critique_response = chat_completion(
            messages=[
                SystemMessage(self.prompt_config.critic_system_prompt),
                HumanMessage(message4critic),
            ],
            model=self.model,
        )
        assert critique_response is not None
        self.critiques.append(critique_response)

        message4refine = "\n".join(
            [
                f"<problem>\n{self.problem}\n</problem>",
                f"<current_answer>\n{node.answer}\n</current_answer>",
                f"<critique>\n{critique_response}\n</critique>",
            ]
        )
        refined_answer_response = chat_completion(
            messages=[
                SystemMessage(self.prompt_config.refine_system_prompt),
                HumanMessage(message4refine),
            ],
            model=self.model,
        )
        assert refined_answer_response is not None
        self.refinements.append(refined_answer_response)

        return MCTSNode(answer=refined_answer_response, parent=node)

    def _evaluate_answer(self, node: MCTSNode) -> int:
        messages = [
            SystemMessage(self.prompt_config.evaluate_system_prompt),
            HumanMessage(f"<problem>\n{self.problem}\n</problem>\n<answer>\n{node.answer}\n</answer>"),
        ]
        for attempt in range(3):
            try:
                response = chat_completion(messages=messages, model=self.model)
                assert response is not None
                return int(response)
            except ValueError:
                messages.extend(
                    [
                        AIMessage(response),
                        HumanMessage("Failed to parse reward as an integer."),
                    ]
                )
                if attempt == 2:
                    raise

    def self_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the answer. Sample `num_samples` times and average the results."""
        reward = self._evaluate_answer(node)

        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty

        node.add_reward(reward)

    def backpropagate(self, node: MCTSNode):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visits += 1
            parent = parent.parent

    def uct(self, node: MCTSNode):
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    def select_node(self):
        """Select a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if not candidates:
            return self.root

        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(candidates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(range(len(candidates)), weights=uct_scores, k=1)[0]
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [(i, j) for i in range(len(candidates)) for j in range(len(candidates))]
            pair_weights = [max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j]) for i, j in pairs]
            selected_pair_idx = random.choices(range(len(pairs)), weights=pair_weights, k=1)[0]
            selected_candidate_idx = max(pairs[selected_pair_idx], key=lambda x: uct_scores[x])
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def initialize(self):
        """Generate a zero-shot answer."""
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            self.root = MCTSNode(answer=self.zero_shot())
        elif self.initialize_strategy == InitializeStrategy.DUMMY_ANSWER:
            self.root = MCTSNode(answer="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")

    def run(self):
        self.initialize()
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)

        return self.get_best_answer()

    def get_best_answer(self):
        to_visit = deque([self.root])
        visited = [self.root]
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
                visited.append(current_node)
            to_visit.extend(current_node.children)

        problems_and_observations = "\n\n".join(
            [f"<problem>\n{self.problem}\n</problem>"]
            + [f"<solution_and_observation>\n{node.answer}\n</solution_and_observation>" for node in visited]
        )
        final_solution = chat_completion(
            messages=[
                SystemMessage(self.prompt_config.final_solution_system_prompt),
                HumanMessage(problems_and_observations),
            ],
            model=self.model,
        )

        return final_solution

    def print(self):
        print_tree(self.root)


def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)
