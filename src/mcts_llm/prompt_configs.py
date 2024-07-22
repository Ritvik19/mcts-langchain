from pydantic import BaseModel, Field


class RefineResponse(BaseModel):
    thought: str = Field(..., description="The thought process behind the answer.")
    answer: float = Field(..., description="The answer to the problem.")

class PromptConfig(BaseModel):
    zero_shot_system_prompt: str = "The user will provide a problem. Solve the problem. Think step by step."
    critic_system_prompt: str = ("Provide a detailed and constructive critique to improve the answer."
    "Highlight specific areas that need refinement or correction.")
    refine_system_prompt: str="""# Instruction
Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.
- Respond with only the answer.
"""
    evaluate_system_prompt: str=(
        "Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
        "Do not give a full score above 95. Make sure the reward score is an integer. "
        "Return *ONLY* the score."
    )
    final_solution_system_prompt: str="Given the following solutions and observations, construct a final solution"


llama_3_8b_prompt_config = PromptConfig()

gpt_4o_prompt_config = PromptConfig(
    refine_system_prompt="""# Instruction
Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.

# JSON Response format
{
    "thought": "The thought process behind the answer.",
    "answer": "A float representing the answer to the problem."
}
""")
