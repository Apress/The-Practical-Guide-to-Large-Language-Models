from typing import Any
from smolagents import CodeAgent, TransformersModel, tool, AgentText, InferenceClientModel

# Local model
# coder_model = TransformersModel(
#     model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# )

# Inference Client model
hf_token = 'your_token_here'  # Replace with your Hugging Face token
coder_model = InferenceClientModel(
    model_id = 'Qwen/Qwen2.5-Coder-32B-Instruct',
    api_key = hf_token
)


# Define a custom final answer check function
# This function checks if the final answer is a valid integer.
def is_completed(final_answer: Any, agent_memory = None) -> bool:
    final_answer = str(final_answer)
    try:
        int(final_answer.strip())
        return True
    except ValueError:
        return False


# Next goes tool for summing two integers
@tool
def sum_ints(
        a: int,
        b: int,
) -> int:
    """
    Returns the sum of two integers.
    Example:
        sum(3, 5) returns 8
    Args:
        a (int): The first integer
        b (int): The second integer
    """
    return a + b


# Now we can create the CodeAgent with the custom final answer check and the tool
agent = CodeAgent(
    provide_run_summary = True,
    model = coder_model,
    tools = [
        sum_ints
    ],
    final_answer_checks = [
        is_completed
    ],
)

# Run the agent with a simple task: counting the sum of two integers
result: AgentText = agent.run(
    "Count: 3 + 5"
)

# Print the result
print(result)

# ╭────────────────────────────────── New run ───────────────────────────────────╮
# │                                                                              │
# │ Count: 3 + 5                                                                 │
# │                                                                              │
# ╰─ InferenceClientModel - Qwen/Qwen2.5-Coder-32B-Instruct ─────────────────────╯
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ─ Executing parsed code: ─────────────────────────────────────────────────────
#   result = sum_ints(a=3, b=5)
#   final_answer(result)
#  ──────────────────────────────────────────────────────────────────────────────
# Out - Final answer: 8
