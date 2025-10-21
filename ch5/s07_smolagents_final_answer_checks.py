from typing import Any

from smolagents import CodeAgent, TransformersModel, InferenceClientModel

# Local model
# coder_model = TransformersModel(
#     model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# )

# Inference Client model
hf_token = 'your_token'
coder_model = InferenceClientModel(
    model_id = 'Qwen/Qwen2.5-Coder-32B-Instruct',
    api_key = hf_token
)


# check if the final answer is an integer
def is_integer(final_answer: Any, agent_memory = None) -> bool:
    final_answer = str(final_answer)
    try:
        int(final_answer.strip())
        return True
    except ValueError:
        return False


# check if the final answer is greater than zero
def is_greater_than_zero(final_answer: Any, agent_memory = None) -> bool:
    final_answer = str(final_answer)
    try:
        value = int(final_answer.strip())
        return value > 0
    except ValueError:
        return False


# Now we can create the CodeAgent with the custom final answer checks
agent = CodeAgent(
    model = coder_model,
    tools = [],
    # Define final answer checks
    # if the final answer does not pass these checks,
    # an error will be raised
    # and the agent will try to solve the task again
    # using different code
    final_answer_checks = [
        is_integer,
        is_greater_than_zero
    ]
)

result = agent.run(
    "Count: 3 + 5"
)

print(result)
