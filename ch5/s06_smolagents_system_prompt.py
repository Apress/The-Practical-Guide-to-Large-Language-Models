from smolagents import CodeAgent, TransformersModel, tool, AgentText, InferenceClientModel

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

agent = CodeAgent(
    model = coder_model,
    tools = []
)

# Get the system prompt of the agent
prompt = agent.system_prompt

# Print the system prompt
print(prompt)
# You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you
# can.
# To do so, you have been given access to a list of tools: these tools are basically Python functions which you can
# call with code.
# To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:',
# and 'Observation:' sequences.
# ...

# You can modify the system prompt by adding additional text to it.
agent.prompt_templates["system_prompt"] =\
    prompt + "\n\n" +\
    "Some additional text to the system prompt."
