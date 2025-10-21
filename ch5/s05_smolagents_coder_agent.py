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
    additional_authorized_imports = ["math", "numpy"],
    model = coder_model,
    tools = []  # List of tools
)

result: AgentText = agent.run(
    "Count: 3 + 5"
)

print(result)

# ╭────────────────────────────────── New run ───────────────────────────────────╮
# │                                                                              │
# │ Count: 3 + 5                                                                 │
# │                                                                              │
# ╰─ InferenceClientModel - Qwen/Qwen2.5-Coder-32B-Instruct ─────────────────────╯
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ─ Executing parsed code: ─────────────────────────────────────────────────────
#   result = 3 + 5
#   final_answer(result)
#  ──────────────────────────────────────────────────────────────────────────────
# Out - Final answer: 8
# [Step 1: Duration 4.31 seconds| Input tokens: 2,003 | Output tokens: 54]
# 8
#
# Process finished with exit code 0
