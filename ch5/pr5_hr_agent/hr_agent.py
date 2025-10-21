# Importing necessary libraries
import os
from datetime import datetime
from typing import List
from smolagents import CodeAgent, tool, InferenceClientModel
from huggingface_hub import InferenceClient

# Replace with your Hugging Face token:
# https://huggingface.co/settings/tokens
hf_token = "your_token_here"


# Next goes tool for extracting CVs from a CSV file.
# This tool reads a CSV file and returns a list of CVs as strings.
@tool
def cv_list_tool(csv_file_name: str) -> List[str]:
    """
    Returns a list of CVs from a CSV file line by line.
    Example:
        1,Michael Scott,46,"Senior product manager with 10+ years of experience leading...
        2,Dwight Schrute,35,"Assistant to the regional manager with a strong background...

    Args:
        csv_file_name (str): The name of the CSV file containing CVs.
    """
    import pandas as pd

    df = pd.read_csv(csv_file_name)
    cv_list = df.to_dict(orient = 'records')
    response = [str(cv) for cv in cv_list]
    return response


# Here we define CV match LLM id for making decisions on CV matching
cv_match_model = 'meta-llama/Llama-3.1-8B-Instruct'

# We will use CV match model via Inference API
hf_inference_client = InferenceClient(api_key = hf_token)


# Next goes CV match tool to check how closely a CV matches a requirement.
# This tool uses the CV match model to return an integer from 0 to 5.
@tool
def cv_match_tool(requirement: str, cv_text: str) -> int:
    """
    Returns integer from 0 to 5 if the CV matches the requirement.
    0 means no match, 5 means perfect match.
    Example:
        "
        Requirement: Senior product manager with 5+ years of experience.
        CV: Michael Scott, 46, Senior product manager with 10+ years of experience leading...
        return 4
        "

    Args:
        requirement (str): The requirement for the candidate
        cv_text (str): The CV text of the candidate
    """
    prompt = f"""
You are an HR specialist. Carefully analyze the following candidate CV and decide if it meets the requirement below. 

0 means no match
1 means very weak match
2 means weak match
3 means moderate match
4 means strong match
5 means perfect match
 
Return an integer from 0 to 5.

Requirement: {requirement}

CV: {cv_text}

Does the CV match the Requirement? Answer only integer from 0 to 5.
    """

    response = hf_inference_client.chat.completions.create(
        model = cv_match_model,
        messages = [
            {
                "role":    "user",
                "content": prompt
            }
        ],
        max_tokens = 3,
        temperature = 0.0
    ).choices[0].message.content

    try:
        # removing input prompt from the response
        response = response.split(prompt)[-1].strip()
        # leaving only integer in the response
        response_int = int(''.join(filter(str.isdigit, response)))
        current_time_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time_str}] CV match tool response: {response_int}")
        return min(max(response_int, 0), 5)
    except ValueError:
        return 0


# Then we define the LLM model for the HR agent.
# This Coder Model is used remotely via Inference API too.
coder_model = InferenceClientModel(
    model_id = 'Qwen/Qwen2.5-Coder-32B-Instruct',
    api_key = hf_token
)

# Defining the HR Agent as smolagents CodeAgent
hr_agent = CodeAgent(
    provide_run_summary = True,
    model = coder_model,
    tools = [
        cv_list_tool,
        cv_match_tool
    ],
    # Number of steps before planning
    planning_interval = 1,
)

# Here is the task and candidate requirement:
# Machine Learning Engineer
task = "Find top 5 candidates who match the requirement"
candidate_requirement = "Machine Learning Engineer"

# Finance manager
# task = "Find a candidate CV whose name contains 'Thomas' and matches the requirement"
# candidate_requirement = "Program manager"

# We modify the system prompt of the HR agent:
prompt = hr_agent.system_prompt
prompt_add = "Create a list of candidates who match the requirement. \n"\
             "Use passed cv_extractor_tool function on final stage: "\
             "Like this: final_answer(response) \n"\
             "Don't use any 'import csv' or other import statements in your code! \n"\
             "Use final_answer(...) function to return the final answer. \n"

# Modifying the system prompt of the agent
hr_agent.prompt_templates["system_prompt"] =\
    prompt + f"\n {prompt_add}"

current_file_path = os.path.dirname(os.path.abspath(__file__))

# Running the HR manager agent
result = hr_agent.run(
    task,
    additional_args = {
        "developer cvs": f"{current_file_path}/data/developers_cvs.csv",
        "managers cvs":  f"{current_file_path}/data/managers_cvs.csv",
        "requirement":   candidate_requirement
    }
)

# Printing the result
print("Result:")
print(result)

# Machine Learning Engineer requirement example output:
# [
#   "{'id': 5, 'name': 'Noah Davis', 'age': 44, 'CV': 'Machine learning engineer with hands-on experience in model
#   development, deployment, and monitoring. Familiar with TensorFlow, PyTorch, and MLOps practices.'}",
#   "{'id': 13, 'name': 'William Lee', 'age': 27, 'CV': 'NLP engineer focused on retrieval-augmented generation and
#   prompt optimization. Skilled with Hugging Face, vector databases, and evaluation suites.'}",
#   "{'id': 14, 'name': 'Charlotte Walker', 'age': 32, 'CV': 'Data engineer proficient in Apache Spark, Airflow,
#   and dbt. Builds reliable ETL pipelines and data models for analytics and ML.'}",
#   "{'id': 19, 'name': 'Evelyn Scott', 'age': 41, 'CV': 'Applied ML scientist with experience in recommendation
#   systems and experimentation. Comfortable with PyTorch, offline evaluation, and A/B testing.'}",
#   "{'id': 30, 'name': 'David Mitchell', 'age': 28, 'CV': 'MLOps engineer implementing model serving and monitoring.
#   Uses FastAPI, BentoML, and Grafana to keep models healthy in production.'}"
# ]
