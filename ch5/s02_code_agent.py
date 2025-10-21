import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# LLM model initialization
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


def chat(prompt):
    chat = [
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt = True,
        tokenize = False
    )
    input_ids = tokenizer(
        input_text,
        return_tensors = "pt"
    ).to(model.device).input_ids

    output = model.generate(
        input_ids,
        max_new_tokens = 200
    )
    # Decoding the output with special tokens
    generated = tokenizer.decode(
        output[0],
        skip_special_tokens = False
    )

    # Extracting the answer from the generated text using special tokens
    # from <|im_start|>assistant to <|im_end|>
    answer = generated.split(
        "<|im_start|>assistant"
    )[-1].replace(
        "<|im_end|>", ""
    ).strip()

    return answer


# Tool: Calculator for mathematical expressions
def calculator(expression):
    try:
        return eval(expression)
    except Exception as e:
        return f"Error in calculation: {e}"


# Function to determine if a question requires calculation
def needs_calculation(question):
    keywords = [
        'calculate', 'compute', '+', '-', '*', '/',
        'sum', 'difference', 'product', 'quotient'
    ]
    return any(kw in question.lower() for kw in keywords)


# Main agent function that decides whether to use the calculator or LLM
def agent(question):
    if needs_calculation(question):
        # Extracting the mathematical expression from the question
        expression = re.findall(r'[\d\s\+\-\*/\.]+', question)
        expression = expression[0] if expression else ""

        if expression:
            result = calculator(expression)
            return f"Calculation result: {result}"
        else:
            return "I can't find any expression to calculate."
    else:
        # Using LLM for response generation
        response = chat(question)
        return response.strip()


questions = [
    "What is the capital of France?",
    "Calculate 12 * 8 - 5"
]

for q in questions:
    print(f"Question: {q}\nAnswer: {agent(q)}\n")
