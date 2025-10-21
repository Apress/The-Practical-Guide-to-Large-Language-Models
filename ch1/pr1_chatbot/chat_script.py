import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define constants
# Model size: 1GB
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# Maximum number of chat history entries to include in the context
MAX_CHAT_HISTORY = 5
# Maximum number of tokens to generate in the response
MAX_NEW_TOKENS = 300

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype = "auto",
    device_map = "auto"
).to(device)

# Initialize chat history
chat_history = []


# Generate response based on user prompt
def generate_response(prompt):
    global chat_history

    # Add a system message to define the assistant
    messages = [
        {
            "role":    "system",
            "content": "You are a helpful assistant."
        }
    ]

    # Add the chat history to the context
    # Include only the last 5 entries in the chat history
    for entry in chat_history[-MAX_CHAT_HISTORY:]:
        role, content = entry.split(": ", 1)
        messages.append({
            "role":    role,
            "content": content
        })

    # Add the last user prompt
    messages.append({"role": "user", "content": prompt})

    # unify the chat messages into a standardized text format for the model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )

    # tokenize the text
    model_inputs = tokenizer([text], return_tensors = "pt").to(device)

    # generate the response
    outputs = model.generate(
        **model_inputs,
        max_new_tokens = MAX_NEW_TOKENS,
        pad_token_id = tokenizer.eos_token_id
    )
    generated_ids = outputs[:, model_inputs.input_ids.shape[-1]:]

    # Decode the response
    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens = True
    )[0]

    # Append request and response to chat history
    chat_history.append(f"user: {prompt}")
    chat_history.append(f"assistant: {response}")

    return response


if __name__ == "__main__":
    prompt_1 = "What should I buy at the store to make pancakes? Just make a list of ingredients."
    response_1 = generate_response(prompt_1)
    print(response_1)
    # Sure! Here's a simple recipe for pancakes that you can use as a starting point:
    #
    # ### Ingredients:
    # - 2 cups all-purpose flour
    # - 1 cup granulated sugar
    # - 3/4 cup unsalted butter, cold and cut into small pieces
    # ...
    # Serve hot and enjoy your delicious homemade pancakes!
    print('------------------------------------')

    prompt_2 = "I'd like to make it more interesting. Can you suggest a unique ingredient?"
    response_2 = generate_response(prompt_2)
    print(response_2)
    # Absolutely! A great way to spice up pancakes is by adding a unique ingredient such as:
    #
    # ### Unique Ingredient:
    # - **Pineapple Seeds**: These seeds add a tropical twist to pancakes and give them a fun and unique flavor profile.
    #
    # Here’s how you can incorporate pineapple seeds into your pancake recipe:
    # ...
    # Enjoy your unique and tasty pancake creation!
    print('------------------------------------')

    prompt_3 = "What is the best way to serve it?"
    response_3 = generate_response(prompt_3)
    print(response_3)
    # When serving your pancakes, consider the following tips to ensure they are enjoyable and memorable:
    #
    # ### Serving Suggestions:
    # 1. **Choose the Right Pantry**: Ensure that the temperature of the pan is around 350°F (175°C) when cooking the
    # pancakes. This ensures that the pancakes cook quickly without burning.
    #
    # 2. **Heat Properly**: If using a skillet, heat it over medium-high heat first. This helps prevent the pancakes
    # from sticking and ensures they cook evenly.
    # ...
    # By following these suggestions, you can create a perfect dish that satisfies both your taste buds and those
    # around you. Enjoy your pancakes!
    print('------------------------------------')
