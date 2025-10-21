from transformers import AutoTokenizer, AutoModelForCausalLM
import wikipedia

# LLM model initialization
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


# Chat function that processes the prompt and generates a response
def chat(prompt, max_new_tokens = 200):
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
        max_new_tokens = max_new_tokens
    )
    # Decoding the output with special tokens
    generated = tokenizer.decode(output[0], skip_special_tokens = False)

    # Extracting the answer from the generated text
    # using special tokens from <|im_start|>assistant to <|im_end|>
    answer = generated.split(
        "<|im_start|>assistant"
    )[-1].replace(
        "<|im_end|>", ""
    ).strip()

    return answer


# Function to search Wikipedia
def wiki_search(query):
    try:
        page = wikipedia.page(query)
        return page.summary
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.page(e.options[0]).summary
    except Exception:
        return "Could not retrieve information from Wikipedia"


# Agent function that decides whether to use Wikipedia or not
def agent(query):
    # Check if Wikipedia is needed
    prompt =\
        f"Question: {query}\n"\
        f"Do you need additional Wikipedia information to answer this question?"\
        f"Answer only yes or no:"

    # Get the decision from the LLM
    router_decision = chat(
        prompt,
        max_new_tokens = 5).lower()

    # If the decision is 'yes', we will use Wikipedia and get the context
    if "yes" in router_decision:
        # If Wikipedia is needed, get the context
        context = wiki_search(query)
        prompt_with_context =\
            f"Context from Wikipedia: {context}\n"\
            f"Question: {query}\n"\
            f"Answer:"
        response = chat(prompt_with_context)
    # If the decision is 'no', we answer directly
    else:
        response = chat(f"Question: {query}\nAnswer:")

    return response.strip()


# Example usage
if __name__ == "__main__":
    query = "What is Alan Turing known for?"
    answer = agent(query)
    print(f"Agent's Answer:\n{answer}")

# Based on the information provided in the context, Alan Turing is best known for his contributions to theoretical
# computer science, particularly through the development of the Turing machine and his work on computational theory.
# Here are some key points about his notable achievements:
#
# 1. **Development of the Turing Machine**: Turing introduced the concept of a universal Turing machine,
# which he described as "the most powerful machine that can compute anything computable." This laid the foundation
# for modern theoretical computer science.
#
# 2. **Theoretical Computer Science**: Turing's work in this field helped establish the principles of algorithmic
# thinking and computational complexity theory.
#
# 3. **Computational Complexity Theory**: Turing's work on algorithms and computation has had a profound impact on
# computer science and artificial intelligence, influencing areas like cryptography, parallel processing,
# and the design of efficient data structures.
