from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import torch


# Custom stopping criteria to stop generation when a specific punctuation token is encountered
class StopOnPunctuationCriteria(StoppingCriteria):

    def __init__(self, tokenizer, stop_tokens: list):
        self.tokenizer = tokenizer
        # Converting stop tokens to their corresponding IDs
        self.stop_token_ids = [
            tokenizer.encode(token, add_special_tokens = False)[0]
            for token in stop_tokens
        ]

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Check if the last token in the input_ids is one of the stop tokens
        last_token_id = input_ids[0, -1].item()
        return last_token_id in self.stop_token_ids


# Loading the model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text for generation
input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors = "pt").input_ids

# Creating custom stopping criteria
custom_stopping = [
    StopOnPunctuationCriteria(tokenizer, stop_tokens = [".", "!"])
]

# Generating text with custom stopping criteria
output_ids = model.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample = True,
    temperature = 0.9,
    stopping_criteria = custom_stopping
)

# Decoding and printing the generated text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens = True)
print("Generated text:")
print(output_text)
# Once upon a time in the town of Digital, there was a peculiar little shop named "Byte's Boutique," where the
# peculiarities of the gadgets sold were only rivaled by the stories they brought into the lives of their owners.
