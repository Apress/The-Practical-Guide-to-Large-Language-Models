from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

# Loading the model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Custom logits processor to ban specific tokens
class BanTokensProcessor(LogitsProcessor):

    def __init__(self, banned_token_ids):
        self.banned_token_ids = set(banned_token_ids)

    def __call__(self, input_ids, scores):
        # Set the scores of banned tokens to a very low value
        for token_id in self.banned_token_ids:
            scores[:, token_id] = -float("inf")
        return scores


# Input text for generation
input_text = "This London is a capital"
input_ids = tokenizer(input_text, return_tensors = "pt").input_ids

# Example usage of the custom logits processor to ban specific words
banned_words = ["Great Britain", "England", "United Kingdom"]
banned_token_ids = [tokenizer.encode(w, add_special_tokens = False)[0] for w in banned_words]

output_ids = model.generate(
    input_ids,
    max_new_tokens = 30,
    do_sample = False,
    logits_processor = [BanTokensProcessor(banned_token_ids)]
)

print(tokenizer.decode(output_ids[0], skip_special_tokens = True))
# This London is a capital city, and it's the largest city in the UK. It's known for its rich history,
# diverse culture, and iconic landmarks
