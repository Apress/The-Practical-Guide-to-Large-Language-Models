from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "What happens to you if you eat watermelon seeds?"
input_ids = tokenizer(input_text, return_tensors = "pt").input_ids

# Greedy Sampling
output_ids = model.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample = False
)
print("Greedy Sampling Output:")
print(tokenizer.decode(output_ids[0], skip_special_tokens = True))
# Eating watermelon seeds is generally safe for most people. The seeds are not toxic and do not pose a health risk.
# However, they are hard and can potentially cause digestive discomfort

# Top-k Sampling
output_ids = model.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample = True,
    top_k = 50
)
print("Top-k Sampling Output:")
print(tokenizer.decode(output_ids[0], skip_special_tokens = True))
# Eating watermelon seeds is generally not harmful, and nothing dangerous happens. The seeds are actually edible,
# and they have a pleasant crunch when eaten raw. In fact

# Top-p Sampling (Nucleus Sampling)
output_ids = model.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample = True,
    top_p = 0.9
)
print("Top-p Sampling Output:")
print(tokenizer.decode(output_ids[0], skip_special_tokens = True))
# Watermelon seeds are rich in healthy fats, protein, fiber, vitamins, and minerals. While eating them in moderation
# is unlikely to cause any adverse effects, consuming a

# Temperature Sampling
output_ids = model.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample = True,
    temperature = 0.7
)
print("Temperature Sampling Output:")
print(tokenizer.decode(output_ids[0], skip_special_tokens = True))
# I've heard this claim many times, but I'm not sure if it's true.
# Google says this isn't true, but I can't find a source backing it up.
