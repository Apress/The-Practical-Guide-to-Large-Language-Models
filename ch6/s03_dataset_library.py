# loading dataset package
from datasets import load_dataset

# Load.
# Loading the dataset from Hugging Face
# The dataset is cached in ~/.cache/huggingface/datasets/
dataset = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    name = "en",
    split = "train"
)

# Also possible to load dataset from local files:
# json: load_dataset("json", data_files="ds.jsonl")
# csv: load_dataset("csv", data_files="ds.csv")
# text: load_dataset("text", data_files="ds.txt")
# pandas (pickled DataFrame): load_dataset("pandas", data_files="ds.pkl")

# Displaying the first 5 samples of the dataset
for i in range(3):
    print(f"Sample {i + 1}")
    print(f"Question: {dataset[i]['Question']}")
    print(f"Complex_CoT: {dataset[i]['Complex_CoT']}")
    print(f"Response: {dataset[i]['Response']}")
    print("=" * 80)

# Split.
# Splitting the dataset into train and test sets
train_test_split = dataset.train_test_split(
    test_size = 0.2,
    seed = 42
)
train_ds = train_test_split["train"]
test_ds = train_test_split["test"]

# Shuffle.
# shuffling the train dataset
train_ds = train_ds.shuffle(seed = 42)

# Filter.
# Filter out samples with too short responses
train_ds = train_ds.filter(lambda s: len(s["Response"]) > 10)


# Map.
# Creating new column based on existing ones
def map_fn(s):
    return {
        "Text":
            f"Question: {s['Question']}\n"
            f"Complex_CoT: {s['Complex_CoT']}\n"
            f"Response: {s['Response']}"
    }


train_ds_merged = train_ds.map(map_fn)
