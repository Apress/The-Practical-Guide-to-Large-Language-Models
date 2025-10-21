from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
import json


# Film Schema
class Film(BaseModel):
    name: str
    year: int


# Loading the model wrapped by Outlines
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map = "auto"),
    AutoTokenizer.from_pretrained(MODEL_ID),
)

# Film description
film_description = """
Dumb and Dumber is a 1994 American buddy comedy film directed by Peter Farrelly,[1][2] who cowrote the screenplay 
with Bobby Farrelly and Bennett Yellin. It is the first installment in the Dumb and Dumber franchise. Starring Jim 
Carrey and Jeff Daniels, it tells the story of Lloyd Christmas (Carrey) and Harry Dunne (Daniels), two dumb but 
well-meaning friends from Providence, Rhode Island, who set out on a cross-country road trip to Aspen, Colorado, 
to return a briefcase full of money to its owner, thinking it was abandoned as a mistake, though it was actually left 
as a ransom. Lauren Holly, Karen Duffy, Mike Starr, Charles Rocket, and Teri Garr play supporting roles.
"""

prompt = (
        "Read the following film description and extract structured JSON "
        "that matches the Film model:\n\n" + film_description
)

# Generating raw output
raw = model(prompt, Film, max_new_tokens = 128)

# Validating and printing the structured output
film = Film.model_validate_json(raw)
print(
    json.dumps(
        film.model_dump(),
        indent = 2,
        ensure_ascii = False
    )
)
# {
#   "name": "Dumb and Dumber",
#   "year": 1994
# }
