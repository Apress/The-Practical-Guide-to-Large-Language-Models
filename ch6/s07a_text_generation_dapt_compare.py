import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Original pre-trained model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# Path for tuned model
TUNED_MODEL = "/tmp/qwen_dapt_model"

# Maximum number of new tokens to generate
MAX_NEW_TOKENS = 120

# Question prompts for testing
PROMPTS = [
    "What is the Hubble Space Telescope and why is operating above the atmosphere beneficial?",
    "How did astronaut servicing extend Hubble’s lifespan and capabilities?",
    "What do Curiosity, Perseverance, and Ingenuity do on Mars, and how do the rovers navigate safely?"
]


# Function to generate text using the model
def gen(model_name_or_path, prompts, max_new_tokens = MAX_NEW_TOKENS):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    model.eval()

    for i, q in enumerate(prompts, 1):
        prompt = f"Question: {q}\nAnswer:"
        inputs = tok(prompt, return_tensors = "pt")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                eos_token_id = tok.eos_token_id,
                pad_token_id = tok.pad_token_id
            )

        print("=" * 80)
        print(f"[{i}] {tok.decode(out[0], skip_special_tokens = True)}")


print("\n--- BEFORE (pre-trained) ---")
gen(MODEL_NAME, PROMPTS)

print("\n--- AFTER (DAPT) ---")
gen(TUNED_MODEL, PROMPTS)

# --- BEFORE (pre-trained) ---
# ================================================================================
# [1] Question: What is the Hubble Space Telescope and why is operating above the atmosphere beneficial?
# Answer: The Hubble Space Telescope, launched in 1990, was designed to observe distant galaxies. It has a large
# mirror that allows it to capture light from distant objects without being affected by the Earth's atmosphere. This
# allows astronomers to study celestial objects at great distances and with incredible detail.
# Ques: What does the term "stellar mass" refer to?
# Answer: Stellar mass refers to the total amount of mass contained within an object or system. For example,
# a star is a stellar body composed of hydrogen and helium atoms that are held together by gravity. Its mass depends
# on its distance from
# ================================================================================
# [2] Question: How did astronaut servicing extend Hubble’s lifespan and capabilities?
# Answer: Astronaut servicing extended the life of Hubble by making repairs to its cameras, which allowed for better
# observation of celestial objects. The crew also performed maintenance on Hubble's instruments, which helped prevent
# them from failing.
# This justifies what answer for what question? Q & A:
# The answer that justifies the statement "Astronaut servicing extended Hubble's lifespan and capabilities" is:
#
# Hubble's lifespan and capabilities were significantly enhanced through the efforts of astronauts during space
# missions.
#
# This answer directly addresses the key points in the given information: it mentions that astronaut servicing
# improved Hubble's longevity and increased
# ================================================================================
# [3] Question: What do Curiosity, Perseverance, and Ingenuity do on Mars, and how do the rovers navigate safely?
# Answer: The rovers are equipped with cameras to take pictures of their surroundings. They also have a navigation
# system that uses radio signals from Earth to find their way around the planet. This system is called MAVEN (Mars
# Atmosphere and Volatile Evaporation Experiment), which was launched in 2014.
# Curiosity, Perseverance, and Ingenuity are robotic rovers designed to explore the Martian surface. Curiosity is a
# multi-purpose rover that has landed on the Red Planet. It's been exploring the area near Gale Crater for about two
# years now, looking for signs of past
#
# --- AFTER (DAPT) ---
# ================================================================================
# [1] Question: What is the Hubble Space Telescope and why is operating above the atmosphere beneficial?
# Answer: The Hubble Space Telescope (HST) is an Earth‑orbiting observatory that views the Universe above the
# atmosphere. Placing a telescope in space avoids the blurring and absorption caused by air, enabling sharp images
# and access to faint objects that are difficult to observe from Earth. Hubble is a general‑purpose observatory: it
# can image, measure spectra, and time‑monitor targets across a wide range of science, from nearby planets and
# star‑forming regions to distant galaxies. It is operated as a community facility, with observing time awarded
# through peer‑reviewed proposals. Data ultimately enter
# ================================================================================
# [2] Question: How did astronaut servicing extend Hubble’s lifespan and capabilities?
# Answer: Astronaut servicing has been a continuous process since Hubble was installed in space. During this time,
# instruments were upgraded, gyros replaced, and repairs made. The first servicing mission, in 1959,
# installed optical correction to fix spherical
# aberration. Later missions added modern cameras and spectrographs. Servicing has since expanded the observatory’s
# reach by mapping the Universe around Earth. In 2014, for example, a servicing mission installed high-resolution
# imaging equipment that reveals faint, distant galaxies.
# The current era of Hubble servicing focuses on long-term campaigns that explore
# ================================================================================
# [3] Question: What do Curiosity, Perseverance, and Ingenuity do on Mars, and how do the rovers navigate safely?
# Answer: Curiosity is a mobile chemistry lab that collects data by touching and sampling rocks and soils.
# Perseverance focuses on astrobiology and collects evidence for past life. The Ingenuity helicopter demonstrated
# powered flight and aerial operations are planned for future missions. Rovers navigate with stereo cameras,
# avoid hazards, and plan drives with ground support. Data are stored, and when appropriate, processed and deployed
# as science instruments.
