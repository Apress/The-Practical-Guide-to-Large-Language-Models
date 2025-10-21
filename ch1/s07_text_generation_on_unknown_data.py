from transformers import pipeline

generator = pipeline(
    "text-generation",
    # Model size: 954M
    model = "Qwen/Qwen2.5-0.5B-Instruct"
)

request = "When the action of the movie Alien: Romulus takes place?"

prompt = f"User: {request}\nQwen:"

response_list = generator(
    prompt,
    max_new_tokens = 200,
    num_return_sequences = 2
)

for i in range(len(response_list)):
    r = response_list[i]
    generated_text = r["generated_text"]
    response = generated_text.split("\nQwen:")[1]
    print(f"Response {i + 1}:")
    print(response)
    print('====================')

# Response 1:
#  To answer this question, we need to first understand what the movie Alien: Romulus is about.
#
# Alien: Romulus is a 2019 American science fiction horror film directed by Stephen Spielberg and produced by Steven
# Spielberg's company, DreamWorks Animation. It stars Tom Hardy as the main character, who is a former alien soldier
# turned human.
#
# Now that we know the basic plot of the movie, let us focus on the action scene where Tom Hardy plays the role of a
# scientist named Dr. Marcus, who discovers a mysterious artifact in the remote village of Romulus. The movie starts
# with Dr. Marcus discovering the artifact during his experiments and begins to investigate its origins. As he delves
# deeper into the story, the movie shows how different cultures interact with each other, leading to a series of
# unexpected twists and turns.
#
# So, when the action of the movie "Alien: Romulus"
# ====================
# Response 2:
#  The movie Alien: Romulus is a science fiction horror film directed by Robert Zemeckis and stars Leonardo DiCaprio,
#  Carrie-Anne Moss, and Tom Hardy. It was released on November 26, 2019. In the movie, an alien named Romulus
#  arrives on Earth to destroy humanity, but he encounters a group of survivors who have been trapped in a space
#  station called Romulus. They must work together to save them from Romulus's destruction. During the action
#  sequence of the movie, it takes place in a space station called Romulus. The movie features a variety of special
#  effects, including a flying saucer, a rocket ship, and a giant robot. However, the exact location where the action
#  takes place is not specified in the provided information. Therefore, the answer is that the action of the movie
#  Alien: Romulus takes place in a space station called Rom
# ====================
