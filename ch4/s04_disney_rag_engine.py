import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# 0: Initialization
# Text splitting function to create fixed-size chunks with overlap
def split_text(text, size = 500, overlap = 100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += size - overlap  # Move the start index forward, considering overlap
        if end >= len(words):
            break
    return chunks


# Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 8-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0,
    llm_int8_skip_modules = None,
    llm_int8_enable_fp32_cpu_offload = False
)

# LLM
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = "cuda"  # Use GPU if available
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. Load
# Articles about Disney Movies
doc_urls = [
    "https://en.wikipedia.org/wiki/Snow_White_and_the_Seven_Dwarfs",
    "https://en.wikipedia.org/wiki/Pinocchio_(1940_film)",
    "https://en.wikipedia.org/wiki/Fantasia_(1940_film)",
    "https://en.wikipedia.org/wiki/Dumbo_(film)",
    "https://en.wikipedia.org/wiki/Bambi",
    "https://en.wikipedia.org/wiki/Saludos_Amigos",
    "https://en.wikipedia.org/wiki/The_Three_Caballeros",
    "https://en.wikipedia.org/wiki/Make_Mine_Music",
    "https://en.wikipedia.org/wiki/Fun_and_Fancy_Free",
    "https://en.wikipedia.org/wiki/Melody_Time",
    "https://en.wikipedia.org/wiki/The_Adventures_of_Ichabod_and_Mr._Toad",
    "https://en.wikipedia.org/wiki/Cinderella_(1950_film)",
    "https://en.wikipedia.org/wiki/Alice_in_Wonderland_(1951_film)",
    "https://en.wikipedia.org/wiki/Peter_Pan_(1953_film)",
    "https://en.wikipedia.org/wiki/Lady_and_the_Tramp",
    "https://en.wikipedia.org/wiki/Sleeping_Beauty_(1959_film)",
    "https://en.wikipedia.org/wiki/The_Little_Mermaid_(1989_film)",
    "https://en.wikipedia.org/wiki/Beauty_and_the_Beast_(1991_film)",
    "https://en.wikipedia.org/wiki/Aladdin_(1992_film)",
    "https://en.wikipedia.org/wiki/The_Lion_King",
    "https://en.wikipedia.org/wiki/Pocahontas_(film)",
    "https://en.wikipedia.org/wiki/The_Hunchback_of_Notre_Dame_(1996_film)",
    "https://en.wikipedia.org/wiki/Hercules_(1997_film)",
    "https://en.wikipedia.org/wiki/Mulan_(film)",
    "https://en.wikipedia.org/wiki/Tarzan_(1999_film)",
    "https://en.wikipedia.org/wiki/The_Emperor%27s_New_Groove",
    "https://en.wikipedia.org/wiki/Lilo_%26_Stitch",
    "https://en.wikipedia.org/wiki/Brother_Bear",
    "https://en.wikipedia.org/wiki/The_Princess_and_the_Frog",
    "https://en.wikipedia.org/wiki/Tangled",
    "https://en.wikipedia.org/wiki/Winnie_the_Pooh_(2011_film)",
    "https://en.wikipedia.org/wiki/Wreck-It_Ralph",
    "https://en.wikipedia.org/wiki/Frozen_(2013_film)",
    "https://en.wikipedia.org/wiki/Big_Hero_6",
    "https://en.wikipedia.org/wiki/Zootopia",
    "https://en.wikipedia.org/wiki/Moana",
    "https://en.wikipedia.org/wiki/Ralph_Breaks_the_Internet",
    "https://en.wikipedia.org/wiki/Frozen_II",
    "https://en.wikipedia.org/wiki/Raya_and_the_Last_Dragon",
    "https://en.wikipedia.org/wiki/Encanto",
    "https://en.wikipedia.org/wiki/Strange_World",
    "https://en.wikipedia.org/wiki/Wish_(2023_film)",
    "https://en.wikipedia.org/wiki/Moana_2",
    "https://en.wikipedia.org/wiki/The_Return_of_Jafar",
    "https://en.wikipedia.org/wiki/Aladdin_and_the_King_of_Thieves",
    "https://en.wikipedia.org/wiki/Beauty_and_the_Beast:_The_Enchanted_Christmas",
    "https://en.wikipedia.org/wiki/Belle%27s_Magical_World",
    "https://en.wikipedia.org/wiki/Pocahontas_II:_Journey_to_a_New_World",
    "https://en.wikipedia.org/wiki/The_Lion_King_II:_Simba%27s_Pride",
    "https://en.wikipedia.org/wiki/Cinderella_II:_Dreams_Come_True",
    "https://en.wikipedia.org/wiki/Cinderella_III:_A_Twist_in_Time",
    "https://en.wikipedia.org/wiki/Bambi_II",
    "https://en.wikipedia.org/wiki/Brother_Bear_2",
    "https://en.wikipedia.org/wiki/The_Fox_and_the_Hound_2",
    "https://en.wikipedia.org/wiki/Pooh%27s_Grand_Adventure:_The_Search_for_Christopher_Robin",
    "https://en.wikipedia.org/wiki/Lilo_%26_Stitch_2:_Stitch_Has_a_Glitch",
    "https://en.wikipedia.org/wiki/Kronk%27s_New_Groove",
    "https://en.wikipedia.org/wiki/Tinker_Bell_(film)",
]
documents = []
for i, doc_url in enumerate(doc_urls, start = 1):
    print(f"Loading document {i}/{len(doc_urls)}: {doc_url}")
    response = requests.get(doc_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    documents.append(text)

# 2. Split into chunks
chunk_store = {}
chunk_id = 0
for text in documents:
    chunks = split_text(text, size = 500, overlap = 100)
    for chunk in chunks:
        chunk_id += 1
        chunk_store[chunk_id] = chunk

# 3. Embeddings
vector_store = {}
for chunk_id, chunk in chunk_store.items():
    embedding = embedding_model.encode(
        chunk,
        convert_to_tensor = True
    )
    vector_store[chunk_id] = embedding

# 4. Store
# Not implemented

# 5. Retrieve
question = "What is the name of the main character in the movie 'The Lion King'?"

# Question embedding
question_embedding = embedding_model.encode(question, convert_to_tensor = True)

# Comparing the question embedding with each chunk embedding
similarities = {}
for chunk_id, embedding in vector_store.items():
    sim = util.cos_sim(question_embedding, embedding)
    similarities[chunk_id] = sim.item()

# Sorting the chunks based on similarity scores
sorted_chunks = sorted(
    similarities.items(),
    key = lambda x: x[1],
    reverse = True
)

# Selecting the top N chunks according to the similarity scores and max_chunk_num value
# Maximum number of chunks for the context
# 5 * 500 = 2500 ~ tokens
max_chunk_num = 5
context_chunks = []
for chunk_id, score in sorted_chunks[:max_chunk_num]:
    context_chunks.append(chunk_store[chunk_id])

# 6. Prompt
context = "\n\n".join(context_chunks)
prompt =\
    f"Use the context below to answer the question. \n"\
    f"Context: {context} \n"\
    f"Question: {question} \n"\
    f"Assistant:"

# 7. Generate
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
    max_new_tokens = 200,
    do_sample = False
)
# Decoding the output with special tokens
generated = tokenizer.decode(output[0], skip_special_tokens = False)

# Extracting the answer from the generated text using special tokens
# from <|im_start|>assistant to <|im_end|>
answer = generated.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()

print("Prompt:")
print(prompt)

print("=" * 50)

print("Answer:")
print(answer)

# Prompt:
# Use the context below to answer the question.
# Context: The Lion King is a 1994 American animated musical coming-of-age drama film[3][4] directed by Roger Allers
# and Rob Minkoff, produced by Don Hahn, and written by Irene Mecchi, Jonathan Roberts, and Linda Woolverton.
# Produced by Walt Disney Feature Animation and released by Buena Vista Pictures Distribution under Walt Disney
# Pictures, the film features an ensemble voice cast consisting of Matthew Broderick, James Earl Jones, Jeremy Irons,
# Jonathan Taylor Thomas, Moira Kelly, Niketa Calame, Nathan Lane, Ernie Sabella, Whoopi Goldberg, Cheech Marin,
# Rowan Atkinson, and Robert Guillaume. The film follows a young lion, Simba, who flees his kingdom when his father,
# King Mufasa, is murdered by his uncle, Scar. After growing up in exile, Simba returns home to confront his uncle
# and reclaim his throne. The Lion King was conceived during conversations among various Disney executives,
# to whom several writers submitted early treatments. Original director George Scribner had envisioned The Lion King
# as a nature documentary-style film, with Allers joining as co-director after having worked in the story departments
# of several successful animated Disney films. Considered to be Disney's first original animated film,
# The Lion King's plot draws inspiration from several sources, notably William Shakespeare's play Hamlet. Woolverton,
# screenwriter for Disney's Beauty and the Beast (1991), drafted early versions of The Lion King's script,
# which Mecchi and Roberts were hired to revise once Woolverton left to prioritize other projects. Scribner departed
# due to disagreements over the studio's decision to reimagine the film as a musical, with original songs by Elton
# John and Tim Rice, and Minkoff was hired to replace him in April 1992. Throughout production, the creative team
# visited Kenya for research and inspiration. Released on June 15, 1994, The Lion King was praised by critics for its
# music, story, themes, and animation. With an initial worldwide gross of $763 million, it completed its theatrical
# run as the highest-grossing film of 1994 and the second-highest-grossing film of all time, behind Jurassic Park (
# 1993). It held the title of highest-grossing animated film until it was replaced by Finding Nemo in 2003. The film
# remains the highest-grossing traditionally animated film of all time, as well as the best-selling film on home
# video, having sold over 55 million copies worldwide. It won two Academy Awards, as well as the Golden Globe Award
# for Best Motion Picture – Musical or Comedy. It is considered by many to be among the greatest animated films ever
# made. The success of the film launched a multibillion-dollar franchise comprising a Broadway adaptation,
# two direct-to-video follow-ups, two television series, and a photorealistic remake (which itself spawned a
# prequel), which in 2019 also became the highest-grossing animated film at the time of its release. In 2016,
# The Lion King was selected for preservation in the United States National Film Registry by the Library of Congress
# as being "culturally, historically, or aesthetically significant". In the Pride Lands of Tanzania, a pride of lions
# rules over the kingdom from Pride Rock. King Mufasa and Queen Sarabi's newborn son, Simba, is presented to
#
# made. The success of the film launched a multibillion-dollar franchise comprising a Broadway adaptation,
# two direct-to-video follow-ups, two television series, and a photorealistic remake (which itself spawned a
# prequel), which in 2019 also became the highest-grossing animated film at the time of its release. In 2016,
# The Lion King was selected for preservation in the United States National Film Registry by the Library of Congress
# as being "culturally, historically, or aesthetically significant". In the Pride Lands of Tanzania, a pride of lions
# rules over the kingdom from Pride Rock. King Mufasa and Queen Sarabi's newborn son, Simba, is presented to the
# gathered animals by Rafiki, the mandrill who serves as the kingdom's shaman and advisor. However, Mufasa's younger
# brother, Scar, covets the throne and secretly plots to eliminate both Mufasa and Simba so that he may become king.
# When Simba grows into a young cub, Mufasa shows him the Pride Lands and explains to Simba the responsibilities of
# kingship and the "circle of life," which connects all living things. But then Scar manipulates Simba into exploring
# an elephants' graveyard beyond the Pride Lands, which Mufasa forbade him to do so, stating that only the bravest
# lions go there. Wanting to prove his courage, Simba escapes Mufasa's majordomo, the hornbill Zazu and sneaks into
# the elephants' graveyard along with his best friend, Nala, though Zazu catches up to them. Upon arriving, however,
# the trio is chased by three spotted hyenas named Shenzi, Banzai, and Ed. Zazu manages to alert Mufasa,
# and he arrives to chase the hyenas away. Though disappointed in Simba for disobeying him and endangering himself
# and Nala, Mufasa forgives him while teaching him the importance of bravery. He then explains that the great kings
# of the past watch over them from the night sky, from which he will one day watch over Simba. Meanwhile, Scar,
# who orchestrated the hyenas' attack, convinces them to help him murder Mufasa and Simba in exchange for hunting
# rights in the Pride Lands. The next day, Scar lures Simba into a gorge and signals the hyenas to drive a large herd
# of wildebeests into a stampede to trample him. Scar alerts Mufasa, who manages to save Simba and tries to escape
# the gorge; he begs Scar for help, but Scar betrays him by throwing him into the stampede to his death. Scar then
# deceives Simba into believing that Mufasa's death was his fault and tells him to leave the Pride Lands and never
# return. He then orders the hyenas to kill Simba, but Simba escapes, and they decide not to tell him of it. Unaware
# of Simba's survival, Scar tells the pride that both Mufasa and Simba are dead, and steps forward as the new king,
# allowing the hyenas into the Pride Lands, much to the shock of the pride and the sadness of Rafiki. Simba collapses
# in a desert, but is rescued by two other outcasts; a meerkat and a warthog named Timon and Pumbaa. Simba grows up
# with his two new
#
# the first film, similar to The Godfather Part II. Jeff Nathanson, the screenwriter for the remake, has reportedly
# finished a draft.[267][268] In August 2021, it was reported that Aaron Pierre and Kelvin Harrison Jr. had been cast
# as Mufasa and Scar respectively.[269] The film will not be a remake of The Lion King II: Simba's Pride,
# the 1998 direct-to-video sequel to the original animated film.[270] In September 2022 at the D23 Expo,
# it was announced that the film will be titled Mufasa: The Lion King and it will follow the titular character's
# origin story. Seth Rogen, Billy Eichner, and John Kani will reprise their roles as Pumbaa, Timon, and Rafiki,
# respectively. The film was released on December 20, 2024.[271] Along with the film release, three different video
# games based on The Lion King were released by Virgin Interactive in December 1994. The main title was developed by
# Westwood Studios, and published for PC and Amiga computers and the consoles SNES and Sega Mega Drive/Genesis. Dark
# Technologies created the Game Boy version, while Syrox Developments handled the Master System and Game Gear
# version.[272] The film and sequel Simba's Pride later inspired another game, Torus Games' The Lion King: Simba's
# Mighty Adventure (2000) for the Game Boy Color and PlayStation.[273] Timon and Pumbaa also appeared in Timon &
# Pumbaa's Jungle Games, a 1995 PC game collection of puzzle games by 7th Level, later ported to the SNES by
# Tiertex.[274] The Square Enix series Kingdom Hearts features Simba as a recurring summon,[275][276] as well as a
# playable in the Lion King world, known as Pride Lands, in Kingdom Hearts II. There the plotline is loosely related
# to the later part of the original film, with all of the main characters except Zazu and Sarabi.[277] The Lion King
# also provides one of the worlds featured in the 2011 action-adventure game Disney Universe,[278] and Simba was
# featured in the Nintendo DS title Disney Friends (2008).[279] The video game Disney Magic Kingdoms includes some
# characters of the film and some attractions based on locations of the film as content to unlock for a limited
# time.[280][281] Walt Disney Theatrical produced a musical stage adaptation of the same name, which premiered in
# Minneapolis, Minnesota in July 1997, and later opened on Broadway in October 1997 at the New Amsterdam Theatre. The
# Lion King musical was directed by Julie Taymor[282] and featured songs from both the movie and Rhythm of the Pride
# Lands, along with three new compositions by Elton John and Tim Rice. Mark Mancina did the musical arrangements and
# new orchestral tracks.[283] To celebrate the African culture background the story is based on, there are six
# indigenous African languages sung and spoken throughout the show: Swahili, Zulu, Xhosa, Sotho, Tswana, Congolese.[
# 284] The musical became one of the most successful in Broadway history, winning six Tony Awards including Best
# Musical, and despite moving to the Minskoff Theatre in 2006, is still running to this day in New York, becoming the
# third longest-running show and highest grossing Broadway production
#
# maul him to death. With Scar and the hyenas gone, Simba takes his rightful place as king, with Nala as his queen.
# With the Pride Lands restored, Rafiki presents Simba and Nala's newborn cub to the assembled animals,
# thus continuing the circle of life. The origin of the concept for The Lion King is widely disputed.[7][8][9]
# According to Charlie Fink (then-Walt Disney Feature Animation's vice president for creative affairs), he approached
# Jeffrey Katzenberg, Roy E. Disney, and Peter Schneider with a "Bambi in Africa" idea with lions. Katzenberg balked
# at the idea at first, but nevertheless encouraged Fink and his writers to develop a mythos to explain how lions
# serviced other animals by eating them.[10] Another anecdote states that the idea was conceived during a
# conversation between Katzenberg, Roy E. Disney, and Schneider on a flight to Europe during a promotional tour.[l]
# During the conversation, the topic of a story set in Africa came up, and Katzenberg immediately jumped at the
# idea.[12] Katzenberg decided to add elements involving coming of age and death, and ideas from personal life
# experiences, such as some of his trials in his career in politics, saying about the film, "It is a little bit about
# myself."[13] On October 11, 1988, Thomas Disch (the author of The Brave Little Toaster) had met with Fink and Roy
# E. Disney to discuss the idea, and within the next month, he had written a nine-paged treatment entitled King of
# the Kalahari.[14][15] Throughout 1989, several Disney staff writers, including Jenny Tripp, Tim Disney,
# Valerie West and Miguel Tejada-Flores, had written treatments for the project. Tripp's treatment, dated on March 2,
# 1989, introduced the name "Simba" for the main character, who gets separated from his pride and is adopted by
# Kwashi, a baboon, and Mabu, a mongoose. He is later raised in a community of baboons. Simba battles an evil jackal
# named Ndogo, and reunites with his pride.[16] Later that same year, Fink recruited his friend J. T. Allen,
# a writer, to develop new story treatments. Fink and Allen had earlier made several trips to a Los Angeles zoo to
# observe the animal behavior that was to be featured in the script. Allen completed his script, which was titled The
# Lion King, on January 19, 1990. However, Fink, Katzenberg, and Roy E. Disney felt Allen's script could benefit from
# a more experienced screenwriter, and turned to Ronald Bass, who had recently won an Academy Award for Best Original
# Screenplay for Rain Man (1988). At the time, Bass was preoccupied to rewrite the script himself, but agreed to
# supervise the revisions. The new script, credited to both Allen and Bass, was retitled King of the Beasts and
# completed on May 23, 1990.[16] Sometime later, Linda Woolverton, who was also writing Beauty and the Beast (1991),
# spent a year writing several drafts of the script, which was titled King of the Beasts and then King of the
# Jungle.[17] The original version of the film was vastly different from the final product. The plot
#
# Nala at the film's premiere.[44] English actors Tim Curry, Malcolm McDowell, Alan Rickman, Patrick Stewart,
# and Ian McKellen were considered for the role of Scar,[45] which eventually went to fellow Englishman Jeremy
# Irons.[46] Irons initially turned down the part, as he felt uncomfortable going to a comedic role after his
# dramatic portrayal of Claus von Bülow in Reversal of Fortune (1990). His performance in that film inspired the
# writers to incorporate more of his acting as von Bülow in the script – adding one of that character's lines,
# "You have no idea" – and prompted animator Andreas Deja to watch Reversal of Fortune and Damage (1992) in order to
# incorporate Irons' facial traits and tics.[40][47] "The Lion King was considered a little movie because we were
# going to take some risks. The pitch for the story was a lion cub gets framed for murder by his uncle set to the
# music of Elton John. People said, 'What? Good luck with that.' But for some reason, the people who ended up on the
# movie were highly passionate about it and motivated." The development of The Lion King coincided with that of
# Pocahontas (1995), which most of the animators of Walt Disney Feature Animation decided to work on instead,
# believing it would be the more prestigious and successful of the two.[30] The story artists also did not have much
# faith in the project, with Chapman declaring she was reluctant to accept the job "because the story wasn't very
# good",[48] and Burny Mattinson telling his colleague Joe Ranft: "I don't know who is going to want to watch that
# one."[49] Most of the leading animators either were doing their first major work supervising a character,
# or had much interest in animating an animal.[13] Thirteen of these supervising animators, both in California and in
# Florida, were responsible for establishing the personalities and setting the tone for the film's main characters.
# The animation leads for the main characters included Mark Henn on young Simba, Ruben A. Aquino on adult Simba,
# Andreas Deja on Scar, Aaron Blaise on young Nala, Anthony DeRosa on adult Nala, and Tony Fucile on Mufasa.[5]
# Nearly twenty minutes of the film, including the "I Just Can't Wait to Be King" sequence,[18] was animated at the
# Disney-MGM Studios facility. More than 600 artists, animators, and technicians contributed to The Lion King.[21]
# Weeks before the film's release, the 1994 Northridge earthquake shut down the studio and required the animators to
# complete via remote work.[50] The character animators studied real-life animals for reference, as was done for
# Bambi (1942). Jim Fowler, renowned wildlife expert, visited the studios on several occasions with an assortment of
# lions and other savannah inhabitants to discuss behavior and help the animators give their drawings authenticity.[
# 51] The animators also studied animal movements at the Miami MetroZoo under guidance from wildlife expert Ron
# Magill.[52] The Pride Lands are modeled on the Kenyan national park visited by the crew. Varied focal lengths and
# lenses were employed to differ from the habitual portrayal
# Question: What is the name of the main character in the movie 'The Lion King'?
# Assistant:
# ==================================================
# Answer:
# According to the context provided, the main character in the movie "The Lion King" is Simba.
