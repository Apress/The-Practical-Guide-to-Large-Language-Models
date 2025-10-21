import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util


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


# Article about Alien: Romulus movie
alien_romulus_url = "https://avp.fandom.com/wiki/Alien:_Romulus"

# parsing the webpage to get the text
response = requests.get(alien_romulus_url)
soup = BeautifulSoup(response.text, 'html.parser')
text = ' '.join([p.text for p in soup.find_all('p')])

# Splitting the context into fixed-size chunks
chunks = split_text(text, size = 500, overlap = 100)

question = "When does the action of the movie Alien: Romulus take place?"

# Defining the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Question embedding
question_embedding = embedding_model.encode(
    question,
    convert_to_tensor = True
)

# Generating embeddings map for each chunk
embeddings_map = {}
for i, chunk in enumerate(chunks):
    embeddings = embedding_model.encode(
        chunk,
        convert_to_tensor = True
    )
    embeddings_map[i] = embeddings

# Comparing the question embedding with each chunk embedding
similarities = {}
for i, embeddings in embeddings_map.items():
    sim = util.cos_sim(question_embedding, embeddings)
    similarities[i] = sim.item()

# Sorting the chunks based on similarity scores
sorted_chunks = sorted(
    similarities.items(),
    key = lambda x: x[1],
    reverse = True
)

# Printing the top 3 most relevant chunks with their similarity scores
print("\nTop 3 most relevant chunks:")
for i, score in sorted_chunks[:3]:
    header = f"Chunk {i + 1} (Score: {score:.4f}):"
    print(f"{header}\n{chunks[i]}\n")
    print("-" * 40)

# Top 3 most relevant chunks:
# Chunk 1 (Score: 0.6395):
# Alien: Romulus Cast & Crew Directed by Fede Álvarez Produced by Ridley ScottMichael Pruss Written by Fede
# ÁlvarezRodo Sayagues Starring Cailee Spaeny[1]David JonssonArchie RenauxIsabela MercedSpike FearnAileen Wu Music
# Benjamin Wallfisch Release Information Release date(s) August 16, 2024[2] Production companies Scott Free
# ProductionsBrandywine Productions Distributed by 20th Century Studios Running time 119 Minutes Budget $80 million[
# 3] Worldwide gross $350.9 million[4] Rating MPAA: RBBFC: Chronology Preceded by Alien Followed by Aliens Rodo
# Sayagues Brandywine Productions Alien: Romulus is a feature film in the Alien franchise. Directed by Fede Álvarez
# and produced by Ridley Scott, it was released on August 16, 2024. While scavenging the deep ends of a derelict
# space station, a group of young space colonizers come face to face with the most terrifying life form in the
# universe. On February 9, 2142, the Echo 203 probe enters the floating wreckage of the USCSS Nostromo and picks up a
# large, organic object. It is eventually brought to a research facility where masked scientists open the object with
# lasers to reveal a cocoon-like structure, inside of which is a curled-up Xenomorph. 170 days later, at the mining
# colony Jackson's Star, scientist Rain Carradine has fulfilled her work contract with the Weyland-Yutani Corporation
# and expects to be allowed to leave for the planet Yvaga III with her adoptive brother Andy, a malfunctioning
# synthetic reprogrammed by Rain's late father. However, due to a worker shortage, a Weyland-Yutani clerk refuses
# Rain's request and instead extends her work contract from the 12,000 hours she already delivered to 24,000,
# telling her to later report to the mines. Afterward, Rain is summoned by her ex-boyfriend, Tyler Harrison. At his
# residence, she meets Tyler's pregnant sister Kay, his cousin Bjorn and Bjorn’s adoptive sister Navarro. Tyler
# informs Rain of a derelict space station, which they presume to be a ship, drifting in orbit above the colony,
# telling her they could use the station's still-functioning hypersleep chambers to facilitate an escape to Yvaga,
# as the hauler they already have can get them there. Rain is initially reluctant to take part in a criminal trespass
# of Weyland-Yutani property, but eventually goes along, specially as Andy is necessary to communicate with the
# station's computers. Tyler's group leaves in the mining hauler Corbelan IV to the derelict craft, which turns out
# to be the Weyland-Yutani research outpost Renaissance, divided into the sections, Romulus and Remus. Tyler,
# Andy and Bjorn board the station and fix the unstable gravity and atmosphere generators. However, they discover the
# hypersleep chambers don't have enough coolant fuel for the nine-year trip to Yvaga. The trio locates a large,
# still-functioning cryonic chamber next to a laboratory and scavenge its fuel. In doing so, they inadvertently
# revive the frozen Facehugger specimens inside and trigger an automatic lockdown. The Facehuggers attack them,
# and Andy doesn't have the security clearance needed to override the lockdown, prompting Navarro and Rain to come to
# their rescue. Rain notices a destroyed synthetic and removes a chip from its brain, and through a gap in the
# laboratory
#
# ----------------------------------------
# Chunk 6 (Score: 0.6296):
# whose characters had died and continued filming without them.[23] Romulus takes place in 2142[9] between the events
# of Alien and Aliens.[8] Most of the film takes place on the space station Renaissance. Many of the characters are
# siblings, whether it be through blood or as surrogate siblings.[9] The film is designed to be accessible to those
# who have not watched any prior Alien films.[24] The film's level of connectivity to said films has differed between
# sources. According to Collider, Romulus will have no connection to any previous Alien film or the Alien TV series.[
# 8] According to Rodo Sayagues (one of Alvarez's collaborators), the film's script does not connect with any prior
# Alien film sans the titular creature.[6] According to Alvarez himself however, connections to the other films are
# not ignored in Romulus, specifically mentioning Alien and Alien: Covenant,[10] and has stated that Romulus does not
# override any prior installment.[6] Humanity's relationship with AI is a theme of the film (represented by
# characters such as Andy), but is not a theme the film seeks to make a commentary on.[25] Siblinghood is another
# intended theme of the film. This is reflected in the character roster (many of the characters are siblings),
# the design and namesakes of Renaissance (the Romulus and Remus modules), and the titular Roman figures.[9]
#
# ----------------------------------------
# Chunk 4 (Score: 0.6107):
# with the ship's cargo pod. After Rain regains control of the ship and saves it from crashing into the rings of
# Jackson's Star, she puts Andy in one of the hypersleep chambers, and in a final log before entering stasis herself,
# reflects on the uncertainty of their arrival at Yvaga but maintains hope. The idea for Romulus was pitched by
# Alvarez to Ridley Scott sometime before 2021. In this year, Scott called Alvarez and asked him if he was still
# willing to make the Alien movie he had pitched. Alvarez succeeded. According to 20th Century Studios president
# Steve Asbell, 20C decided to move forward with the project "purely off the strength of Fede's [Alvarez's] pitch,
# " as it was "just a really good story with a bunch of characters you haven't seen before."[6] An idea floated for
# Romulus early in development was that it wouldn't be sold as an Alien film. Alvarez suggested that the title be
# bereft of "Alien," and would instead have an apparently unrelated title. In this version, the characters would walk
# into a room and see a Xenomorph egg, alerting the audience to the fact that they were actually watching an Alien
# film.[7] Alvarez was attached to the project in 2022.[8] One of the first things Alvarez considered was when to set
# the film. He decided to set it between Alien and Aliens.[9] Ridley Scott served as a producer, while James Cameron
# worked as an unofficial consultant.[10] Romulus uses a retrofuturistic aesthetic reminiscent of the original film,
# [11] with the technology present being the same (or at least similar) to that in Alien, whereas the technology seen
# in Aliens is on the cusp of being developed.[9] The first hour of the film adopts the slower, dread-laden
# atmosphere of Alien, while the second hour is more akin to Aliens in style and tempo.[12] Alvarez used Xenopedia
# when writing the film. Romulus is designed to fit into the canon of the films as much as possible, but according to
# Alvarez, "things got hard" when he started factoring in the novels.[13] Before production began, Alvarez had
# conversations with Ridley Scott about Ian Holm's lack of appearance compared to other actors that had played other
# synthetics such as Lance Henriksen and Michael Fassbender. Before committing to the idea, Alvarez contacted Holm's
# family and widow, who gave consent to use his likeness on the new synthetic character Rook, created to be
# completely different from Holm's portrayal of Ash. Rook was achieved through a combination of animatronics built to
# resemble Holm and actor Daniel Betts performing his lines. As soon as Alvarez finished his rough cut of the film,
# he showed it to Holm's family, who were one of the first groups of people to see it, following James Cameron and
# Ridley Scott. [14][15] The characters are younger than those found in other Alien films. The idea for this came
# from the extended cut of Aliens, where children can be seen in Hadley's Hope before the Xenomorph outbreak.
# According to Alvarez, "what would it
#
# ----------------------------------------
