from llm_api_call import initialize_api,generate_content_with_retries
from itertools import permutations
import pandas as pd
import logging

DATASET_PATH = "data/raw/codeswitch_v02.json"

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# BACKGROUND:
background = """Background:
Generate a conversation between two people (personas specified below) involving code-switching between English and Hindi. Read both personas below and generate a scenario based on both speakers.

"""

# PERSONAS (persona/role prompting):
people = [
    "Arjun Mehta is a 29-year-old male software engineer at Microsoft on the Azure Storage team, and he speaks English and Hindi.",
    "Maya Rao is a 20-year-old female student at Northeastern University majoring in Computer Science and Data Science, and she speaks English and Hindi.",
    "Daniel Kim is a 47-year-old male algorithms professor at a research university, and he speaks English and Hindi.",
    "Priya Shah is a 35-year-old female project manager at Amazon Web Services on the Serverless Analytics team, and she speaks English and Hindi.",
    "Ethan Patel is a 12-year-old male 7th-grade student at Lincoln Middle School, and he speaks English and Hindi.",
    "Sofia Martinez is a 21-year-old female student in a struggling group project at Boston University majoring in Information Systems, and she speaks English and Hindi."
]

# creates formatted text for prompt
def generate_person(person1, person2):
    return f"""
    Personas: 
    Person1: {person1}
    Person2: {person2}
    """

# PROMPTS (few-shot):
# Inter-sentential
prompt1 = """
Prompt:
This conversation should mimic inter-sentential code-switching. Inter-sentential code-switching is the practice of alternating between languages at sentence boundaries, where one complete sentence is spoken in one language and the next sentence is in a different language. Use the examples below as a reference for this conversation.

Inter-sentential Example Sentences:
1. The TA extended the deadline by 24 hours. <Hindi>कृपया आज ही ड्राफ्ट अपलोड कर दें।</Hindi>
2. <Hindi>मैंने डेटा क्लीन कर दिया है।</Hindi> The visualization notebook is ready for review.
3. I pushed the hotfix to the staging branch. The tests passed on CI. <Hindi>कल सुबह प्रोडक्शन पर डिप्लॉय करेंगे।</Hindi>
4. <Hindi>मीटिंग पाँच बजे शुरू होगी।</Hindi> <Hindi>सब लोग समय पर जुड़ें।</Hindi> The Zoom link is in the calendar invite.
5. I booked the conference room for two hours and shared the agenda with the team. We’ll walk through risks and dependencies first. <Hindi>अगर कुछ जोड़ना हो तो अभी बता दें।</Hindi>
6. <Hindi>कल का डेमो बहुत अच्छा गया।</Hindi> The client specifically liked the offline mode.
7. The crawler finished indexing the new sources. I’ve queued the embeddings job for tonight. <Hindi>सुबह तक डैशबोर्ड अपडेट मिल जाएगा।</Hindi>
8. <Hindi>फीडबैक डॉक्यूमेंट शेयर कर दिया है।</Hindi> Please add comments inline. We’ll consolidate everything by Friday.
9. I’ll send you the draft proposal by noon. <Hindi>आप बस रेफरेंसेज़ चेक कर लीजिए।</Hindi>
10. <Hindi>मैंने सभी टिकट्स को प्राथमिकता दे दी है।</Hindi> Let me know if you want any of them reassigned.

"""

# Intra-sentential
prompt2 = """
Prompt:
This conversation should mimic intra-sentential code-switching. Intra-sentential code-switching is the mixing of two or more languages, dialects, or registers within a single sentence. This type of switching occurs in the middle of a sentence without hesitation, seamlessly blending elements from different linguistic systems. Use the examples below as a reference for this conversation.

Intra-sentential Example Sentences:
1. <Hindi>आज थोड़ी देरी हो जाएगी</Hindi>, but I’ll still join the standup.
2. <Hindi>अगर तुम्हें सही लगे</Hindi>, we can deploy the patch tonight.
3. <Hindi>थोड़ा नेटवर्क स्लो है</Hindi>, so the upload might take longer.
4. I reviewed your PR, <Hindi>बस डॉक्यूमेंटेशन अपडेट कर दो</Hindi>, and then we can merge.
5. The lecture was great, <Hindi>खासकर जब प्रोफेसर ने उदाहरण दिए</Hindi>, it all clicked.
6. Meet me outside the library, <Hindi>जहाँ हम डिबगिंग शुरू करेंगे</Hindi>, and bring your laptop.
7. Let’s finalize the dataset split by noon, <Hindi>ताकि ट्रेनिंग समय पर शुरू हो सके</Hindi>.
8. I’ll handle the client email first, <Hindi>फिर हम रिपोर्ट पर साथ में काम करेंगे</Hindi>.
9. The prototype runs fine on my machine, <Hindi>लेकिन प्रोडक्शन के लिए और टेस्ट चाहिए</Hindi>.

"""

# Tag Switching
prompt3 = """
Prompt:
This conversation should mimic tag switching. Tag switching is a type of code-switching where short phrases or words from one language are inserted into a sentence of another. Use the examples below as a reference for this conversation.

Tag Switching Example Sentences:
1. <Hindi>वैसे</Hindi>, the deploy is scheduled for tonight.
2. The dataset is ready, <Hindi>ठीक है?</Hindi> let’s kick off the job.
3. Can you review the PR, <Hindi>कृपया</Hindi>.
4. <Hindi>देखो</Hindi>, the bug only appears on iOS 18.
5. We’ll present first and share the repo link after, <Hindi>चलो</Hindi>.
6. <Hindi>सुनो</Hindi>, the lab will be open until 9 PM.
7. The report is in your inbox, <Hindi>है ना?</Hindi>
8. We can ship the hotfix today, <Hindi>ठीक?</Hindi> if QA signs off.
9. <Hindi>अच्छा</Hindi>, I’ll update the roadmap after the meeting.
10. Push the latest notebook to Drive, <Hindi>कृपया</Hindi>.

"""

# dictionary showing what code-switch type is being generated per prompt
prompts = {"inter-sentential" : prompt1, "intra-sentential" : prompt2, "tag-switching" : prompt3}

# SPECIFICATIONS (chain-of-thought):
specifications = """
Specifications:
Generate 10-12 sentences per conversation. Alternate the speakers at every sentence. 25 percent of sentences need to include a code switch (1 out of 4 sentences). Generate ALL Hindi words in Devanagari characters, DO NOT generate Hindi words in romanized English. Generate each sentence in the following form: “<Speaker>: <sentence>.<line-break> ”. Look at the examples of code-switched and non-code-switched sentences below.

Example code-switched sentence: “Person1: Hey there, I hope everything is going well and <Hindi>आप कैसे हैं</Hindi> I heard the midterm on searching algorithms is next week."

Example non-code-switched sentence: “Person1: Hi Rohan, thanks for joining the call, I wanted to discuss the current status of the feature delivery.”

"""

# loads a dataframe if one exisits at path or none
def load_or_init_dataset(path):
    try:
        return pd.read_json(path)
    except Exception:
        return pd.DataFrame([])


if __name__ == "__main__":
    log.info("Script Started")
    # initialize OpenRouter LLM API

    try:
        log.info("Initialzing OpenRouter API...")
        initialize_api()
    except Exception as e:
        log.warning(f"Error initializing LLM API: {e}")

    # add to current dataset or start from scratch
    dataset = load_or_init_dataset(DATASET_PATH)

    # number of conversations generated
    total_generations = 0

    try:
        log.info("Generating conversations...")
         # keep looping until 200 conversations are in the dataset
        while total_generations < 178:

            # try each combination of people
            # 6 people x 5 other people = 30 conversations
            for p1, p2 in permutations(people, 2):
                persona = generate_person(p1, p2)

                # for each combination of people, generate a conversation for each code-switch type
                # 3 prompts x 30 conversations = 90 conversations per iteration
                for strategy_name, base_prompt in prompts.items():

                    # construct the full prompt
                    prompt = background + persona + base_prompt + specifications

                    # call the LLM API
                    response = generate_content_with_retries(prompt=prompt)

                    # create row and add to dataset
                    row = pd.DataFrame([{"utterance": response,
                                        "generation_strategy": strategy_name,
                                        "source":"LLM",
                                        "metadata":["minimax/minimax-m2:free",prompt],
                                        "id":len(dataset)}])
                    dataset = pd.concat([dataset, row], ignore_index=True)
                    total_generations += 1

                    # save after each row to save data in case of issues during generation
                    dataset.to_json(DATASET_PATH, orient="records", indent=2)
        log.info(f"Successfully generated {total_generations} conversations!")
    except Exception as e:
        log.warning(f"Error generating conversations: {e}")
    
    log.info("Script ended")