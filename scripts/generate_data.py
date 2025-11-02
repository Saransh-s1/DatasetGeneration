from llm_api_call import initialize_api,generate_content_with_retries
import pandas as pd

initialize_api()

responses = []

while True:
    try:
        dataset = pd.read_json("data/raw/codeswitch_v01.json")
    except:
        dataset = pd.DataFrame({})
    if len(dataset) >= 10:
        print("here")
        break
    print(dataset)
    print(len(dataset))
    prompt1 = """
    Act as a 30 year old male software engineer from India working for Microsoft who can speak English and Hindi.
    You are talking to your 45 year old male manager who is a lead software engineer also from India. You guys are speaking
    in English and code switching to Hindi using intra-sentential switching. You guys are talking about a delay in a current
    project where you are responsible for the delay. Generate me the conversation with the code switching in Hindi output in
    English characters.Use the following lines as examples:
    I'm going to the library. Kal wahan bahut shor tha. Let's see if it's quiet today.
    Tumne kal ka match dekha? It was really exciting. Sab log stadium mein cheer kar rahe the.
    We should leave early. Raste mein traffic ho sakta hai. Better to be on time.
    Maine usse kal phone kiya. She didn't pick up. Shayad woh busy thi.
    Let's grab some coffee. Mujhe neend aa rahi hai. After that, we can continue working.
    Aaj mausam bahut accha hai. The sky looks so clear. Hum thoda walk kar lete hain.
    He always works late. Uske colleagues pareshaan hote hain. But they respect his dedication.
    Kal exam hai. I still have two chapters left. Mujhe raat ko padhna padega.
    We were planning a trip. Maine tickets book kar li hain. Everyone is excited now.
    She loves dancing. Usne bachpan se training li hai. That's why she performs so confidently.
    """

    prompt2 = """
    Use the 10 examples below to generate a conversation between 2 people.
    Make each character not speak for more than 10 times. The 2 people will be code switching between
    English and Hindi. Make person 1 have the first code switch. Make the code switches highlighting intra-sentinel switching.
    Examples: 
    I was so tired kal raat that I fell asleep instantly.
    Usne bola “I’ll call you baad mein,” but phir call hi nahi kiya.
    Let’s meet kal subah at the cafe near your ghar.
    I don’t think mujhe woh movie pasand aayegi.
    Humne project complete kar liya, but abhi presentation pending hai.
    The teacher said ki tumhe zyada practice ki zaroorat hai.
    Mujhe samajh nahi aata why he always comes late.
    They were laughing itna zyada ki sab log dekhne lage.
    I’m hungry, kya hum abhi kuch order karen?
    Kal ka test tough tha but thankfully sab questions practice wale hi the
    """

    prompt3 = """
    You are an international student from India. You are talking to person 2 in your
    AI class about the upcoming midterm on searching algorithms. You both speak English and Hindi.
    Generate a conversation between the 2 people.
    Make person 1 have the first code switch. Highlight the code switch as tag switching.
    """

    specficiations = "Average between 10-12 conversations per turn, "
    specficiations += "simulate a natural billingual conversation, "
    specficiations += "structure the result conversation to only include periods at the end of a sentence, "
    specficiations += "only return the conversation, have it in a format similar to speaker: sentence, do not include anything else."
    specficiations += "IMPORTANT: Only code switch in EXACTLY 1 out of 4 sentences (25% of sentences should have Hindi). "
    specficiations += "This means 3 sentences should be pure English, then 1 sentence with Hindi mixed in. "
    specficiations += "Do NOT code switch more frequently than this ratio. "
    specficiations += "generate all hindi words using the the devangari"
    specficiations += "one example is Person1: Hey there, I hope everything is going well and <Hindi>आप कैसे हैं</Hindi> I heard the midterm on searching algorithms is next week."
    specficiations += "ensure to ONLY USE DEVANGARI and not romanized hindi"
  
    
    people = [
        "software engineer at Microsoft",
        "student at northeastern university",
        "algorithms professor",
        "project manager on AWS",
        "son studying for math test",
        "student in a struggling group project",
    ]

    prompts = {
        prompt1:"intra_sentinal",
        prompt2:"intra_sentinal",
        prompt3:"tag_switching"
    }
    for i,v in enumerate(people):
        for j,val in enumerate(people):
            if i == j:
                continue

            for base,type in prompts.items():
                prompt = (
                    base
                    + specficiations
                    + f"person1 is {v}\n"
                    + f"person2 is {val}"
                )
                response = generate_content_with_retries(prompt=prompt)
                responses.append(response)

                row = pd.DataFrame([{"utterance": response,
                                    "generation_strategy":type,
                                    "source":"LLM",
                                    "metadata":["minimax/minimax-m2:free",prompt],
                                    "id":len(dataset)}])  # wrap in list -> one-row DF
                dataset = pd.concat([dataset, row], ignore_index=True)  # append correctly
                dataset.to_json("data/raw/codeswitch_v01.json", orient="records", indent=2)  # fix path + nice formatting


