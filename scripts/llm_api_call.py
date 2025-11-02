from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import logging
import pandas as pd

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# pulls in API Key and URL and creates intial connection to LLM client
def initialize_api():
    # loads API Key and URL from .env file
    load_dotenv()

    # sets API Key and URL
    API_KEY = os.getenv("API_KEY")
    API_URL = os.getenv("API_URL")

    # checks if API Key and URL are not null and throws exception if they are
    if not API_KEY or not API_URL:
        raise ValueError("API_KEY or API_URL not found in environment variables.")

    # initializes client for LLM connection
    client = OpenAI(
    base_url=API_URL,
    api_key=API_KEY,)

    return client 

# makes the api call using the client intialized and prompt
def make_api_call(client, prompt):
    # checks if client and prompt are valid
    if not client:
        raise ValueError("Client is None.")
    if not prompt:
        raise ValueError("Prompt is empty.")
    
    # makes the api call and returns the content of the response
    completion = client.chat.completions.create(
    extra_body={},
    model="nvidia/nemotron-nano-9b-v2:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content

def generate_content_with_retries(prompt):
    log.info("Initializing API client...")
    # Connects to the API
    client = initialize_api()
    log.info("API client initialized.")

    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            log.info(f"Attempting API call (Attempt {retries + 1})...")
            log.info("Making API call...")
            # Makes the API call
            response = make_api_call(client, prompt)

            # Check if response is valid
            if not response:
                raise ValueError("Empty response received from API.")
            
            return response
        except Exception as e:
            log.error(f"API call failed: {e}")

            # Exponential backoff before retrying
            if retries < max_retries - 1:
                wait_time = 2 ** retries
                log.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                log.error("Max retries reached. API call failed permanently.")
                return None

    return None

if __name__ == "__main__":
    prompt = "Generate a conversation between two friends discussing their favorite movies." \
    "Make each sentence have a code switch between English and Arabic. Return the arabic words in arabic characters."
    response = generate_content_with_retries(prompt)
    print("----------------- LLM Response: -----------------")
    print(response)