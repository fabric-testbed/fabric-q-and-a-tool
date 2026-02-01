import openai
import os
from dotenv import load_dotenv

load_dotenv()

def check_openai_api_key(api_key):
    openai.api_key = api_key
    try:
        # Attempt to list models, a basic API call
        openai.models.list()
        return True
    except openai.AuthenticationError:
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

# Load API key from .env file
api_key = os.getenv('OPEN_AI_SECRET')

if not api_key:
    print("Error: OPEN_AI_SECRET not found in .env file")
else:
    is_valid = check_openai_api_key(api_key)

    if is_valid:
        print("Valid OpenAI API key.")
    else:
        print("Invalid OpenAI API key.")