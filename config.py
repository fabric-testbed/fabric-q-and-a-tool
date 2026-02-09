from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------------------------------------------------------------------------------------- #
# -------------------------------------------- Configuration ------------------------------------------ #
# ----------------------------------------------------------------------------------------------------- #
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
OPEN_AI_SECRET = os.getenv('OPEN_AI_SECRET')

LOG_DIR = os.getenv('LOG_DIR')

QA_DB_FILE = os.getenv('QA_DB_FILE')
CG_DB_FILE = os.getenv('CG_DB_FILE')

QA_PROMPT = os.getenv('QA_PROMPT')
CG_PROMPT = os.getenv('CG_PROMPT')

QA_MODEL = os.getenv('QA_MODEL')
CG_MODEL = os.getenv('CG_MODEL')

HOST = os.getenv('HOST')
PORT = os.getenv('PORT')

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_VERIFY_SSL = os.getenv('OLLAMA_VERIFY_SSL', 'true').lower() != 'false'
# ----------------------------------------------------------------------------------------------------- #
# ------------------------------------------ Hyperparameters ------------------------------------------ #
# ----------------------------------------------------------------------------------------------------- #
QA_DOCS = 6
CG_DOCS = 4

QA_TEMP = 0.2
CG_TEMP = 0
