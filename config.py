import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    UPLOAD_FOLDER = 'data/'
    SQLITE_DB_PATH = 'test_answers.db'  