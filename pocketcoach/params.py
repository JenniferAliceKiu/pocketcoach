import os

with open(os.getenv("SYSTEM_PROMPT_FILE"), encoding="utf8") as f:
    SYSTEM_PROMPT = f.read()

##################  VARIABLES  ##################
LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")
API_URL = os.environ.get("API_URL")
LOGIN_URL = os.environ.get("LOGIN_URL")
