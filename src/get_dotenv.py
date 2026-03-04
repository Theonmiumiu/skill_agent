from dotenv import load_dotenv
import os

load_dotenv()
class Config:
    LLM_API_KEY = os.getenv('LLM_API_KEY')
    LLM_URL = os.getenv('LLM_URL')
    ROUTER_MODEL = os.getenv('ROUTER_MODEL')
    ACTOR_MODEL = os.getenv('ACTOR_MODEL')
    PLANNER_MODEL = os.getenv('PLANNER_MODEL')
    REPORTER_MODEL = os.getenv('REPORTER_MODEL')

config = Config()