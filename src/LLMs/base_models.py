# base_models.py
from src.get_dotenv import config
from langchain_openai import ChatOpenAI


router_model = ChatOpenAI(
    model = config.ROUTER_MODEL,
    base_url = config.LLM_URL,
    api_key = config.LLM_API_KEY,
    temperature = 0.1,
    top_p=0.5,
    max_retries=2
)

actor_model = ChatOpenAI(
    model = config.ACTOR_MODEL,
    base_url = config.LLM_URL,
    api_key = config.LLM_API_KEY,
    temperature = 0.1,
    top_p=0.5,
    max_retries=2
)

planner_model = ChatOpenAI(
    model = config.PLANNER_MODEL,
    base_url = config.LLM_URL,
    api_key = config.LLM_API_KEY,
    temperature = 0.1,
    top_p=0.5,
    max_retries=2
)

