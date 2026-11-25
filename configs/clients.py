import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic_ai import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

local_provider = OpenAIProvider(
    base_url="http://127.0.0.1:9931/v1", api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize client targeting your local llama.cpp endpoint
openai_client = OpenAI(
    base_url="http://127.0.0.1:9931/v1", api_key=os.getenv("OPENAI_API_KEY")
)


# 1. Subclass the model to override the display label
class LabeledOpenAIChatModel(OpenAIChatModel):
    @property
    def label(self) -> str:
        return "Local"


# 2. Initialize using your custom labeled class
local_model = LabeledOpenAIChatModel(
    model_name=os.getenv("LOCAL_MODEL_NAME", ""),
    provider=local_provider,
    settings=ModelSettings(thinking="medium"),
)
