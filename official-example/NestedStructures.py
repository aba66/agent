from pydantic import BaseModel, Field
import os
from langchain_deepseek import ChatDeepSeek

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")


api_key = os.getenv("DEEPSEEK_API_KEY", "sk-090d98717bb741ecb7f770242743be47")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
model = ChatDeepSeek(model=model_name, api_key=api_key)

model_with_structure = model.with_structured_output(MovieDetails, include_raw=True)
response = model_with_structure.invoke("Provide details about the movie Titanic")
print(response)