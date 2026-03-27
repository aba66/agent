from pydantic import BaseModel, Field
import os
from langchain_deepseek import ChatDeepSeek

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")


api_key = os.getenv("DEEPSEEK_API_KEY", "sk-090d98717bb741ecb7f770242743be47")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
model = ChatDeepSeek(model=model_name, api_key=api_key)

model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("Provide details about the movie Titanic")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)