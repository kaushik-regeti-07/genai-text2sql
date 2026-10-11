from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

class Route(BaseModel):
    category: Literal["database_question", "greeting", "off_topic"] = Field(
        description="database_question = anything answerable from the music store database "
                    "(customers, sales, artists, albums, tracks, employees). "
                    "greeting = hello/thanks/small talk. off_topic = anything else."
    )
    reason: str = Field(description="One short sentence explaining the choice")

router = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify the user's message for a music-store database assistant."),
        ("human", "{question}"),
    ])
    | llm.with_structured_output(Route)
)

for q in ["How many albums does Aerosmith have?",
          "Hi there!",
          "What's the weather in Paris?",
          "Which country buys the most jazz?"]:
    r = router.invoke({"question": q})
    print(f"{q:40} → {r.category:18} ({r.reason})")