from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

class SupportTicket(BaseModel):
    """Information extracted from a customer message."""
    customer_name: Optional[str] = Field(None, description="The customer's name, if mentioned")
    product: str = Field(description="The album, track or product the message is about")
    sentiment: Literal["positive", "neutral", "negative"]
    wants_refund: bool = Field(description="True if the customer asks for their money back")
    summary: str = Field(description="One-sentence summary of the message")

extractor = llm.with_structured_output(SupportTicket)

# 1. Directly on a string
message = ("Hi, this is Ravi. I bought the album 'Let There Be Rock' yesterday "
           "but 3 tracks won't play. Really disappointed, I want my money back.")
ticket = extractor.invoke(message)
print(type(ticket).__name__)            # SupportTicket, a real Python object!
print(ticket)
if ticket.wants_refund:                 # ← normal code can now make decisions
    print("→ Send to the refunds team")

# 2. Inside an LCEL chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract support ticket info from the customer's message."),
    ("human", "{message}"),
])
chain = prompt | extractor
print(chain.invoke({"message": "Loving the new AC/DC playlist, great job!"}))

# 3. Debug view: see the raw reply AND any parsing error
debug = llm.with_structured_output(SupportTicket, include_raw=True).invoke(message)
print("Keys:", list(debug.keys()))      # ['raw', 'parsed', 'parsing_error']
print("Parsing error:", debug["parsing_error"])