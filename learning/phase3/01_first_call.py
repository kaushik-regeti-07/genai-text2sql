from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load GOOGLE_API_KEY from the .env file into environment variables.
#    load_dotenv() searches for .env starting from this script's folder and going up,
#    so it finds F:\Project\Gen_AI\.env
load_dotenv()

# 2. Create the chat model. It reads GOOGLE_API_KEY from the environment automatically.
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 3. Send a plain string (LangChain turns it into a HumanMessage for us).
response = llm.invoke("What is SQL in one sentence?")

# 4. Look at what came back.
print("Type:   ", type(response).__name__)   # AIMessage
print("Answer: ", response.text)             # just the text
print("Tokens: ", response.usage_metadata)   # input / output / total tokens
