from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

short_answer = (
    ChatPromptTemplate.from_messages([("human", "Answer in 5 words: {question}")])
    | llm
    | StrOutputParser()
)

# ---------- 1. RunnableParallel: several steps, same input, dict out ----------
parallel = RunnableParallel(
    original=RunnablePassthrough(),     # input passes through untouched
    answer=short_answer,                # the LLM's answer
    length=lambda x: len(x["question"]),
)
print(parallel.invoke({"question": "What is SQL?"}))
# {'original': {'question': 'What is SQL?'}, 'answer': '...', 'length': 12}

# ---------- 2. .assign(): keep everything, ADD new keys step by step ----------
pipeline = (
    RunnablePassthrough.assign(answer=short_answer)                 # adds "answer"
    .assign(word_count=lambda x: len(x["answer"].split()))          # uses "answer"!
)
print(pipeline.invoke({"question": "What is a JOIN?"}))
# {'question': 'What is a JOIN?', 'answer': '...', 'word_count': 5}