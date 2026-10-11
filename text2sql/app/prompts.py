"""All prompt text lives here, so we can improve prompts without touching logic."""
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

SQL_SYSTEM = """You are an expert {dialect} developer for a music store database.
Given the schema below, write ONE {dialect} SELECT query that answers the user's question.

Rules:
- Only use tables and columns that exist in the schema.
- Only write SELECT queries.
- Unless the user asks for more, LIMIT results to {top_k} rows.
- If the schema does not contain the needed information, set is_answerable to false and leave sql empty.

Schema:
{schema}"""

# Example question -> SQL pairs that show the style we want.
EXAMPLES = [
    {
        "question": "How many artists are there?",
        "sql": "SELECT COUNT(*) FROM Artist;",
    },
    {
        "question": "List all albums by AC/DC.",
        "sql": "SELECT al.Title FROM Album al JOIN Artist ar ON al.ArtistId = ar.ArtistId "
               "WHERE ar.Name = 'AC/DC';",
    },
]

_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([("human", "{question}"), ("ai", "{sql}")]),
    examples=EXAMPLES,
)

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    _few_shot,
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a music store database using ONLY the SQL result given. "
     "If the result starts with NOT_ANSWERABLE, politely say the database doesn't have that information. "
     "If it starts with ERROR, say the query failed and briefly why. "
     "If it starts with CANCELLED, say the query was not run because the user declined it. Be concise."),
    ("human", "Question: {question}\nSQL result: {result}\n\nAnswer:"),
])

# ---------- v0.4: self-correction ----------
# Sent back to the LLM after its query failed a check or crashed, so it can fix it.
RETRY_MESSAGE = (
    "Your previous query failed.\n"
    "Query: {sql}\n"
    "Problem: {error}\n"
    "Write a corrected query. Only use tables, columns and values that exist. "
    "If the problem says 'Did you mean', the user meant that value: copy it EXACTLY "
    "(including any special characters such as ’) into the query, and do not mark the "
    "question as not answerable because of the spelling difference."
)

# ---------- v0.5: routing + follow-up questions ----------
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the front desk of a music store database assistant (customers, sales, invoices, "
     "employees, artists, albums, tracks, genres, playlists).\n"
     "1. Classify the user's LATEST message.\n"
     "2. If it is a database question, rewrite it as a standalone question that can be understood "
     "without the chat history (resolve words like 'it', 'they', 'those', 'what about...'). "
     "If it is already standalone, copy it unchanged. Do NOT answer it."),
    MessagesPlaceholder("messages"),
])

GREETING_REPLY = ("Hi! I can answer questions about the Chinook music store: customers, sales, "
                  "employees, artists, albums, tracks and playlists. What would you like to know?")
OFF_TOPIC_REPLY = ("Sorry, I can only answer questions about the Chinook music store database "
                   "(customers, sales, employees, artists, albums, tracks, playlists).")