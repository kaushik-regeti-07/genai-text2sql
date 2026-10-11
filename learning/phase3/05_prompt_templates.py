from langchain_core.prompts import PromptTemplate

# 1. Create a template with two blanks: {keyword} and {level}.
prompt = PromptTemplate.from_template(
    "Explain the SQL keyword {keyword} to a {level} in one sentence."
)

# 2. LangChain finds the variables for us.
print("Variables:", prompt.input_variables)   # ['keyword', 'level']

# 3. Fill in the blanks with a dictionary.
filled = prompt.invoke({"keyword": "JOIN", "level": "10-year-old"})
print("Type:", type(filled).__name__)          # StringPromptValue
print("Text:", filled.to_string())

# 4. Forget a variable and you get a clear error (an f-string would fail silently).
try:
    prompt.invoke({"keyword": "JOIN"})
except KeyError as e:
    print("Error:", e)