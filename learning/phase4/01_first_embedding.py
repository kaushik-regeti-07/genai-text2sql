import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector = embeddings.embed_query("Which genre sells the most?")

print("Type:        ", type(vector).__name__)            # list
print("Length:      ", len(vector))                      # 3072
print("First 5:     ", [round(x, 4) for x in vector[:5]])
print("Vector norm: ", round(float(np.linalg.norm(vector)), 4))   # 1.0