from pydantic import BaseModel, Field, ValidationError

class Customer(BaseModel):
    name: str
    country: str
    total_spent: float = Field(ge=0)      # must be 0 or more
    vip: bool = False                     # optional, default False

# 1. Valid data. Notice "49.62" (a string) gets CONVERTED to a float.
c = Customer(name="Helena", country="Czech Republic", total_spent="49.62")
print(c)
print("total_spent type:", type(c.total_spent).__name__)   # float
print("As dict:", c.model_dump())

# 2. Invalid data → a clear error instead of bad data sneaking through.
try:
    Customer(name="X", country="Y", total_spent=-5)
except ValidationError as e:
    print("Rejected:", e.errors()[0]["msg"])

# 3. The JSON Schema. This is what LangChain sends to the LLM.
print(Customer.model_json_schema())