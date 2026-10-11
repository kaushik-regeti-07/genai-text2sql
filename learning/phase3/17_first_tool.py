from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_track_price_in_inr(price_usd: float) -> str:
    """Convert a track price from US dollars to Indian rupees."""
    return f"₹{price_usd * 88:.2f}"      # rough exchange rate, fine for learning

# 1. The decorator turned the function into a tool object.
print("Type:       ", type(multiply).__name__)      # StructuredTool
print("Name:       ", multiply.name)                # multiply
print("Description:", multiply.description)         # from the docstring
print("Args:       ", multiply.args)                # from the type hints

# 2. You can run a tool yourself. Tools are Runnables, so they have .invoke().
print("Result:     ", multiply.invoke({"a": 6, "b": 7}))                  # 42.0
print("In rupees:  ", get_track_price_in_inr.invoke({"price_usd": 0.99}))