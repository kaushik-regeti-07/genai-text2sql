"""Evaluate schema RAG (v0.3) against sending the full schema (v0.2).

Run from the Gen_AI folder:   python -m text2sql.evals.eval_schema_rag
"""
from sqlalchemy import text

from text2sql.app.chains import build_sql_chain
from text2sql.app.db import get_db, get_engine, run_sql
from text2sql.app.schema_index import find_tables

# question, tables it needs, a correct ("gold") SQL query written by hand
GOLD = [
    ("How many customers are from Brazil?", {"Customer"},
     "SELECT COUNT(*) FROM Customer WHERE Country = 'Brazil'"),
    ("Which genre earned the most money?", {"Genre", "Track", "InvoiceLine"},
     "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
     "JOIN InvoiceLine il ON il.TrackId = t.TrackId GROUP BY g.GenreId "
     "ORDER BY SUM(il.UnitPrice * il.Quantity) DESC LIMIT 1"),
    ("Who is Jane Peacock's manager?", {"Employee"},
     "SELECT m.FirstName FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "WHERE e.FirstName = 'Jane' AND e.LastName = 'Peacock'"),
    ("What is the longest song by AC/DC?", {"Track", "Album", "Artist"},
     "SELECT t.Name FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId "
     "JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'AC/DC' "
     "ORDER BY t.Milliseconds DESC LIMIT 1"),
    ("Which artist has the most albums?", {"Artist", "Album"},
     "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
     "GROUP BY ar.ArtistId ORDER BY COUNT(*) DESC LIMIT 1"),
    ("Which country spends the most?", {"Invoice"},
     "SELECT BillingCountry FROM Invoice GROUP BY BillingCountry ORDER BY SUM(Total) DESC LIMIT 1"),
    ("Which sales agent sold the most in 2023?", {"Employee", "Customer", "Invoice"},
     "SELECT e.LastName FROM Employee e JOIN Customer c ON c.SupportRepId = e.EmployeeId "
     "JOIN Invoice i ON i.CustomerId = c.CustomerId WHERE strftime('%Y', i.InvoiceDate) = '2023' "
     "GROUP BY e.EmployeeId ORDER BY SUM(i.Total) DESC LIMIT 1"),
    ("How many tracks are in the 90's Music playlist?", {"Playlist", "PlaylistTrack"},
     "SELECT COUNT(*) FROM PlaylistTrack pt JOIN Playlist p ON p.PlaylistId = pt.PlaylistId "
     "WHERE p.Name = '90’s Music'"),  # the real name uses a CURLY apostrophe (’), not '
    ("How many different media types are there?", {"MediaType"},
     "SELECT COUNT(*) FROM MediaType"),
    ("How many customers bought Jazz tracks?", {"Customer", "Invoice", "InvoiceLine", "Track", "Genre"},
     "SELECT COUNT(DISTINCT i.CustomerId) FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
     "JOIN Track t ON t.TrackId = il.TrackId JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Jazz'"),
    ("What was the total revenue in 2022?", {"Invoice"},
     "SELECT ROUND(SUM(Total), 2) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2022'"),
    ("How many albums have a track longer than 10 minutes?", {"Album", "Track"},
     "SELECT COUNT(DISTINCT AlbumId) FROM Track WHERE Milliseconds > 600000"),
]


def expected_value(gold_sql: str) -> str:
    """The first value of the gold query's result, e.g. 'Rock' or '5'."""
    with get_engine().connect() as conn:
        return str(conn.execute(text(gold_sql)).scalar())


def first_row_values(sql: str) -> set[str]:
    """Every cell of the first result row, as text. Compared exactly, so '0' can't match '10'."""
    with get_engine().connect() as conn:
        row = conn.execute(text(sql)).first()
    return {str(v) for v in row} if row else set()


def main() -> None:
    db = get_db()
    full_chars = len(db.get_table_info())

    # ---------- 1. Retrieval: did we fetch every table the question needs? ----------
    print("=== 1. Table retrieval ===")
    table_hits, rag_chars = 0, []
    for question, needed, _ in GOLD:
        tables = find_tables(question)
        ok = needed <= set(tables)
        table_hits += ok
        rag_chars.append(len(db.get_table_info(tables)))
        print(f"{'✅' if ok else '❌'} {question:52} {len(tables):2} tables  missing={sorted(needed - set(tables))}")
    avg_rag = sum(rag_chars) / len(rag_chars)
    print(f"\nAll needed tables found: {table_hits}/{len(GOLD)}")
    print(f"Schema size: full = {full_chars} chars, schema RAG avg = {avg_rag:.0f} chars "
          f"({100 * (1 - avg_rag / full_chars):.0f}% smaller)")

    # ---------- 2. End to end: is the SQL's answer correct? ----------
    print("\n=== 2. SQL correctness: full schema (v0.2) vs schema RAG (v0.3) ===")
    scores = {}
    for label, rag in [("full schema", False), ("schema RAG", True)]:
        chain = build_sql_chain(schema_rag=rag)
        correct = 0
        for question, _, gold_sql in GOLD:
            expected = expected_value(gold_sql)
            sql = chain.invoke({"question": question})["sql_query"].sql
            ok, result = run_sql(sql) if sql else (False, "(no SQL)")  # safety check + readable result
            hit = ok and expected in first_row_values(sql)
            correct += hit
            if not hit:
                print(f"  ❌ [{label}] {question}\n       expected {expected!r} in: {result[:120]}")
        scores[label] = correct
    for label, correct in scores.items():
        print(f"{label:12}: {correct}/{len(GOLD)} correct")


if __name__ == "__main__":
    main()
