"""Hold-out evaluation (v1.0): 24 NEW questions the system was never tuned on.

The earlier sets (eval_schema_rag.py, eval_graph.py) were partly used while choosing the table
descriptions and the retry message, so their scores may be optimistic. This set is the honest test.

Run from the Gen_AI folder:   python -m text2sql.evals.eval_holdout
Results are also saved to text2sql/evals/results/holdout_<date>.json
"""
import json
import time
from datetime import date
from pathlib import Path

from sqlalchemy import text

from text2sql.app.chains import build_sql_chain
from text2sql.app.db import check_sql, get_engine
from text2sql.app.graph import build_sql_graph

UNANSWERABLE = None   # marker: the right behaviour is to DECLINE (is_answerable = False)

# (category, question, gold SQL or UNANSWERABLE). Every gold answer was checked against the
# database, and ranking questions were checked for ties (a 3-way tie for "city with the most
# customers" was dropped).
HOLDOUT = [
    ("simple",      "How many employees work at the store?", "SELECT COUNT(*) FROM Employee"),
    ("simple",      "What is the highest price of any track?", "SELECT MAX(UnitPrice) FROM Track"),
    ("simple",      "What is the average invoice total?", "SELECT ROUND(AVG(Total), 2) FROM Invoice"),
    ("join",        "How many tracks are in the Rock genre?",
     "SELECT COUNT(*) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Rock'"),
    ("join",        "How many albums does Iron Maiden have?",
     "SELECT COUNT(*) FROM Album al JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = 'Iron Maiden'"),
    ("join",        "How many albums does Metallica have?",
     "SELECT COUNT(*) FROM Album al JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = 'Metallica'"),
    ("join",        "What genre is the song Smells Like Teen Spirit?",
     "SELECT g.Name FROM Track t JOIN Genre g ON g.GenreId = t.GenreId WHERE t.Name = 'Smells Like Teen Spirit'"),
    ("join",        "How many tracks are in the Grunge playlist?",
     "SELECT COUNT(*) FROM PlaylistTrack pt JOIN Playlist p ON p.PlaylistId = pt.PlaylistId WHERE p.Name = 'Grunge'"),
    ("join",        "How many customers does Steve Johnson support?",
     "SELECT COUNT(*) FROM Customer c JOIN Employee e ON e.EmployeeId = c.SupportRepId "
     "WHERE e.FirstName = 'Steve' AND e.LastName = 'Johnson'"),
    ("join",        "How many Classical tracks are there?",
     "SELECT COUNT(*) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Classical'"),
    ("aggregation", "What is the total revenue from Germany?",
     "SELECT ROUND(SUM(Total), 2) FROM Invoice WHERE BillingCountry = 'Germany'"),
    ("aggregation", "How many Jazz tracks are longer than 6 minutes?",
     "SELECT COUNT(*) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId "
     "WHERE g.Name = 'Jazz' AND t.Milliseconds > 360000"),
    ("aggregation", "How many albums have more than 20 tracks?",
     "SELECT COUNT(*) FROM (SELECT AlbumId FROM Track GROUP BY AlbumId HAVING COUNT(*) > 20)"),
    ("ranking",     "Which media type has the most tracks?",
     "SELECT m.Name FROM Track t JOIN MediaType m ON m.MediaTypeId = t.MediaTypeId "
     "GROUP BY m.MediaTypeId ORDER BY COUNT(*) DESC LIMIT 1"),
    ("ranking",     "Who is the youngest employee?",
     "SELECT FirstName FROM Employee ORDER BY BirthDate DESC LIMIT 1"),
    ("ranking",     "Which composer wrote the most tracks?",
     "SELECT Composer FROM Track WHERE Composer IS NOT NULL GROUP BY Composer ORDER BY COUNT(*) DESC LIMIT 1"),
    ("ranking",     "Which customer has spent the most money?",
     "SELECT c.FirstName FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
     "GROUP BY c.CustomerId ORDER BY SUM(i.Total) DESC LIMIT 1"),
    ("ranking",     "Which country has the highest average invoice total?",
     "SELECT BillingCountry FROM Invoice GROUP BY BillingCountry ORDER BY AVG(Total) DESC LIMIT 1"),
    ("date",        "How many invoices were issued in 2024?",
     "SELECT COUNT(*) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2024'"),
    ("date",        "Which year had the highest total sales?",
     "SELECT strftime('%Y', InvoiceDate) FROM Invoice GROUP BY 1 ORDER BY SUM(Total) DESC LIMIT 1"),
    ("messy value", "how many customers are from the united kingdom?",
     "SELECT COUNT(*) FROM Customer WHERE Country = 'United Kingdom'"),
    ("messy value", "how many tracks does led zeppelin have?",
     "SELECT COUNT(*) FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId "
     "JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = 'Led Zeppelin'"),
    ("unanswerable", "What is the profit margin on each track?", UNANSWERABLE),
    ("unanswerable", "Which customers left a product review?", UNANSWERABLE),
]


def scalar(sql: str):
    with get_engine().connect() as conn:
        return conn.execute(text(sql)).scalar()


def first_row(sql: str) -> list:
    with get_engine().connect() as conn:
        row = conn.execute(text(sql.strip().rstrip(";"))).first()
    return list(row) if row else []


def matches(expected, row: list) -> bool:
    """Numbers: equal within rounding (5.65 vs 5.6533). Text: contained, ignoring case
    ('Jane' in 'Jane Peacock'). Fair to different-but-correct SQL."""
    for cell in row:
        if isinstance(expected, (int, float)) and isinstance(cell, (int, float)):
            if abs(cell - expected) <= max(0.01, abs(expected) * 0.001):
                return True
        elif isinstance(expected, str) and cell is not None and expected.lower() in str(cell).lower():
            return True
    return False


def grade(query, gold_sql) -> bool:
    """Was this SQLQuery a correct response to the question?"""
    if gold_sql is UNANSWERABLE:
        return query is not None and not query.is_answerable
    if query is None or not query.is_answerable or not query.sql or check_sql(query.sql):
        return False
    try:
        return matches(scalar(gold_sql), first_row(query.sql))
    except Exception:
        return False


def main() -> None:
    chain, graph = build_sql_chain(), build_sql_graph()
    rows, totals = [], {"v0.3 chain": 0, "v0.4+ graph": 0}
    by_category: dict[str, list[int]] = {}

    for category, question, gold_sql in HOLDOUT:
        t0 = time.time()
        chain_query = chain.invoke({"question": question})["sql_query"]
        t1 = time.time()
        out = graph.invoke({"question": question})
        t2 = time.time()
        chain_ok, graph_ok = grade(chain_query, gold_sql), grade(out.get("sql_query"), gold_sql)
        totals["v0.3 chain"] += chain_ok
        totals["v0.4+ graph"] += graph_ok
        by_category.setdefault(category, [0, 0, 0])
        by_category[category][0] += chain_ok
        by_category[category][1] += graph_ok
        by_category[category][2] += 1
        print(f"{'✅' if chain_ok else '❌'} {'✅' if graph_ok else '❌'}  [{category:12}] {question:55} "
              f"attempts={out.get('attempts')}  {t2 - t1:4.1f}s")
        for fix in out.get("fix_log") or []:
            print(f"        🔧 {fix}")
        rows.append({"category": category, "question": question, "chain_correct": chain_ok,
                     "graph_correct": graph_ok, "graph_attempts": out.get("attempts"),
                     "graph_fixes": out.get("fix_log") or [],
                     "graph_sql": out["sql_query"].sql if out.get("sql_query") else "",
                     "chain_seconds": round(t1 - t0, 1), "graph_seconds": round(t2 - t1, 1)})

    n = len(HOLDOUT)
    print("\n(first mark = v0.3 chain, second = v0.4+ graph)")
    print(f"{'category':14} {'chain':>7} {'graph':>7}")
    for category, (c, g, total) in by_category.items():
        print(f"{category:14} {c:>3}/{total:<3} {g:>3}/{total:<3}")
    for label, score in totals.items():
        print(f"{label:12}: {score}/{n} correct ({100 * score / n:.0f}%)")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"holdout_{date.today()}.json"
    path.write_text(json.dumps({"date": str(date.today()), "totals": totals, "n": n,
                                "by_category": by_category, "questions": rows}, indent=2),
                    encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
