"""Evaluate the self-correcting graph (v0.4) against the plain chain (v0.3).

Run from the Gen_AI folder:   python -m text2sql.evals.eval_graph
"""
from text2sql.app.chains import build_sql_chain
from text2sql.app.db import check_sql
from text2sql.app.graph import build_sql_graph
from text2sql.evals.eval_schema_rag import GOLD, expected_value, first_row_values

# Extra questions where the user types a VALUE slightly differently from the database.
# These are exactly what v0.4's value check + retry loop is built for.
MESSY_VALUES = [
    ("How many customers are from Brasil?", set(),
     "SELECT COUNT(*) FROM Customer WHERE Country = 'Brazil'"),
    ("How many albums does ACDC have?", set(),
     "SELECT COUNT(*) FROM Album al JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'AC/DC'"),
    ("How many tracks are in the 90s Music playlist?", set(),
     "SELECT COUNT(*) FROM PlaylistTrack pt JOIN Playlist p ON p.PlaylistId = pt.PlaylistId "
     "WHERE p.Name = '90’s Music'"),
]


def is_correct(sql: str, expected: str) -> bool:
    return bool(sql) and check_sql(sql) is None and expected in first_row_values(sql)


def main() -> None:
    questions = GOLD + MESSY_VALUES
    chain = build_sql_chain()          # v0.3: one shot
    graph = build_sql_graph()          # v0.4: validate + retry
    scores = {"v0.3 chain": 0, "v0.4 graph": 0}
    total_fixes = 0

    for question, _, gold_sql in questions:
        expected = expected_value(gold_sql)

        chain_sql = chain.invoke({"question": question})["sql_query"].sql
        chain_ok = is_correct(chain_sql, expected)

        out = graph.invoke({"question": question})
        graph_sql = out["sql_query"].sql if out.get("sql_query") else ""
        graph_ok = is_correct(graph_sql, expected)
        fixes = out.get("fix_log") or []
        total_fixes += len(fixes)

        scores["v0.3 chain"] += chain_ok
        scores["v0.4 graph"] += graph_ok
        print(f"{'✅' if chain_ok else '❌'} {'✅' if graph_ok else '❌'}  {question:52} "
              f"attempts={out.get('attempts')}")
        for problem in fixes:
            print(f"        🔧 fixed: {problem}")

    print(f"\n(first mark = v0.3 chain, second = v0.4 graph, expected answer checked exactly)")
    for label, score in scores.items():
        print(f"{label}: {score}/{len(questions)} correct")
    print(f"v0.4 problems caught and fixed by the loop: {total_fixes}")


if __name__ == "__main__":
    main()
