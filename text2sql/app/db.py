"""Everything that touches the database: connecting, safety checks, value checks, running queries."""
import difflib
from functools import lru_cache

import sqlglot
from langchain_community.utilities import SQLDatabase
from sqlalchemy import Engine, create_engine, inspect, text
from sqlglot import exp

from text2sql.app.config import DB_PATH

# Statement types that may never appear anywhere in a query (checked on every node of the parse tree).
WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)


@lru_cache
def get_engine() -> Engine:
    """SQLAlchemy engine for Chinook in READ-ONLY mode, so even a bad query can't change data."""
    return create_engine(f"sqlite:///file:{DB_PATH.as_posix()}?mode=ro&uri=true")


@lru_cache
def get_db() -> SQLDatabase:
    return SQLDatabase(get_engine())


@lru_cache
def get_foreign_keys() -> dict[str, list[str]]:
    """For each table, the tables it points to. Example: Track -> [Album, Genre, MediaType]."""
    inspector = inspect(get_engine())
    return {
        table: sorted({fk["referred_table"] for fk in inspector.get_foreign_keys(table)})
        for table in get_db().get_usable_table_names()
    }


@lru_cache
def get_columns() -> dict[str, set[str]]:
    """For each table, its column names. Example: Artist -> {ArtistId, Name}."""
    inspector = inspect(get_engine())
    return {
        table: {col["name"] for col in inspector.get_columns(table)}
        for table in get_db().get_usable_table_names()
    }


def check_sql(sql: str) -> str | None:
    """Return a reason if the SQL is not allowed, or None if it is a single read-only SELECT.

    v0.4: the query is PARSED (sqlglot) instead of searched for keywords, so it's an
    ALLOWLIST (only SELECT gets through) and LIKE '%update%' no longer causes a false alarm.
    """
    if not sql.strip():
        return "Empty query."
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        return f"SQL syntax error: {str(e).splitlines()[0]}"
    if len(statements) != 1:
        return "Only one statement is allowed."
    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union)):
        return f"Only SELECT queries are allowed (got {statement.key.upper()})."
    if any(isinstance(node, WRITE_NODES) for node in statement.walk()):
        return "Query contains a write operation."
    return None


def _string_comparisons(sql: str) -> list[tuple[str, str, str]]:
    """Find `column = 'text'` and `column IN ('a', 'b')` in a query → [(table, column, value)]."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    alias_to_table = {t.alias_or_name: t.name for t in tree.find_all(exp.Table)}
    tables_in_query = set(alias_to_table.values())
    columns = get_columns()

    def resolve(col: exp.Column) -> str | None:
        if col.table:                                          # e.g. P.Name → Playlist
            return alias_to_table.get(col.table)
        owners = [t for t in tables_in_query if col.name in columns.get(t, set())]
        return owners[0] if len(owners) == 1 else None         # unqualified, unambiguous only

    found = []
    for eq in tree.find_all(exp.EQ):
        col, lit = eq.left, eq.right
        if isinstance(col, exp.Literal):
            col, lit = lit, col
        if isinstance(col, exp.Column) and isinstance(lit, exp.Literal) and lit.is_string:
            if table := resolve(col):
                found.append((table, col.name, lit.this))
    for in_ in tree.find_all(exp.In):
        if isinstance(in_.this, exp.Column) and (table := resolve(in_.this)):
            for lit in in_.expressions:
                if isinstance(lit, exp.Literal) and lit.is_string:
                    found.append((table, in_.this.name, lit.this))
    return found


def check_values(sql: str) -> str | None:
    """Return feedback if the query compares a column to a text value that doesn't exist,
    but SIMILAR values do (e.g. '90's Music' vs the real '90’s Music'). Otherwise None.

    A value with no similar values (e.g. 'Atlantis') is fine: the honest answer is "no rows".
    """
    columns = get_columns()
    problems = []
    with get_engine().connect() as conn:
        for table, column, value in _string_comparisons(sql):
            if table not in columns or column not in columns[table]:
                continue
            q = f'SELECT 1 FROM "{table}" WHERE "{column}" = :v LIMIT 1'
            if conn.execute(text(q), {"v": value}).first():
                continue                                       # value exists
            distinct = f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT 5000'
            values = [str(r[0]) for r in conn.execute(text(distinct))]
            lowered = {v.lower(): v for v in values}
            close = difflib.get_close_matches(value.lower(), list(lowered), n=3, cutoff=0.6)
            if close:
                suggestions = ", ".join(repr(lowered[c]) for c in close)
                problems.append(f"{table}.{column} has no value {value!r}. Did you mean: {suggestions}?")
    return " ".join(problems) or None


def run_sql(sql: str) -> tuple[bool, str]:
    """Check and run a query. Returns (ok, rows_or_error). Never raises."""
    problem = check_sql(sql)
    if problem:
        return False, problem
    try:
        return True, get_db().run(sql.strip().rstrip(";")) or "(no rows)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
