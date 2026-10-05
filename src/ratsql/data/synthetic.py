"""A small synthetic, Spider-format Text-to-SQL dataset (for DEBUG mode and tests).

Five toy databases are generated deterministically (SQLite files, a
``tables.json`` and question/SQL pairs produced by ~20 templates covering
projection, aggregation, filtering with values, joins, GROUP BY / HAVING,
ORDER BY / LIMIT, nested sub-queries and set operations).  Three databases
are used for training and two *unseen* databases for dev, mimicking Spider's
cross-database setup.

The data is clearly synthetic and templated; it is only meant to exercise
the full pipeline quickly, never to report benchmark numbers.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

CITIES = ["Hyderabad", "Chennai", "Mumbai", "Delhi", "Pune", "Kolkata", "Boston", "Austin", "Denver", "Seattle"]
FIRST = ["Asha", "Ravi", "Meera", "Arjun", "Nina", "Omar", "Lena", "Kiran", "Maya", "Vikram", "Sara", "Tom", "Ivy", "Raj", "Zoe", "Dev", "Ana", "Leo"]
LAST = ["Rao", "Shah", "Iyer", "Khan", "Smith", "Lee", "Patel", "Gupta", "Brown", "Das", "Mehta", "Reddy"]


@dataclass
class Col:
    name: str  # SQL identifier
    nl: str  # natural-language name
    type: str  # text | number
    gen: str  # generator key
    pk: bool = False
    fk: tuple[str, str] | None = None  # (table, column)


@dataclass
class Tab:
    name: str
    nl: str
    nl_plural: str
    cols: list[Col]
    rows: int = 12
    name_col: str | None = None  # a human-readable "name/title" column
    cat_cols: list[str] = field(default_factory=list)  # categorical text columns
    num_cols: list[str] = field(default_factory=list)  # numeric (non-key) columns


def _dbs() -> dict[str, list[Tab]]:
    return {
        "company": [
            Tab("department", "department", "departments", [
                Col("dept_id", "department id", "number", "id", pk=True),
                Col("name", "name", "text", "dept_name"),
                Col("city", "city", "text", "city"),
                Col("budget", "budget", "number", "money"),
            ], rows=6, name_col="name", cat_cols=["city"], num_cols=["budget"]),
            Tab("employee", "employee", "employees", [
                Col("emp_id", "employee id", "number", "id", pk=True),
                Col("name", "name", "text", "person"),
                Col("age", "age", "number", "age"),
                Col("city", "city", "text", "city"),
                Col("salary", "salary", "number", "money"),
                Col("dept_id", "department id", "number", "fk", fk=("department", "dept_id")),
            ], rows=30, name_col="name", cat_cols=["city"], num_cols=["age", "salary"]),
            Tab("project", "project", "projects", [
                Col("proj_id", "project id", "number", "id", pk=True),
                Col("title", "title", "text", "project_title"),
                Col("budget", "budget", "number", "money"),
                Col("dept_id", "department id", "number", "fk", fk=("department", "dept_id")),
            ], rows=10, name_col="title", num_cols=["budget"]),
        ],
        "school": [
            Tab("student", "student", "students", [
                Col("student_id", "student id", "number", "id", pk=True),
                Col("name", "name", "text", "person"),
                Col("age", "age", "number", "student_age"),
                Col("major", "major", "text", "major"),
                Col("home_city", "home city", "text", "city"),
            ], rows=30, name_col="name", cat_cols=["major", "home_city"], num_cols=["age"]),
            Tab("course", "course", "courses", [
                Col("course_id", "course id", "number", "id", pk=True),
                Col("title", "title", "text", "course_title"),
                Col("credits", "credits", "number", "credits"),
                Col("department", "department", "text", "major"),
            ], rows=10, name_col="title", cat_cols=["department"], num_cols=["credits"]),
            Tab("enrollment", "enrollment", "enrollments", [
                Col("enroll_id", "enrollment id", "number", "id", pk=True),
                Col("student_id", "student id", "number", "fk", fk=("student", "student_id")),
                Col("course_id", "course id", "number", "fk", fk=("course", "course_id")),
                Col("grade", "grade", "number", "grade"),
            ], rows=40, num_cols=["grade"]),
        ],
        "store": [
            Tab("customer", "customer", "customers", [
                Col("customer_id", "customer id", "number", "id", pk=True),
                Col("name", "name", "text", "person"),
                Col("city", "city", "text", "city"),
                Col("age", "age", "number", "age"),
            ], rows=20, name_col="name", cat_cols=["city"], num_cols=["age"]),
            Tab("product", "product", "products", [
                Col("product_id", "product id", "number", "id", pk=True),
                Col("name", "name", "text", "product_name"),
                Col("category", "category", "text", "category"),
                Col("price", "price", "number", "price"),
            ], rows=15, name_col="name", cat_cols=["category"], num_cols=["price"]),
            Tab("orders", "order", "orders", [
                Col("order_id", "order id", "number", "id", pk=True),
                Col("customer_id", "customer id", "number", "fk", fk=("customer", "customer_id")),
                Col("product_id", "product id", "number", "fk", fk=("product", "product_id")),
                Col("quantity", "quantity", "number", "quantity"),
            ], rows=40, num_cols=["quantity"]),
        ],
        "library": [
            Tab("author", "author", "authors", [
                Col("author_id", "author id", "number", "id", pk=True),
                Col("name", "name", "text", "person"),
                Col("country", "country", "text", "country"),
                Col("birth_year", "birth year", "number", "year"),
            ], rows=12, name_col="name", cat_cols=["country"], num_cols=["birth_year"]),
            Tab("book", "book", "books", [
                Col("book_id", "book id", "number", "id", pk=True),
                Col("title", "title", "text", "book_title"),
                Col("genre", "genre", "text", "genre"),
                Col("pages", "pages", "number", "pages"),
                Col("author_id", "author id", "number", "fk", fk=("author", "author_id")),
            ], rows=25, name_col="title", cat_cols=["genre"], num_cols=["pages"]),
        ],
        "hospital": [
            Tab("doctor", "doctor", "doctors", [
                Col("doctor_id", "doctor id", "number", "id", pk=True),
                Col("name", "name", "text", "person"),
                Col("specialty", "specialty", "text", "specialty"),
                Col("city", "city", "text", "city"),
                Col("experience", "experience", "number", "experience"),
            ], rows=12, name_col="name", cat_cols=["specialty", "city"], num_cols=["experience"]),
            Tab("patient", "patient", "patients", [
                Col("patient_id", "patient id", "number", "id", pk=True),
                Col("name", "name", "text", "person"),
                Col("age", "age", "number", "age"),
                Col("doctor_id", "doctor id", "number", "fk", fk=("doctor", "doctor_id")),
            ], rows=30, name_col="name", num_cols=["age"]),
        ],
    }


TRAIN_DBS = ("company", "school", "store")
DEV_DBS = ("library", "hospital")

_POOLS = {
    "dept_name": ["Research", "Sales", "Finance", "Marketing", "Support", "Operations"],
    "project_title": ["Apollo", "Beacon", "Comet", "Delta", "Echo", "Falcon", "Gemini", "Horizon", "Iris", "Juno"],
    "major": ["Physics", "Biology", "History", "Economics", "Chemistry"],
    "course_title": ["Algebra", "Genetics", "Mechanics", "Poetry", "Statistics", "Databases", "Ethics", "Optics", "Geology", "Rhetoric"],
    "product_name": ["Laptop", "Phone", "Tablet", "Camera", "Monitor", "Keyboard", "Mouse", "Printer", "Speaker", "Router", "Charger", "Headset", "Webcam", "Scanner", "Drone"],
    "category": ["Electronics", "Office", "Audio", "Accessories"],
    "country": ["India", "France", "Japan", "Brazil", "Canada"],
    "book_title": ["Silent River", "Blue Harbor", "Iron Garden", "Paper Moon", "Golden Road", "Hidden Lake", "Winter Song", "Red Desert", "Glass Tower", "Wild Coast",
                   "Night Train", "Stone Bridge", "Open Sky", "Last Letter", "Deep Forest", "Bright Star", "Quiet Street", "Lost City", "Broken Clock", "White Sail",
                   "Dark Water", "Green Valley", "Old Castle", "Long Journey", "First Light"],
    "genre": ["Fantasy", "Mystery", "Romance", "Science", "History"],
    "specialty": ["Cardiology", "Neurology", "Oncology", "Pediatrics"],
}


def _gen_value(col: Col, i: int, rng: random.Random, fk_sizes: dict) -> object:
    g = col.gen
    if g == "id":
        return i + 1
    if g == "fk":
        return rng.randint(1, fk_sizes[col.fk[0]])
    if g == "person":
        return f"{FIRST[(i * 7 + 3) % len(FIRST)]} {LAST[(i * 5 + 1) % len(LAST)]}"
    if g == "city":
        return CITIES[rng.randrange(len(CITIES))]
    if g in _POOLS:
        pool = _POOLS[g]
        return pool[i % len(pool)] if g in ("dept_name", "project_title", "course_title", "product_name", "book_title") else pool[rng.randrange(len(pool))]
    if g == "money":
        return rng.randrange(10, 200) * 1000
    if g == "age":
        return rng.randint(22, 65)
    if g == "student_age":
        return rng.randint(18, 26)
    if g == "credits":
        return rng.choice([2, 3, 4, 5])
    if g == "grade":
        return rng.randint(50, 100)
    if g == "price":
        return rng.randrange(5, 150) * 10
    if g == "quantity":
        return rng.randint(1, 9)
    if g == "year":
        return rng.randint(1920, 1995)
    if g == "pages":
        return rng.randrange(10, 80) * 10
    if g == "experience":
        return rng.randint(1, 35)
    raise ValueError(g)


def _create_db(path: Path, tables: list[Tab], rng: random.Random) -> dict[str, list[dict]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    data: dict[str, list[dict]] = {}
    sizes = {t.name: t.rows for t in tables}
    for t in tables:
        cols_sql = []
        for c in t.cols:
            ctype = "TEXT" if c.type == "text" else "INTEGER"
            s = f"{c.name} {ctype}"
            if c.pk:
                s += " PRIMARY KEY"
            cols_sql.append(s)
        for c in t.cols:
            if c.fk:
                cols_sql.append(f"FOREIGN KEY ({c.name}) REFERENCES {c.fk[0]}({c.fk[1]})")
        conn.execute(f"CREATE TABLE {t.name} ({', '.join(cols_sql)})")
        rows = []
        for i in range(t.rows):
            row = {c.name: _gen_value(c, i, rng, sizes) for c in t.cols}
            rows.append(row)
            conn.execute(
                f"INSERT INTO {t.name} ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", list(row.values())
            )
        data[t.name] = rows
    conn.commit()
    conn.close()
    return data


def _tables_json(db_id: str, tables: list[Tab]) -> dict:
    column_names = [[-1, "*"]]
    column_names_original = [[-1, "*"]]
    column_types = ["text"]
    pks, fks = [], []
    index = {}
    for ti, t in enumerate(tables):
        for c in t.cols:
            index[(t.name, c.name)] = len(column_names)
            column_names.append([ti, c.nl])
            column_names_original.append([ti, c.name])
            column_types.append(c.type)
            if c.pk:
                pks.append(index[(t.name, c.name)])
    for t in tables:
        for c in t.cols:
            if c.fk:
                fks.append([index[(t.name, c.name)], index[c.fk]])
    return {
        "db_id": db_id,
        "table_names": [t.nl for t in tables],
        "table_names_original": [t.name for t in tables],
        "column_names": column_names,
        "column_names_original": column_names_original,
        "column_types": column_types,
        "primary_keys": pks,
        "foreign_keys": fks,
    }


def _templates(db_id: str, tables: list[Tab], data: dict[str, list[dict]], rng: random.Random) -> list[tuple[str, str]]:
    """Generate (question, sql) pairs from templates using real cell values."""
    out: list[tuple[str, str]] = []
    by_name = {t.name: t for t in tables}
    col_nl = {(t.name, c.name): c.nl for t in tables for c in t.cols}

    def vals(t: Tab, c: str) -> list:
        vs = sorted({r[c] for r in data[t.name]}, key=str)
        # put the running example value first so that "Which employees work in Hyderabad?" always exists
        return [v for v in vs if v == "Hyderabad"] + [v for v in vs if v != "Hyderabad"]

    for t in tables:
        P, S = t.nl_plural, t.nl
        out += [
            (f"How many {P} are there?", f"SELECT count(*) FROM {t.name}"),
            (f"Count the number of {P}.", f"SELECT count(*) FROM {t.name}"),
        ]
        if t.name_col:
            n = t.name_col
            out += [
                (f"List the {col_nl[(t.name, n)]} of all {P}.", f"SELECT {n} FROM {t.name}"),
                (f"What are the {col_nl[(t.name, n)]}s of the {P}?", f"SELECT {n} FROM {t.name}"),
            ]
            for c in t.cat_cols:
                cn = col_nl[(t.name, c)]
                for v in vals(t, c)[:3]:
                    out += [
                        (f"Which {P} have {cn} {v}?", f"SELECT {n} FROM {t.name} WHERE {c} = '{v}'"),
                        (f"Find the {col_nl[(t.name, n)]} of {P} whose {cn} is {v}.", f"SELECT {n} FROM {t.name} WHERE {c} = '{v}'"),
                        (f"How many {P} have {cn} {v}?", f"SELECT count(*) FROM {t.name} WHERE {c} = '{v}'"),
                    ]
                    if c in ("city", "home_city"):
                        verb = {"employee": "work", "doctor": "practice"}.get(t.name, "live")
                        out.append((f"Which {P} {verb} in {v}?", f"SELECT {n} FROM {t.name} WHERE {c} = '{v}'"))
                out += [
                    (f"How many {P} are there for each {cn}?", f"SELECT {c}, count(*) FROM {t.name} GROUP BY {c}"),
                    (f"Show each {cn} and the number of {P}.", f"SELECT {c}, count(*) FROM {t.name} GROUP BY {c}"),
                    (f"Which {cn} has the most {P}?", f"SELECT {c} FROM {t.name} GROUP BY {c} ORDER BY count(*) DESC LIMIT 1"),
                    (f"List the distinct {cn}s of {P}.", f"SELECT DISTINCT {c} FROM {t.name}"),
                ]
                k = rng.choice([1, 2, 3])
                out.append((f"Which {cn}s have more than {k} {P}?", f"SELECT {c} FROM {t.name} GROUP BY {c} HAVING count(*) > {k}"))
            for c in t.num_cols:
                cn = col_nl[(t.name, c)]
                values = vals(t, c)
                thr = values[len(values) // 2]
                out += [
                    (f"What is the average {cn} of {P}?", f"SELECT avg({c}) FROM {t.name}"),
                    (f"What is the maximum {cn} of all {P}?", f"SELECT max({c}) FROM {t.name}"),
                    (f"What is the minimum {cn} of {P}?", f"SELECT min({c}) FROM {t.name}"),
                    (f"What is the total {cn} of all {P}?", f"SELECT sum({c}) FROM {t.name}"),
                    (f"Find {P} with {cn} greater than {thr}.", f"SELECT {n} FROM {t.name} WHERE {c} > {thr}"),
                    (f"List the {col_nl[(t.name, n)]} of {P} with {cn} less than {thr}.", f"SELECT {n} FROM {t.name} WHERE {c} < {thr}"),
                    (f"Show the {col_nl[(t.name, n)]} of the {S} with the highest {cn}.", f"SELECT {n} FROM {t.name} ORDER BY {c} DESC LIMIT 1"),
                    (f"Which {S} has the lowest {cn}?", f"SELECT {n} FROM {t.name} ORDER BY {c} ASC LIMIT 1"),
                    (f"List the {col_nl[(t.name, n)]} of {P} sorted by {cn}.", f"SELECT {n} FROM {t.name} ORDER BY {c} ASC"),
                    (f"List {P} in descending order of {cn}.", f"SELECT {n} FROM {t.name} ORDER BY {c} DESC"),
                    (f"Find the {col_nl[(t.name, n)]} of {P} whose {cn} is above the average.",
                     f"SELECT {n} FROM {t.name} WHERE {c} > (SELECT avg({c}) FROM {t.name})"),
                    (f"Show the {col_nl[(t.name, n)]} and {cn} of all {P}.", f"SELECT {n}, {c} FROM {t.name}"),
                    (f"Find {P} with {cn} between {values[0]} and {thr}.", f"SELECT {n} FROM {t.name} WHERE {c} BETWEEN {values[0]} AND {thr}"),
                    (f"What is the top 3 {P} with the largest {cn}?", f"SELECT {n} FROM {t.name} ORDER BY {c} DESC LIMIT 3"),
                ]
                for cc in t.cat_cols[:1]:
                    ccn = col_nl[(t.name, cc)]
                    out.append((f"What is the average {cn} of {P} for each {ccn}?", f"SELECT {cc}, avg({c}) FROM {t.name} GROUP BY {cc}"))
    # joins through foreign keys
    for t in tables:
        for c in t.cols:
            if not c.fk:
                continue
            parent = by_name[c.fk[0]]
            if not (t.name_col and parent.name_col):
                continue
            tn, pn = t.name_col, parent.name_col
            out += [
                (f"Show the {col_nl[(t.name, tn)]} of each {t.nl} and the {col_nl[(parent.name, pn)]} of its {parent.nl}.",
                 f"SELECT T1.{tn}, T2.{pn} FROM {t.name} AS T1 JOIN {parent.name} AS T2 ON T1.{c.name} = T2.{c.fk[1]}"),
                (f"How many {t.nl_plural} does each {parent.nl} have?",
                 f"SELECT T2.{pn}, count(*) FROM {t.name} AS T1 JOIN {parent.name} AS T2 ON T1.{c.name} = T2.{c.fk[1]} GROUP BY T2.{pn}"),
                (f"Which {parent.nl} has the most {t.nl_plural}?",
                 f"SELECT T2.{pn} FROM {t.name} AS T1 JOIN {parent.name} AS T2 ON T1.{c.name} = T2.{c.fk[1]} GROUP BY T1.{c.name} ORDER BY count(*) DESC LIMIT 1"),
                (f"Find the {parent.nl_plural} that have no {t.nl_plural}.",
                 f"SELECT {pn} FROM {parent.name} WHERE {c.fk[1]} NOT IN (SELECT {c.name} FROM {t.name})"),
            ]
            for v in vals(parent, pn)[:3]:
                out.append((f"List the {col_nl[(t.name, tn)]} of {t.nl_plural} of the {parent.nl} {v}.",
                            f"SELECT T1.{tn} FROM {t.name} AS T1 JOIN {parent.name} AS T2 ON T1.{c.name} = T2.{c.fk[1]} WHERE T2.{pn} = '{v}'"))
            for pc in parent.cat_cols[:1]:
                for v in vals(parent, pc)[:2]:
                    out.append((f"Find the {col_nl[(t.name, tn)]} of {t.nl_plural} whose {parent.nl} has {col_nl[(parent.name, pc)]} {v}.",
                                f"SELECT T1.{tn} FROM {t.name} AS T1 JOIN {parent.name} AS T2 ON T1.{c.name} = T2.{c.fk[1]} WHERE T2.{pc} = '{v}'"))
    # set operations over shared categorical columns
    cats = [(t, c) for t in tables for c in t.cat_cols]
    for i, (t1, c1) in enumerate(cats):
        for t2, c2 in cats[i + 1 :]:
            if t1.name != t2.name and col_nl[(t1.name, c1)] == col_nl[(t2.name, c2)]:
                cn = col_nl[(t1.name, c1)]
                out += [
                    (f"Which {cn}s have both {t1.nl_plural} and {t2.nl_plural}?", f"SELECT {c1} FROM {t1.name} INTERSECT SELECT {c2} FROM {t2.name}"),
                    (f"Which {cn}s have {t1.nl_plural} but no {t2.nl_plural}?", f"SELECT {c1} FROM {t1.name} EXCEPT SELECT {c2} FROM {t2.name}"),
                    (f"List all {cn}s of {t1.nl_plural} or {t2.nl_plural}.", f"SELECT {c1} FROM {t1.name} UNION SELECT {c2} FROM {t2.name}"),
                ]
    return out


def generate_synthetic_dataset(out_dir: str | Path, seed: int = 13) -> dict:
    """Create ``tables.json``, ``train.json``, ``dev.json`` and SQLite DBs under ``out_dir``."""
    from ratsql.schema.schema import Schema
    from ratsql.sql.parser import parse_sql, to_jsonable

    out_dir = Path(out_dir)
    rng = random.Random(seed)
    dbs = _dbs()
    tables_json = []
    splits: dict[str, list[dict]] = {"train": [], "dev": []}
    for db_id, tables in dbs.items():
        data = _create_db(out_dir / "database" / db_id / f"{db_id}.sqlite", tables, rng)
        entry = _tables_json(db_id, tables)
        tables_json.append(entry)
        schema = Schema.from_spider(entry)
        split = "train" if db_id in TRAIN_DBS else "dev"
        pairs = _templates(db_id, tables, data, rng)
        seen = set()
        for q, sql in pairs:
            if q in seen:
                continue
            seen.add(q)
            parsed = parse_sql(sql, schema)
            splits[split].append({"db_id": db_id, "question": q, "query": sql, "sql": to_jsonable(parsed)})
    for split in splits:
        rng.shuffle(splits[split])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tables.json", "w", encoding="utf-8") as f:
        json.dump(tables_json, f, indent=1)
    for split, rows in splits.items():
        with open(out_dir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return {"databases": len(dbs), **{s: len(r) for s, r in splits.items()}}
