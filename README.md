# YoDb

**One interface for querying and retrieving data across heterogeneous sources.**

YoDb is an experimental query and retrieval engine designed to help AI agents interact with data through a unified interface.

Today, YoDb queries across PostgreSQL databases: relational tables, vector embeddings
(pgvector) and graph edges (Apache AGE), with one typed query model, a cost-based planner
and a read-only executor.

The longer-term vision is to extend YoDb beyond PostgreSQL to heterogeneous data sources, including other database systems and visual information such as images and videos.

**Current capabilities**
- PostgreSQL-backed querying across several databases (one dataset, several tables)
- Joins over declared relationships: field equality, or edges of an Apache AGE graph
- Similarity search over pgvector embeddings, and a model-judged semantic filter
- MCP integration for AI agents

**Planned directions**
- Additional database connectors
- Extensible source-specific query and retrieval operators
- Visual and multimodal retrieval

*YoDb is in active development. PostgreSQL (with the pgvector and Apache AGE extensions) is currently the supported backend; additional sources and visual retrieval are not yet implemented.*

---

YoDb lets an app or an AI agent ask one question about data that lives in
several PostgreSQL databases. You do not write SQL, and your data is not copied
anywhere.

## The problem

A company's data is rarely in one place. Customers are in one database, their
billing plan in a second, their support tickets in a third. A question like
"which UK customers are on the pro plan?" needs two of them, and no single SQL
query can reach both.

Today you either copy everything into one warehouse first, or you write code
that queries each database and stitches the answers together. If an AI agent is
asking, it has to guess the table names and the join keys, and a wrong guess
gives a confident wrong answer.

## What YoDb does

You describe your data once, in three small YAML files. After that, YoDb treats
the three databases as one set of named datasets:

```text
$ yodb catalog catalog/

customer  - A customer of the shop.
  id         id          from crm
  name       string      from crm
  country    string      from crm
  plan       string      from billing
  signed_up  timestamp   from crm

ticket  - A support ticket.
  id           id        from support
  customer_id  id        from support
  subject      string    from support
  status       string    from support
  priority     int       from support
```

`customer` looks like one table, but `name` and `country` come from the `crm`
database and `plan` comes from `billing`. Now ask a question:

```bash
yodb query catalog/ '{
  "from": {"dataset": "customer"},
  "select": ["name", "country", "plan"],
  "where": {"field": "country", "op": "eq", "value": "UK"},
  "order_by": [{"field": "name", "direction": "asc"}]
}'
```

```text
id   name             country  plan
---  ---------------  -------  ----
c01  Ada Lovelace     UK       pro
c03  Alan Turing      UK       free
c10  Tim Berners-Lee  UK       pro
(3 rows)
```

YoDb asked `crm` for the UK customers, asked `billing` for the plans of those
three customers only, and put the rows together.

You can also follow a relationship between two datasets. This finds US
customers with open support tickets, most urgent first. The customers are in one
database and the tickets in another:

```bash
yodb query catalog/ '{
  "from": {"dataset": "customer"},
  "select": ["name"],
  "where": {"field": "country", "op": "eq", "value": "US"},
  "traverse": [{
    "relationship": "customer_has_ticket", "as": "ticket",
    "select": ["subject", "priority"],
    "where": {"field": "status", "op": "eq", "value": "open"}
  }],
  "order_by": [{"field": "ticket.priority", "direction": "desc"}, {"field": "name", "direction": "asc"}],
  "page": {"first": 3}
}'
```

```text
id   name               ticket.id  ticket.subject           ticket.priority
---  -----------------  ---------  -----------------------  ---------------
c07  Barbara Liskov     t11        Two-factor not working   5
c02  Grace Hopper       t03        App crashes on export    5
c05  Margaret Hamilton  t08        Data missing after sync  5
(3 rows)
```

## How it works

1. **You describe the data.** Three YAML files say which datasets exist, which
   table and column each field comes from, and how datasets relate. This is the
   *catalog*. YoDb checks it against the real databases and refuses to start if
   they do not match.
2. **You ask in terms of the catalog.** A query is a small piece of JSON that
   names datasets and fields. It never names a table, and it is never SQL.
3. **YoDb works out the rest.** It decides which databases to read and in what
   order, sends each one a filtered query, and joins the answers. `yodb explain`
   shows the plan without running it.

## Try it in five minutes

The sample in [examples/shop](examples/shop) is the three-database shop used
above. You need Docker and Python 3.11 or newer.

```bash
pip install .                # from a checkout of this repository
cd examples/shop
./setup.sh                   # starts PostgreSQL and prints three export lines; paste them
yodb validate catalog/       # checks the catalog against the databases
yodb query catalog/ '{"from": {"dataset": "customer"}, "select": ["name", "plan"]}'
```

[examples/shop/README.md](examples/shop/README.md) has more queries to try.
[examples/yelp](examples/yelp) loads a sample of the Yelp Open Dataset with review
embeddings and a graph of friendships, reviews and categories: a data layer spanning all three kinds of data.
[benchmark](benchmark) holds the 240-question evaluation over it.

## Use it from Python

```python
import yodb

with yodb.connect("catalog/") as db:
    result = db.query({
        "from": {"dataset": "customer"},
        "select": ["name", "plan"],
        "where": {"field": "country", "op": "eq", "value": "UK"},
    })
    for row in result.rows:
        print(row["name"], row["plan"])
```

Connection strings stay out of the catalog. Each database is named by a
`connection_ref`, and YoDb reads its connection string from the environment
variable `YODB_CONN_<REF>`.

## Give it to an AI agent

```bash
pip install ".[mcp]"
yodb mcp catalog/
```

This serves the catalog over the Model Context Protocol (MCP), the standard way
to give an AI agent tools. The agent gets three tools: list the datasets, run a
query, and explain a query. It sees dataset and field names only. It cannot send
SQL, see table names, or write anything. See
[docs/reference/mcp.mdx](docs/reference/mcp.mdx).

## Why not let the agent write SQL?

Because a mistake in SQL is silent. A wrong join key or a wrong table still
returns rows. With YoDb:

- **Joins are declared by you, once.** The agent picks a relationship by name.
  It cannot invent one.
- **A mistake is refused, with a reason.** A misspelt field gives
  `error [field_not_found] at select[0]: Unknown field 'nam' on dataset
  'customer'.` SQL text is refused outright.
- **It cannot write.** Every database session is read-only.
- **It cannot run away.** Each query has a time limit and row limits. A query
  that would read too much is stopped and refused.
- **Nothing leaks.** Errors never contain passwords, SQL or data values.

## Searching by similarity, filtering by meaning

With embeddings in a pgvector column, a condition can rank records by how close
their text is to a text you give. This keeps the 20 open tickets nearest to a
late-delivery complaint, nearest first, with each row's distance in the result:

```json
{"all": [
  {"field": "status", "op": "eq", "value": "open"},
  {"similar": {"field": "body", "text": "the delivery arrived late and damaged", "top": 20}}
]}
```

YoDb ships an embedding provider over an open local model
(`yodb.embeddings.FastEmbedProvider`, `pip install ".[semantic]"`). See
[docs/query/similarity-search.mdx](docs/query/similarity-search.mdx).

A filter can also be a sentence that a model you supply judges for every row:

```json
{"semantic": {"field": "body", "proposition": "the customer is asking about a late delivery"}}
```

YoDb can first narrow the candidates with a vector search, and it enforces a cost
and time budget. See [docs/query/semantic-filter.mdx](docs/query/semantic-filter.mdx).

## Following graph edges

A relationship that has no matching column, such as who referred whom, can be
stored as edges in an Apache AGE graph inside PostgreSQL. Declare the graph on
the source and the edge type on the relationship; the same `traverse` step then
follows the edges, and the rows still come from their tables. See
[docs/catalog/graphs.mdx](docs/catalog/graphs.mdx).

## What it cannot do yet

YoDb is at version 0.1. It is tested, but deliberately small:

- **PostgreSQL only.** No other database, and no SaaS APIs. Graph edges need the
  Apache AGE extension, embeddings the pgvector extension.
- **No counting or totals.** There is no `count`, `sum` or `group by`.
- **One relationship per query.** You can follow one `traverse` step (one hop, also over graph edges).
- **One page of results.** Up to 500 rows per query, with no "next page".
- **Read-only.** There is no way to write data.
- **Bounded size.** Rows from different databases are joined in memory. A query
  that would need more than the row limits is refused.

The full list is in [docs/reference/limits.mdx](docs/reference/limits.mdx). Read
it before you rely on a query shape.

## When to use something else

- **Your data is in one database.** Use SQL.
- **You need reports, counts and totals over large data.** Use a data warehouse,
  or a federated SQL engine such as Trino.
- **Your data is in other systems** (MySQL, MongoDB, Salesforce). YoDb cannot
  read them yet.

## What you can rely on

[docs/reference/compatibility.mdx](docs/reference/compatibility.mdx) lists which
parts are stable (the query format, the catalog files, the error codes, the
`yodb` command) and which may still change.

## Documentation

- [Quickstart](docs/getting-started/quickstart.mdx): the sample shop, step by step
- [Concepts](docs/getting-started/overview.mdx): the handful of terms used everywhere
- [Writing a catalog](docs/catalog/overview.mdx)
- [The query format](docs/query/query-model.mdx), [joins](docs/query/joins.mdx), [graphs](docs/catalog/graphs.mdx) and [similarity search](docs/query/similarity-search.mdx)
- [The command line](docs/reference/cli.mdx) and the [Python API](docs/reference/python.mdx)
- [Errors](docs/reference/errors.mdx) and [limits](docs/reference/limits.mdx)
- [Running YoDb](docs/operations/operating.mdx): database roles, secrets and time limits

The docs build to a static site with `python3 docs-site/build.py`. Design notes
are in [plans/](plans/).

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests     # unit tests, no database needed
```

The end-to-end suites run against a throwaway PostgreSQL in Docker and compare
every answer with plain SQL. See [tests/e2e/README.md](tests/e2e/README.md), and
[tests/e2e/age/README.md](tests/e2e/age/README.md) for the graph and similarity suite
(PostgreSQL + Apache AGE + pgvector).
