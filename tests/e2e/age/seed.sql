-- A shop whose customer-ticket relationship lives in an Apache AGE graph, and whose ticket text has
-- pgvector embeddings.  Run as the superuser; YoDb only ever connects as the read-only role yodb_ro.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

CREATE SCHEMA shop;

-- Toy 5-dimensional embeddings: keyword counts + a bias, so the Python ToyEmbedder in run_age.py
-- computes exactly the same vector for a query text.
CREATE FUNCTION shop.kw(t text, k text) RETURNS int LANGUAGE sql IMMUTABLE AS
  $$ SELECT (length(lower(t)) - length(replace(lower(t), k, ''))) / length(k) $$;
CREATE FUNCTION shop.toy_embed(t text) RETURNS vector(5) LANGUAGE sql IMMUTABLE AS
  $$ SELECT ('[' || shop.kw(t,'price') || ',' || shop.kw(t,'cancel') || ',' ||
             shop.kw(t,'refund') || ',' || shop.kw(t,'bug') || ',1]')::vector(5) $$;

CREATE TABLE shop.customers (customer_id text PRIMARY KEY, name text, country text, seats integer);
INSERT INTO shop.customers VALUES
 ('c01','Ann','US',10), ('c02','Bob','UK',5), ('c03','Cai','US',20), ('c04','Dee',NULL,1),
 ('c05','Eli','DE',7),  ('c06','Fay','US',3), ('c07','Gus','UK',9), ('c08','Hal','DE',2);

CREATE TABLE shop.tickets (
  ticket_id text PRIMARY KEY, subject text, status text, priority integer, body text, body_embedding vector(5)
);
INSERT INTO shop.tickets (ticket_id, subject, status, priority, body) VALUES
 ('t01','Login',   'open',    3, 'cannot login'),
 ('t02','Invoice', 'closed',  1, 'refund the invoice, the price was wrong'),
 ('t03','Crash',   'open',    5, 'app crash, a bug for sure'),
 ('t04','Slow',    'open',    2, 'slow pages'),
 ('t05','Export',  'closed',  4, 'export fails with a bug'),
 ('t06','Refund',  'open',    4, 'please refund me, I want to cancel'),
 ('t07','Billing', 'pending', 1, 'billing question about the price'),
 ('t08','Outage',  'open',    5, 'total outage'),
 ('t09','Feature', 'open',    3, 'feature idea'),
 ('t10','Thanks',  'closed',  2, 'thank you'),
 ('t11','Orphan',  'open',    1, 'no customer owns me'),
 ('t12','Cancel',  'closed',  3, 'cancel and refund, price too high'),
 ('t13','Upgrade', 'pending', 5, 'refund after upgrade'),
 ('t14','Empty',   'open',    2, NULL);
UPDATE shop.tickets SET body_embedding = shop.toy_embed(body) WHERE body IS NOT NULL;

-- The relationships, as plain rows: the oracle joins these, YoDb never sees them.  They are loaded
-- into the graph below.  Note t10 belongs to c07 only through the graph: there is no column for it.
CREATE TABLE shop.links (edge_type text, from_id text, to_id text);
INSERT INTO shop.links VALUES
 ('HAS_TICKET','c01','t01'), ('HAS_TICKET','c01','t02'), ('HAS_TICKET','c01','t03'), ('HAS_TICKET','c02','t04'),
 ('HAS_TICKET','c03','t05'), ('HAS_TICKET','c03','t06'), ('HAS_TICKET','c04','t07'), ('HAS_TICKET','c05','t08'),
 ('HAS_TICKET','c05','t09'), ('HAS_TICKET','c07','t10'), ('HAS_TICKET','c01','t13'), ('HAS_TICKET','c02','t12'),
 ('HAS_TICKET','c06','t14'),
 ('REFERRED_BY','c02','c01'), ('REFERRED_BY','c03','c01'), ('REFERRED_BY','c04','c02'), ('REFERRED_BY','c06','c03'),
 ('REFERRED_BY','c08','c05');

-- The graph: one vertex per customer and ticket (holding only the id), and the edges.
SELECT ag_catalog.create_graph('shop_graph');
DO $$
DECLARE r record;
BEGIN
  FOR r IN SELECT customer_id FROM shop.customers LOOP
    EXECUTE format($f$SELECT * FROM ag_catalog.cypher('shop_graph', $c$ CREATE (:Customer {id: %L}) $c$) AS (v ag_catalog.agtype)$f$, r.customer_id);
  END LOOP;
  FOR r IN SELECT ticket_id FROM shop.tickets LOOP
    EXECUTE format($f$SELECT * FROM ag_catalog.cypher('shop_graph', $c$ CREATE (:Ticket {id: %L}) $c$) AS (v ag_catalog.agtype)$f$, r.ticket_id);
  END LOOP;
  FOR r IN SELECT from_id, to_id FROM shop.links WHERE edge_type = 'HAS_TICKET' LOOP
    EXECUTE format($f$SELECT * FROM ag_catalog.cypher('shop_graph', $c$ MATCH (a:Customer {id: %L}), (b:Ticket {id: %L}) CREATE (a)-[:HAS_TICKET]->(b) $c$) AS (v ag_catalog.agtype)$f$, r.from_id, r.to_id);
  END LOOP;
  FOR r IN SELECT from_id, to_id FROM shop.links WHERE edge_type = 'REFERRED_BY' LOOP
    EXECUTE format($f$SELECT * FROM ag_catalog.cypher('shop_graph', $c$ MATCH (a:Customer {id: %L}), (b:Customer {id: %L}) CREATE (a)-[:REFERRED_BY]->(b) $c$) AS (v ag_catalog.agtype)$f$, r.from_id, r.to_id);
  END LOOP;
END $$;

-- Least-privilege, read-only role used by YoDb.  It cannot LOAD a library itself, so AGE is preloaded
-- for its sessions, and ag_catalog is on its search path for AGE's own function lookups.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'yodb_ro') THEN
    CREATE ROLE yodb_ro LOGIN PASSWORD 'yodb_ro';
  END IF;
END $$;
GRANT USAGE ON SCHEMA shop, ag_catalog, shop_graph TO yodb_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA shop, ag_catalog, shop_graph TO yodb_ro;
ALTER ROLE yodb_ro SET session_preload_libraries = 'age';
ALTER ROLE yodb_ro SET search_path = ag_catalog, "$user", public;
ANALYZE;
