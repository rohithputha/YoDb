-- The Yelp data layer: relational tables, a pgvector column for review embeddings, and an Apache AGE
-- graph for the relationships that have no column (friendships, who reviewed what, categories).
-- Run once as the database owner/superuser; load.py fills it.  YoDb connects as the read-only role.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DROP SCHEMA IF EXISTS yelp CASCADE;
CREATE SCHEMA yelp;

CREATE TABLE yelp.business (
  business_id  text PRIMARY KEY,
  name         text NOT NULL,
  address      text,
  city         text,
  state        text,
  postal_code  text,
  latitude     double precision,
  longitude    double precision,
  stars        real,
  review_count integer,
  is_open      boolean,
  categories   text,                 -- the comma-separated list as Yelp gives it (also split into yelp.business_category)
  attributes   jsonb,
  hours        jsonb
);

CREATE TABLE yelp.category (
  category_id text PRIMARY KEY,      -- the category name in lower_snake_case (also the graph vertex id)
  name        text NOT NULL
);

-- The relational form of the business-category relationship (the harness's oracle; YoDb follows the graph).
CREATE TABLE yelp.business_category (
  business_id text REFERENCES yelp.business (business_id),
  category_id text REFERENCES yelp.category (category_id),
  PRIMARY KEY (business_id, category_id)
);

CREATE TABLE yelp.users (
  user_id       text PRIMARY KEY,
  name          text,
  review_count  integer,
  yelping_since timestamp,
  useful        integer,
  funny         integer,
  cool          integer,
  fans          integer,
  elite         text,                -- comma-separated years, as Yelp gives it
  average_stars real
);

-- Friendships among the sampled users, both directions (the oracle; YoDb follows the graph).
CREATE TABLE yelp.friendship (
  user_id   text REFERENCES yelp.users (user_id),
  friend_id text REFERENCES yelp.users (user_id),
  PRIMARY KEY (user_id, friend_id)
);

CREATE TABLE yelp.review (
  review_id      text PRIMARY KEY,
  user_id        text REFERENCES yelp.users (user_id),
  business_id    text NOT NULL REFERENCES yelp.business (business_id),
  stars          real,
  useful         integer,
  funny          integer,
  cool           integer,
  text           text,
  date           timestamp,
  text_embedding vector(384)         -- BAAI/bge-small-en-v1.5, cosine; filled by load.py
);
CREATE INDEX review_business_idx ON yelp.review (business_id);
CREATE INDEX review_user_idx ON yelp.review (user_id);
CREATE INDEX review_stars_idx ON yelp.review (stars);
CREATE INDEX business_city_idx ON yelp.business (city);
CREATE INDEX business_stars_idx ON yelp.business (stars);

-- The graph: vertices hold only the id; edges carry the relationships.  A graph from an earlier load is dropped.
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'yelp_graph') THEN
    PERFORM ag_catalog.drop_graph('yelp_graph', true);
  END IF;
END $$;
SELECT ag_catalog.create_graph('yelp_graph');
SELECT ag_catalog.create_vlabel('yelp_graph', 'Business');
SELECT ag_catalog.create_vlabel('yelp_graph', 'User');
SELECT ag_catalog.create_vlabel('yelp_graph', 'Category');
SELECT ag_catalog.create_elabel('yelp_graph', 'FRIENDS_WITH');     -- User -> User, both directions
SELECT ag_catalog.create_elabel('yelp_graph', 'REVIEWED');         -- User -> Business (once per pair)
SELECT ag_catalog.create_elabel('yelp_graph', 'IN_CATEGORY');      -- Business -> Category
-- Indexes on the id property, for the MATCH ... WHERE v.id = $x that YoDb runs and the {id: ...} patterns the
-- loader runs.  Each is attempted on its own: an AGE build without the operator class only loses the index.
DO $$
DECLARE label text; stmt text;
BEGIN
  FOREACH label IN ARRAY ARRAY['Business', 'User', 'Category'] LOOP
    BEGIN
      stmt := format('CREATE INDEX %I ON yelp_graph.%I USING btree ((ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, ''"id"''::ag_catalog.agtype])))',
                     lower(label) || '_vertex_id', label);
      EXECUTE stmt;
    EXCEPTION WHEN OTHERS THEN RAISE NOTICE 'skipped btree id index on %: %', label, SQLERRM;
    END;
    BEGIN
      stmt := format('CREATE INDEX %I ON yelp_graph.%I USING gin (properties)', lower(label) || '_vertex_props', label);
      EXECUTE stmt;
    EXCEPTION WHEN OTHERS THEN RAISE NOTICE 'skipped gin index on %: %', label, SQLERRM;
    END;
  END LOOP;
END $$;

-- Least-privilege, read-only role used by YoDb; AGE preloaded for its sessions.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'yodb_ro') THEN
    CREATE ROLE yodb_ro LOGIN PASSWORD 'yodb_ro';
  END IF;
END $$;
GRANT USAGE ON SCHEMA yelp, ag_catalog, yelp_graph TO yodb_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA yelp, ag_catalog, yelp_graph TO yodb_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA yelp GRANT SELECT ON TABLES TO yodb_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA yelp_graph GRANT SELECT ON TABLES TO yodb_ro;
ALTER ROLE yodb_ro SET session_preload_libraries = 'age';
ALTER ROLE yodb_ro SET search_path = ag_catalog, "$user", public;
