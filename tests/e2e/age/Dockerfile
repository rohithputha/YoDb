# PostgreSQL 16 with Apache AGE (graph) and pgvector (vectors): the two extensions YoDb's
# graph traversal and similarity search run on.  The apache/age image is the official
# postgres:16 image with AGE built in; pgvector comes from the PGDG apt repository it ships.
FROM apache/age:release_PG16_1.5.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-pgvector \
 && rm -rf /var/lib/apt/lists/*
