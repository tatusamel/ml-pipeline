# Index Metadata Schema

Each chunk must carry enough metadata to support filtering and evaluation.

Required fields:
- `chunk_id`: unique id
- `doc_id`: original document id
- `source`: system/app/service name
- `timestamp`: ISO-8601 of the source content (if available)
- `section`: logical section name or path
- `tokens`: token count (int)
- `hash`: content hash for dedupe
- `run_id`: pipeline run that produced this vector

Optional fields:
- `author`, `tags` (array), `pii_redacted` (bool)
