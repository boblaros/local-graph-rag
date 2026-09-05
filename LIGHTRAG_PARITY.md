# LightRAG adapter assumptions

This note records the version-specific LightRAG behavior required by the
experiment. It explains how the Native extraction is reused for ER and ER+RR;
it does not claim that the derived graphs are internally identical to a Native
LightRAG graph.

## Version

| Item | Value |
| --- | --- |
| Local source | `LightRAG/` |
| Commit | `ffd831f5de49d063da0cb69282781143cfb5b322` |
| Git description | `v1.5.2-30-gffd831f5` |
| Installed package | `lightrag-hku==1.5.2` in `.venv` |

The commit is part of the experiment definition because the local source
contains changes made after the published `v1.5.2` tag. Recheck the behavior in
this file and run the integration tests before changing the LightRAG version.

## LightRAG methods used

The adapter uses these `LightRAG` methods:

- `ainsert` for the single Native indexing run;
- `apipeline_enqueue_documents` and
  `apipeline_process_enqueue_documents` to store the fixed chunks in derived
  workspaces without extracting entities or relations again;
- `acreate_entity` and `acreate_relation` to write the ER and ER+RR graphs;
- `initialize_storages` and `finalize_storages` for workspace lifecycle;
- `get_graph_labels`, `get_processing_status`, `aget_docs_by_track_id`, and
  `get_knowledge_graph` for validation and export;
- `aquery_data` and `aquery_llm` for retrieval.

Relevant definitions are in `lightrag/lightrag.py`,
`lightrag/pipeline.py`, `lightrag/utils_graph.py`, and `lightrag/types.py`.

## One extraction per builder

`LightRAG.ainsert` performs chunking, entity and relation extraction, graph
updates, vector and key-value updates, and persistence. The pinned version has
no public extraction-only method.

The experiment therefore captures each builder response during the Native run,
normalizes the captured records, and uses those saved records for ER and RR.
Calling `ainsert` again for a derived graph would violate the one-extraction
design.

### Raw response capture

The pinned version has no public callback that provides both the raw extraction
response and its chunk and document identifiers. The adapter therefore uses two
version-specific internal access points:

- a temporary wrapper around `lightrag.operate.use_llm_func_with_cache` carries
  the chunk and document identifiers;
- the extraction role is replaced temporarily with a recording Ollama function
  through LightRAG's role-update method.

The recording function passes through the same prompt, history, options,
retries, and response. It stores the physical response byte-for-byte and returns
the same response to LightRAG's parser. It does not change the installed
LightRAG source or any workspace file. These access points must be checked again
after a LightRAG upgrade.

All graph and storage writes for ER and ER+RR use the LightRAG methods listed
above. Exact chunk inspection uses the read-only storage access described under
"Known differences and limitations."

## Why custom-KG insertion is not used

`ainsert_custom_kg` creates chunk IDs from content hashes. Native chunk IDs are
document-scoped, for example `<document_id>-chunk-000`. Custom-KG insertion
would therefore change chunk identifiers and source references between Native
and the derived conditions.

`ainsert_custom_chunks` has the same identifier problem and also performs
extraction. Neither method satisfies the shared-chunk requirement.

## Extraction record inclusion

In the pinned LightRAG version, both tuple and JSON extraction discard entity or
relation records whose description is empty. The offline normalizer applies the
same inclusion rule, so an incomplete record is not added only to a derived
condition.

The raw response remains available in `extraction_calls.jsonl`. For each
successful call, the adapter records the capture version and
`response_mutated=false`. Native parsing continues to use LightRAG's parser and
JSON repair behavior. The experiment-specific normalizer parses the same raw
response separately to create the ER input and provenance records.

The derived graph writer uses deterministic description placeholders only for
implicit endpoints or compatible older staged records. Every placeholder is
recorded in the graph audit. It does not make an incomplete extraction record
eligible.

## Derived workspace construction

ER and ER+RR materialization follow the same procedure:

1. Create and initialize a new workspace.
2. Confirm that it contains no graph or document-status records.
3. Install the fixed snapshot chunker.
4. Enqueue the original documents with their original IDs, file paths, and
   track ID, using `process_options="!"` to skip entity and relation extraction.
5. Process the queued documents and verify document status and the exact stored
   chunk IDs, text, document IDs, order, and token counts.
6. Create graph nodes and edges in deterministic order.
7. Export and validate the graph.
8. Finalize the workspace, open it with a new LightRAG instance, export it
   again, and run two small retrieval checks.

The snapshot chunker identifies documents by exact text hash because the
six-argument LightRAG chunker callback does not receive a document ID. The
`multihoprag_120` corpus has 155 distinct document bodies. A duplicate body or
unexpected chunk split causes validation to fail.

The derived workspace must use the same tokenizer and embedding token limit as
Native because LightRAG recalculates token counts before embedding and may split
an oversized chunk.

## Retrieval response handling

For local, global, hybrid, and mix retrieval, successful context-only output is
stored from `llm_response.content`. The surrounding response object is not
converted to text.

When LightRAG returns `status=failure` with
`metadata.failure_reason=no_results`, the experiment records a valid empty
retrieval result. Other failure responses are recorded as execution failures.
An unanswerable dataset question is not automatically an empty retrieval: it
may still retrieve irrelevant context.

For ranking metrics, the runner preserves the order of `data.chunks`, followed
by additional `data.references`. It assigns one rank to the first occurrence of
each document and does not invent scores that LightRAG did not return.

## Known differences and limitations

### Auxiliary chunk indexes

The public `acreate_entity` and `acreate_relation` methods write the graph and
entity or relation vectors but do not update LightRAG's `entity_chunks` and
`relation_chunks` auxiliary stores. Retrieval is covered by the integration
test, but exact Native behavior for later deletion or editing is not claimed.

### Aliases and canonical identifiers

LightRAG stores the entity display name as the graph node identifier and does
not preserve extra alias or canonical-ID fields. Aliases, stable identifiers,
mention provenance, and canonicalization rationale are therefore saved in
separate audit artifacts rather than inserted into retrieval text.

### Exact chunk reads

The pinned `LightRAG` class has no method for reading an arbitrary list of
chunks by ID. Exact content validation uses the read-only
`rag.text_chunks.get_by_ids(...)` interface in
`PublicLightRAGAdapter.get_chunks_by_ids`. The adapter first checks for a future
public `aget_chunks_by_ids` method. No writes use this storage interface.

### Workspace cleanliness and graph export

`get_knowledge_graph` may truncate output at the instance's `max_graph_nodes`;
validation rejects truncated exports. LightRAG cannot list every possible
orphaned vector record, so each derived graph is built in a new directory.

### Native and derived graph semantics

Native LightRAG may use the builder model to summarize descriptions during
graph merging. ER aggregates descriptions deterministically, and derived
materialization stores one undirected edge per endpoint pair. The controlled
comparison guarantees a shared extraction and chunk lineage, not identical
internal graph construction.

## Verification

Run the focused graph tests:

```bash
.venv/bin/python -m pytest \
  tests/test_graph_rewrite.py \
  tests/test_graph_validation.py \
  tests/test_graph_materialization.py \
  tests/test_graph_boundaries.py -q
```

Run the integration test against the installed local LightRAG package. It uses
deterministic local substitutes and makes no model or network calls:

```bash
.venv/bin/python -m pytest \
  tests/test_graph_lightrag_integration.py -q -m integration
```

Static checks for the same area:

```bash
.venv/bin/ruff check src/graph \
  tests/test_graph_rewrite.py \
  tests/test_graph_validation.py \
  tests/test_graph_materialization.py \
  tests/test_graph_boundaries.py \
  tests/test_graph_lightrag_integration.py
```
