# Notes

- [ ] We need to modify makefile to be more concise and allow `make extract-local` that will extract locally to `PATH=`
- [X] we need a proper cli now
- [ ] need a way to retrieve entire page as well
- [ ] when building the mcp the agent has access to the internet and can fetch urls

- [ ] how can we best retrieve code snippets and examples? Should we investigate a graph approach that links pages together with code snippets found on that page.

- [ ] I'm now trying to think from an llm perspective how can I provide enough endpoints that would allow the llm to get all of the context necessary to solve a problem.

- [ ] Currently they can semantic search, get more context around the result, and retrieve the entire page. What can we also add that would make search better? What are the advantage of adding a summary and including this in the semantic search as a weight to the page? I can imaging having an endpoint that can just take a question and do semantic search on that query. It would perhaps be advantageous to have some summaries that might answer that question be part of the embedding.
- [ ]

New Tasks:
- [ ] Need a way to view an entire doc. The llm should be able to return the entire document and put it into context if deemed relevant.
- [ ] Not entirely sure expand-context is doing anything here
    - [ ] the intent here is when this parameter is called we will load the doc into redis. We should validate that this is the only time we call redis. This seems not to be the case, I believe we are trying to load docs into redis when we call extract which is not correct. Once the doc is loaded into cache, we find the chunk in the doc and return n lines above and below the chunk where n is `--expand-context n`.
- [ ] we should remove the extraction metadata.
- [ ] During extraction I want to extract code snippets from pages before batching them, and save them separately. This way in terms of raw files we have:
    - [ ] whole page
    - [ ] batch
    - [ ] code snippet
- [ ] We should include metadata on:
    - [ ] total size of the parent page
    - [ ] how many embedded links are on this page
        - [ ] how many embedded links are internal (in the database, batched and processed)
        - [ ] how many embedded links are external (outside of the database would require fetch)
    - [ ] how many embedded links are on this batch
        - [ ] how many embedded links are internal (in the database, batched and processed)
        - [ ] how many embedded links are external (outside of the database would require fetch)
    - [ ] list of internal/external links as full urls
        - [ ] if internal include document_id
    - [ ] list to code_snippets on page with relevant document_id


```bash
╭─ Result 2 (Score: 0.263, Doc ID: df76256e-6de7-4542-80ad-57aab0f81111, Type: expanded_chu─╮
│                                                                                           │
│  Layout - ratatui.rs                                                                      │
│  https://ratatui.rs/examples/layout/                                                      │
│  Document ID: df76256e-6de7-4542-80ad-57aab0f81111                                        │
│  Source: crawl4ai | Chunk: 0                                                              │
│  Extracted: 2025-07-05 23:14                                                              │
│  Extraction: 50/133 pages successful                                                      │
│  🔍 Expanded Context (31 lines)                                                           │
│                                                                                           │
│  📋 Complete Metadata:                                                                    │
│  • Document ID: df76256e-6de7-4542-80ad-57aab0f81111                                      │
│  • Chunk ID: c2b1e561-5505-42e3-83bb-7b85e1282826                                         │
│  • Chunk Index: 0                                                                         │
│  • Page URL: https://ratatui.rs/examples/layout/                                          │
│  • Content Type: expanded_chunk                                                           │
│  • Score: 0.263396                                                                        │
│  • Content Length: 1,174 chars                                                            │
│  • Source Type: crawl4ai                                                                  │
│  • Extraction Time: 2025-07-05T23:14:36.168293                                            │
│  • Total Links Found: 133                                                                 │
│  • Filtered Links: 50                                                                     │
│  • Successful Extractions: 50                                                             │
│  • Additional Metadata:                                                                   │
│    • method: paragraph_sentence                                                           │
│    • source_url: https://ratatui.rs/examples/layout/                                      │
│    • chunk_index: 0                                                                       │
│    • source_title: Layout - ratatui.rs                                                    │

```

can we add a summary as well how can we expand metadata and it to the embedding
