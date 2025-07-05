# Notes

- [ ] We need to modify makefile to be more concise and allow `make extract-local` that will extract locally to `PATH=`
- [-] we need a proper cli now
- [ ] need a way to retrieve entire page as well
- [ ] when building the mcp the agent has access to the internet and can fetch urls
- [ ]

can you explain some of the metadata here?
```bash
╭────────── Result 8 (Score: 0.257, Doc ID: 051616b5-a934-4904-b8be-c43a0501295b, Type: chunk) ──────────╮
│                                                                                                        │
│  Paragraph - ratatui.rs                                                                                │
│  https://ratatui.rs/examples/widgets/paragraph/                                                        │
│  Document ID: 051616b5-a934-4904-b8be-c43a0501295b                                                     │
│  Source: crawl4ai | Chunk: 4                                                                           │
│  Extracted: 2025-07-05 02:21                                                                           │
│  Extraction: 50/133 pages successful                                                                   │
│                                                                                                        │
│  📋 Complete Metadata:                                                                                 │
│  • Document ID: 051616b5-a934-4904-b8be-c43a0501295b                                                   │
│  • Chunk ID: 390cba90-d02f-4670-abf6-6eea0136a764                                                      │
│  • Chunk Index: 4                                                                                      │
│  • Page URL: https://ratatui.rs/examples/widgets/paragraph/                                            │
│  • Content Type: chunk                                                                                 │
│  • Score: 0.256851                                                                                     │
│  • Content Length: 3,383 chars                                                                         │
│  • Source Type: crawl4ai                                                                               │
│  • Extraction Time: 2025-07-05T02:21:41.792433                                                         │
│  • Total Links Found: 133                                                                              │
│  • Filtered Links: 50                                                                                  │
│  • Successful Extractions: 50                                                                          │
│  • Additional Metadata:                                                                                │
```

- some of these don't seem useful and some might just be broken like `Extraction: 50/133 pages successful` seems to be on every page regardless of batch or url. Can we add summary and include that within the embedding for a weighted return instead.

how can we best retrieve code snippets and examples? Should we investigate a graph approach that links pages together with code snippets found on that page.

I'm now trying to think from an llm perspective how can I provide enough endpoints that would allow the llm to get all of the context necessary to solve a problem.

Currently they can semantic search, get more context around the result, and retrieve the entire page. What can we also add that would make search better? What are the advantage of adding a summary and including this in the semantic search as a weight to the page? I can imaging having an endpoint that can just take a question and do semantic search on that query. It would perhaps be advantageous to have some summaries that might answer that question be part of the embedding.
