# Things to fix/improve

- [ ] remove all mentions of text-embedding-3-small we should only be using `text-embedding-3-large`
- [ ] simplify cli by having `ctx search code` and `ctx search docs`, currently we have `ctx code search` which is redundant
- [ ] also simplify by having `ctx get code` and `ctx get docs`
- [ ] no way to get individual chunks. mcp_json only provides summary, and we can only retrieve the entire document.
    - [ ] how would I retrieve this chunk using the `id`?
```
        {
            'id': 'a04df91e-cbd1-4d09-b936-b147937cb65d',
            'document_id': '4bbd5454-be21-4e4d-aeb5-1fc9b1ca5d25',
            'title': 'Fit Markdown - docs.crawl4ai.com (Cleaned)',
            'summary': 'Fit Markdown is an essential feature designed to enhance content extraction and
summarization by focusing on the most relevant text. It offers two main content filters:
**PruningContentFilter**, which extracts the most substantial text without needing a user query, and
**BM25ContentFilter**, ideal for query-based searches. Users can further refine their results by utilizing
parameters like `excluded_tags`, `exclude_external_links`, and `word_count_threshold`. The processed text
is stored in `result.markdown.fit_markdown`, facilitating efficient integration into AI pipelines.
Overall, Fit Markdown helps users eliminate irrelevant content, ensuring that only the most pertinent
information is retained for analysis or processing.',
            'score': 0.2958657216203024,
            'url': 'https://docs.crawl4ai.com/core/fit-markdown',
            'has_summary': True,
            'code_snippets_count': 0,
            'code_snippet_ids': [],
            'content_type': 'chunk',
            'chunk_index': None
        }
    ],

```

- [ ] move these to `ctx setup`
completion  -- Shell completion setup commands.
config      -- Show current configuration.
init        -- Initialize Context Server MCP integration...

- [ ] change this to just be `ctx -v`
version     -- Show version information.

search      -- Search documents in contexts using vector,...
get         -- this is new
context     -- Manage documentation contexts for...
job         -- Manage document processing jobs.
server      -- Manage Context Server Docker services and...


# Metadata improvements
## mcp_json code response
ctx search code "simple async crawler" my-docs --limit 10 --format mcp_json
```
{
    'id': '11a4c625-c2ce-4947-861f-41a17e4bd615',
    'document_id': '319324a2-9f3c-4857-99a0-08e47810f209',
    'title': 'Adaptive Strategies - docs.crawl4ai.com (Cleaned)',
    'content': '# Use multiple start points\nstart_urls = [\n
"https://docs.example.com/intro",\n    "https://docs.example.com/api",\n
"https://docs.example.com/guides"\n]\n\n# Crawl in parallel\ntasks = [\n    adaptive.digest(url, query)\n
for url in start_urls\n]\nresults = await asyncio.gather(*tasks)\nCopy',
    'language': 'text',
    'snippet_type': 'code_block',
    'score': 0.47363974532138514,
    'url': 'https://docs.crawl4ai.com/advanced/adaptive-strategies',
    'start_line': None,
    'end_line': None,
    'content_type': 'code_snippet'
}

```
- [X] get rid of language since this always defaults to `text`.
- [X] `start_line` and `end_line` don't seem to properly populate. We should just have number of lines instead.


## json code response
ctx search code "prune" my-docs --format json
```
{
    'id': '11a4c625-c2ce-4947-861f-41a17e4bd615',
    'document_id': '319324a2-9f3c-4857-99a0-08e47810f209',
    'title': 'Adaptive Strategies - docs.crawl4ai.com (Cleaned)',
    'content': '# Use multiple start points\nstart_urls = [\n
"https://docs.example.com/intro",\n    "https://docs.example.com/api",\n
"https://docs.example.com/guides"\n]\n\n# Crawl in parallel\ntasks = [\n    adaptive.digest(url, query)\n
for url in start_urls\n]\nresults = await asyncio.gather(*tasks)\nCopy',
    'summary': None,
    'summary_model': None,
    'score': 0.47363974532138514,
    'metadata': {
        'document': {
            'id': '319324a2-9f3c-4857-99a0-08e47810f209',
            'title': 'Adaptive Strategies - docs.crawl4ai.com',
            'url': 'https://docs.crawl4ai.com/advanced/adaptive-strategies'
        },
        'pages_found': 12,
        'extraction_mode': 'quality',
        'average_page_size': 11066,
        'total_content_length': 88530,
        'type': 'code_block',
        'char_end': 5652,
        'end_line': 207,
        'language': 'text',
        'char_start': 5353,
        'snippet_id': '4dc2b051-a2c0-4697-87d0-a7c36b29612f',
        'start_line': 192
    },
    'url': 'https://docs.crawl4ai.com/advanced/adaptive-strategies',
    'chunk_index': None,
    'content_type': 'code_snippet'
}
```
- [X] get rid of summary and summary/summary_model here
- [X] chunk_index here also seems to be useless
- [X] `\nCopy` seems to be inserted in every code snippet
- [X] `snippet_id` doesn't seem to do anything.\
```
ctx get code my-docs 4dc2b051-a2c0-4697-87d0-a7c36b29612f
✗ Code snippet '4dc2b051-a2c0-4697-87d0-a7c36b29612f' not found in context 'my-docs'

ctx get chunk my-docs 4dc2b051-a2c0-4697-87d0-a7c36b29612f
✗ Chunk '4dc2b051-a2c0-4697-87d0-a7c36b29612f' not found in context 'my-docs'

ctx get document my-docs 4dc2b051-a2c0-4697-87d0-a7c36b29612f
✗ Document '4dc2b051-a2c0-4697-87d0-a7c36b29612f' not found in context 'my-docs'
```

## mcp_json search response
ctx search query "PruningContentFilter" --limit 1  my-docs --format mcp_json

```
{
    'id': 'c8886a24-fe9b-4916-982b-6dba77f95bb7',
    'document_id': '17f52139-d96c-488b-ae0a-e51b9b0f5182',
    'title': 'Quickstart - docs.crawl4ai.com (Cleaned)',
    'summary': "Crawl4AI's default cache mode is set to `CacheMode.ENABLED`, which may require
switching to `CacheMode.BYPASS` for fresh content. The tool automatically generates Markdown from crawled
pages, with output varying based on whether a **markdown generator** or **content filter** is specified.
If neither is provided, the output will typically be raw Markdown. Additionally, Crawl4AI supports
structured data extraction using CSS or XPath selectors, and now offers a utility for generating reusable
extraction schemas without relying on LLMs, enhancing efficiency for repetitive page structures. Users can
also input raw HTML by prefixing it with `raw://` for crawling.",
    'score': 0.22396116191480814,
    'url': 'https://docs.crawl4ai.com/core/quickstart',
    'has_summary': True,
    'code_snippets_count': 2,
    'code_snippet_ids': [
        {
            'id': 'f4bd7929-754e-418c-a6d6-3ed074c74cf7',
            'size': 23,
            'summary': '9-line function snippet'
        },
        {
            'id': '937d3913-7557-4cf0-a073-771a55a6616e',
            'size': 24,
            'summary': '16-line function snippet'
        }
    ],
    'content_type': 'chunk',
    'chunk_index': None
}
```
- [X] we should include a small preview of the code and the `Document` field for each code_snippet_ids
- [X] has_summary can be removed.
- [X] chunk_index here can be removed



## json search response
ctx search query "PruningContentFilter" --limit 1  my-docs --format json
```
{
    'id': 'c8886a24-fe9b-4916-982b-6dba77f95bb7',
    'document_id': '17f52139-d96c-488b-ae0a-e51b9b0f5182',
    'title': 'Quickstart - docs.crawl4ai.com (Cleaned)',
    'content': '[CODE_SNIPPET: language=text, size=535chars, summary="16-line function snippet",
snippet_id=f802f325-19b6-4ea9-80a7-bb390e2baf09]\n> IMPORTANT: By default cache mode is set to
`CacheMode.ENABLED`. So to have fresh content, you need to set it to `CacheMode.BYPASS`\nWe’ll explore
more advanced config in later tutorials (like enabling proxies, PDF output, multi-tab sessions, etc.). For
now, just note how you pass these objects to manage crawling.\n* * *\n## 4. Generating Markdown Output\nBy
default, Crawl4AI automatically generates Markdown from each crawled page. However, the exact output
depends on whether you specify a **markdown generator** or **content filter**.\n  *
**`result.markdown`**:\nThe direct HTML-to-Markdown conversion.\n  *
**`result.markdown.fit_markdown`**:\nThe same content after applying any configured **content filter**
(e.g., `PruningContentFilter`).\n### Example: Using a Filter with
`DefaultMarkdownGenerator`\n[CODE_SNIPPET: language=text, size=681chars, summary="15-line configuration
snippet", snippet_id=be0cbe55-2bc8-487f-ae01-130051b1de0e]\n**Note** : If you do **not** specify a content
filter or markdown generator, you’ll typically see only the raw Markdown. `PruningContentFilter` may adds
around `50ms` in processing time. We’ll dive deeper into these strategies in a dedicated **Markdown
Generation** tutorial.\n* * *\n## 5. Simple Data Extraction (CSS-based)\nCrawl4AI can also extract
structured data (JSON) using CSS or XPath selectors. Below is a minimal CSS-based example:\n> **New!**
Crawl4AI now provides a powerful utility to automatically generate extraction schemas using LLM. This is a
one-time cost that gives you a reusable schema for fast, LLM-free extractions:\n[CODE_SNIPPET:
language=text, size=723chars, summary="17-line configuration snippet",
snippet_id=f5b0135b-9de9-4d11-97fb-bd44653a1265]\nFor a complete guide on schema generation and advanced
usage, see [No-LLM Extraction
Strategies](https://docs.crawl4ai.com/extraction/no-llm-strategies/).\nHere\'s a basic extraction
example:\n[CODE_SNIPPET: language=text, size=1019chars, summary="28-line function snippet",
snippet_id=eb9bc40c-7c89-446b-a3c1-826b61a2b045]\n**Why is this helpful?** - Great for repetitive page
structures (e.g., item listings, articles). - No AI usage or costs. - The crawler returns a JSON string
you can parse or store.\n> Tips: You can pass raw HTML to the crawler instead of a URL. To do so, prefix
the HTML with `raw://`.\n* * *\n## 6. Simple Data Extraction (LLM-based)',
    'summary': "Crawl4AI's default cache mode is set to `CacheMode.ENABLED`, which may require
switching to `CacheMode.BYPASS` for fresh content. The tool automatically generates Markdown from crawled
pages, with output varying based on whether a **markdown generator** or **content filter** is specified.
If neither is provided, the output will typically be raw Markdown. Additionally, Crawl4AI supports
structured data extraction using CSS or XPath selectors, and now offers a utility for generating reusable
extraction schemas without relying on LLMs, enhancing efficiency for repetitive page structures. Users can
also input raw HTML by prefixing it with `raw://` for crawling.",
    'summary_model': 'gpt-4o-mini',
    'score': 0.22399519124441822,
    'metadata': {
        'document': {
            'id': '17f52139-d96c-488b-ae0a-e51b9b0f5182',
            'title': 'Quickstart - docs.crawl4ai.com',
            'url': 'https://docs.crawl4ai.com/core/quickstart',
            'size': 8247,
            'total_chunks': 9
        },
        'chunk': {
            'index': 1,
            'links_count': 1,
            'links': {
                'https://docs.crawl4ai.com/extraction/no-llm-strategies/': {
                    'href': 'https://docs.crawl4ai.com/extraction/no-llm-strategies/',
                    'text': 'No-LLM Extraction Strategies'
                }
            }
        },
        'code_snippets': [
            {
                'id': 'f4bd7929-754e-418c-a6d6-3ed074c74cf7',
                'type': 'code_block',
                'language': 'text',
                'start_line': 20,
                'end_line': 32,
                'preview': '9-line function snippet'
            },
            {
                'id': '937d3913-7557-4cf0-a073-771a55a6616e',
                'type': 'code_block',
                'language': 'text',
                'start_line': 41,
                'end_line': 61,
                'preview': '16-line function snippet'
            }
        ],
        'code_snippets_count': 2,
        'pages_found': 12,
        'extraction_mode': 'quality',
        'average_page_size': 11066,
        'total_content_length': 88530
    },
    'url': 'https://docs.crawl4ai.com/core/quickstart',
    'chunk_index': None,
    'content_type': 'chunk'
}
```
- [ ] seems like chunk_index is defined twice here
- [ ] code_snippet still has language

## raw documents
ctx get document my-docs 319324a2-9f3c-4857-99a0-08e47810f209 --format raw
```
[CODE_SNIPPET: language=text, size=244chars, summary="prune_filter = PruningContentFilter(
threshold=0.5,
threshold_type="fixed",
min_word_threshold=10
)
```
- [X] code_snippets still have language
- [ ] code_snippet should include preview like first few lines


still need to make the whole process async and faster
