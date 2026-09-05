# MultiHop-RAG subset selection report: multihoprag_120

## Result

The immutable subset contains **120 questions** (inference_query=30, comparison_query=30, temporal_query=30, null_query=30), **125 unique gold documents**, and **30 additional hard-negative documents**. The final corpus has **155 documents**.

Selection uses `seed = 42` only for SHA-256-based stable tie breaking. It does not inspect pilot/model outputs or metrics. The 16 pilot question IDs are a blocklist and are not otherwise scored.

## Deterministic question selection

For each of the four source types, exactly 30 questions are selected. Evidence-count quotas are as even as possible across the counts supported by that type: inference 2/3/4, comparison 2/3, temporal 2/3, and null 0. Questions with repeated evidence URLs are ineligible because the experiment requires 2–4 unique gold URLs.

The greedy objective has a soft target of **125 unique gold documents** so that the requested hard negatives lead toward a corpus near 155. No gold URL is ever dropped. A gold document may support at most 3 selected questions before a deterministic relaxation is allowed. The objective then minimizes concentration by document, evidence source, category, publication month, body-length tertile, full-dataset reuse bin, and rare lexical topic signatures.

Evidence-count distribution:

- `inference_query`: 2 docs → 10 questions, 3 docs → 10 questions, 4 docs → 10 questions
- `comparison_query`: 2 docs → 15 questions, 3 docs → 15 questions
- `temporal_query`: 2 docs → 15 questions, 3 docs → 15 questions
- `null_query`: 0 docs → 30 questions

Selected gold-document reuse:

- Maximum selected-question uses of one gold document: **3**.
- Reuse histogram (uses → documents): `{"1": 46, "2": 43, "3": 36}`.

## Stratification distributions

Sources below count evidence occurrences; a multi-hop question contributes once for each gold document.

| Evidence source | Count |
|---|---:|
| TechCrunch | 33 |
| Sporting News | 29 |
| The Verge | 19 |
| The Independent - Life and Style | 17 |
| Polygon | 13 |
| The Guardian | 12 |
| Fortune | 11 |
| The Age | 11 |
| The Sydney Morning Herald | 11 |
| CBSSports.com | 10 |
| Music Business Worldwide | 7 |
| Cnbc \| World Business News Leader | 6 |
| Engadget | 6 |
| Live Science: The Most Interesting Articles | 5 |
| The New York Times | 5 |
| FOX News - Health | 4 |
| Advanced Science News | 3 |
| Business Today \| Latest Stock Market And Economy News India | 3 |
| FOX News - Lifestyle | 3 |
| Globes English \| Israel Business Arena | 3 |
| Sky Sports | 3 |
| The Roar \| Sports Writers Blog | 3 |
| Yardbarker | 3 |
| BBC News - Entertainment & Arts | 2 |
| Business Line | 2 |
| Hacker News | 2 |
| Mashable | 2 |
| Science News For Students | 2 |
| The Independent - Sports | 2 |
| Yahoo News | 2 |
| Zee Business | 2 |
| BBC News - Technology | 1 |
| Business World | 1 |
| The Independent - Travel | 1 |
| Wired | 1 |

| Evidence category | Count |
|---|---:|
| sports | 65 |
| technology | 62 |
| business | 53 |
| entertainment | 43 |
| science | 13 |
| health | 4 |

| Publication month | Count |
|---|---:|
| 2023-10 | 90 |
| 2023-11 | 65 |
| 2023-12 | 52 |
| 2023-09 | 33 |

| Body-length tertile | Count |
|---|---:|
| medium | 86 |
| long | 79 |
| short | 75 |

Full-dataset evidence-document reuse bins:

| Full-dataset uses | Count |
|---|---:|
| 10-49 | 97 |
| 3-9 | 69 |
| 50+ | 39 |
| 1-2 | 35 |

For `null_query`, source/category/date/evidence-length stratification is not defined because `evidence_list` is empty. A deterministic top-BM25 corpus document is used only as a selection profile; it is not relabeled as gold evidence.

## Hard-negative method and audit

A local BM25 index over `title + body` selects exactly **30** documents outside the global selected-gold union. The query side uses the original question plus its evidence titles (when present). A seeded greedy facility-location objective maximizes normalized lexical coverage across all selected questions while applying source/category caps. Every selected question is mapped to at least one of the hard negatives.

For every mapped question/document pair, the audit verifies exact gold exclusion, absence of the complete set of exact evidence facts, and absence of the normalized exact full answer (generic answers such as Yes/No and the null sentinel are not treated as answer strings).

| Corpus index | Title | Source | Category | Linked QA source indices | Eligible BM25 rank range |
|---:|---|---|---|---|---:|
| 13 | Manchester United face Galatasaray with high hopes but bad memories | The Guardian | sports | 335, 2497, 2102 | 2–2 |
| 16 | How to choose the best class for you in Baldur’s Gate 3 | Polygon | entertainment | 2316, 1360 | 1–1 |
| 43 | Founders, are events useful? | TechCrunch | technology | 716, 1008, 375, 2208, 225 | 1–10 |
| 99 | How OpenAI's ChatGPT has changed the world in just a year | Engadget | technology | 402, 446, 1796 | 1–2 |
| 101 | Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn | The Guardian | sports | 1572, 2210 | 1–1 |
| 121 | In the end, the FTX trial was about the friends screwed along the way | The Verge | technology | 1129, 1893, 777, 2265, 1771 | 1–3 |
| 133 | ICC World Cup 2023: India at Cricket World Cup semi-finals so far | Zee Business | business | 1148, 281, 1312 | 1–11 |
| 139 | Patterns of Surface Warming Matter for Climate Sensitivity | Eos: Earth And Space Science News | science | 89, 2353 | 1–2 |
| 163 | Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer | FOX News - Health | health | 708, 2252, 1917 | 1–3 |
| 165 | The Halloween Countdown: 31 days of horror to watch | Polygon | entertainment | 1198, 49, 300, 2401, 2517 | 1–12 |
| 194 | Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status | Essentially Sports | sports | 2082 | 1–1 |
| 198 | Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back | Sporting News | sports | 617, 1392, 2095, 1754, 2536, 220, 1784, 2372 | 1–19 |
| 222 | The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more | Engadget | technology | 912, 1680, 1747, 2245, 652, 684, 1676 | 3–11 |
| 239 | Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma | Business Line | business | 891, 1684 | 1–1 |
| 257 | Taylor Swift's 1989: The stories behind her biggest album | BBC News - Entertainment & Arts | entertainment | 1855, 1172, 1833, 205, 1940 | 1–11 |
| 300 | Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang | TalkSport | sports | 1148 | 4–4 |
| 313 | Sweeping White House AI executive order takes aim at the technology's toughest challenges | Engadget | technology | 720, 969, 2121, 528 | 2–12 |
| 326 | Northern Lights: Here are the best tips to help you spot the stunning display in the US and abroad | FOX News - Lifestyle | entertainment | 1645 | 1–1 |
| 352 | European roundup: Bellingham helps Real Madrid go top while Bayern draw | The Guardian | sports | 565, 2217, 159, 501 | 1–19 |
| 355 | Epic v. Google, explained | The Verge | technology | 131, 1095, 1362, 2407, 181, 677 | 1–8 |
| 368 | The Best NBA Betting Sites and Apps for the 2023-24 Season | Sporting News | sports | 190, 2511, 21, 57, 1973 | 1–1 |
| 386 | Monday Night Football DraftKings Picks: NFL DFS lineup advice for Week 15 Eagles-Seahawks Showdown tournaments | Sporting News | sports | 706, 1978, 2460, 2525 | 1–11 |
| 418 | Fundamental Investing: The Art of Relative Valuation | Business Line | business | 1, 1472, 2116, 2022, 2408 | 1–21 |
| 438 | Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton | The Roar \| Sports Writers Blog | sports | 74, 659, 2376 | 1–8 |
| 485 | ASX set to rise ahead of inflation report after big tech lifts Wall Street | The Sydney Morning Herald | business | 138, 1962, 406, 1189, 456, 496 | 1–3 |
| 536 | The inside story of Dave Clark's tumultuous last days at Flexport | Cnbc \| World Business News Leader | business | 1117, 1648, 110, 139, 576, 1335, 1718, 2255 | 1–6 |
| 539 | CEO David Baszucki’s mission to make Roblox a billion-player platform | The Verge | technology | 1740, 2312, 884, 1732 | 1–9 |
| 545 | Can anyone survive Fortnite as a job? | Polygon | entertainment | 325, 2225, 1363 | 1–3 |
| 559 | Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage | The Independent - Life and Style | entertainment | 67, 965, 95, 2371, 2370 | 2–5 |
| 598 | Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas | TechCrunch | technology | 297, 475, 1547, 1767, 2278, 1954 | 1–5 |

## Pilot isolation and overlap

- Question overlap with pilot: **0**.
- Document overlap with pilot: **17** (14 gold, 3 hard negative in this subset).
- Pilot questions, pilot artifacts, and pilot results were not modified.

## Deterministic relaxations

- None. Base document/source/category caps were sufficient.

## Integrity and reproducibility

- Question and document IDs use the exact pilot formulas.
- Exact source query/body strings and all decoded source fields, including JSON nulls, are preserved.
- The manifest records source-file, content, source-record, exact JSONL-line, and output-file SHA-256 values.
- Re-running the builder with identical parameters either produces byte-identical artifacts or reports the existing immutable subset as unchanged.
- No chunking, LightRAG/Ollama indexing, model inference, or experiment execution is performed.

## Limitations

- The source corpus itself is imbalanced and covers only September–December 2023; stratification cannot create unavailable source/date coverage.
- Entity/topic diversity is approximated by rare lexical signatures, not by a learned NER or semantic model.
- BM25 is lexical. The model-free leakage audit catches exact evidence chains and normalized exact answers, but cannot prove that a paraphrase never conveys an answer.
- `null_query` has no gold evidence, so its source/category/date profile is only a lexical proxy and is explicitly kept separate from gold distributions.
