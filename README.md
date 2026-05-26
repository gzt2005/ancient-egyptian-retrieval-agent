# Ancient Egyptian Text Retrieval System

A web-based intelligent retrieval system for Ancient Egyptian textual materials, designed for Chinese users.

## Version

Version 1.0

## Features

- SQLite-backed keyword retrieval
- DIALOG-style retrieval architecture:
  - Main Documents
  - Term Dictionary
  - Inverted File
- Chinese query expansion
- Ancient Egyptian transliteration normalization
- Field-weighted ranking
- AI semantic retrieval based on sentence embeddings
- Evidence-based result display
- System performance evaluation module

## Data Structure

- `database_demo/egypt_demo.db`: SQLite retrieval database
- `data_demo/`: demo CSV data
- `data_semantic_demo/`: semantic vector index
- `evaluation_results/`: performance evaluation results
- `src/`: data processing and evaluation scripts

## Demo Queries

Keyword search:

- 神
- 奥西里斯
- 国王
- ntr
- wsjr
- osiris

AI semantic search:

- 太阳神和国王
- Osiris and afterlife
- offering rituals
- texts about gods and kingship
- enemies of Osiris

## Performance Evaluation

The system includes a performance evaluation module comparing keyword-based retrieval and AI semantic retrieval. The evaluation records query mode, elapsed time, result count, Top-1 document ID, Top-1 score, corpus, and translation preview.

After semantic model warm-up, the current demo shows:

| Mode | Average Search Time |
|---|---:|
| AI Semantic Search | 0.0173 s |
| Keyword Search | 0.0859 s |

The semantic search time reflects warm-start retrieval after the embedding model and semantic index have been loaded. The first semantic query in the web app may take longer due to model loading.

## Future Work

- Hieroglyph image retrieval
- Vector glyph matching
- Gardiner sign recognition
- Image-to-text retrieval linkage