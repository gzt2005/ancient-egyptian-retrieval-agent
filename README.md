# Ancient Egyptian Text Retrieval Agent

A web-based retrieval agent for Ancient Egyptian textual materials, designed for Chinese users.

## Features

- DIALOG-style retrieval architecture:
  - Main Documents
  - Term Dictionary
  - Inverted File
- Chinese query expansion
- Ancient Egyptian transliteration search
- English/German translation keyword search
- Evidence-based result display with transliteration, lemmas, corpus, date, and findspot

## Demo Queries

Chinese:
- 神
- 奥西里斯
- 国王
- 太阳神
- 供奉
- 来世

Transliteration / English:
- ntr
- wsjr
- nswt
- osiris
- king

## Project Structure

```text
egypt_agent_project/
├── app.py
├── requirements.txt
├── README.md
├── data_demo/
│   ├── main_documents.csv
│   ├── term_dictionary.csv
│   ├── inverted_file.csv
│   └── query_expansion.csv
└── src/
    ├── build_main_documents.py
    ├── build_token_records.py
    ├── build_term_dictionary.py
    ├── build_inverted_file.py
    ├── build_query_expansion.py
    └── build_demo_data.py

- Field-weighted ranking mechanism prioritizing lemma, transliteration, MDC, and translation matches
- Chinese evidence explanation based on matched terms and matched fields