from pathlib import Path
import re
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
MAIN_DOCS_CSV = PROJECT_DIR / "data_processed" / "main_documents.csv"
OUTPUT_CSV = PROJECT_DIR / "data_processed" / "token_records.csv"


# =========================
# 2. 停用词表
# =========================
# 第一版先放常见英文/德文功能词，避免索引里全是 the / and / der / die
STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "his", "her", "their", "my", "your", "our", "at", "into", "about",
    "which", "who", "whom", "whose", "what", "when", "where", "how",

    # German common words
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
    "eines", "einem", "einen", "und", "oder", "zu", "in", "im", "im",
    "auf", "mit", "von", "für", "als", "ist", "sind", "war", "waren",
    "er", "sie", "es", "sein", "seine", "seiner", "nicht", "auch",

    # meaningless symbols
    "nan", "none", "unknown"
}


# =========================
# 3. 通用词项归一化
# =========================
def normalize_term(term: str) -> str:
    """
    统一检索词格式：
    - 转小写
    - 去掉多余标点
    - 保留字母、数字、连字符、点号
    """
    if not isinstance(term, str):
        return ""

    term = term.strip().lower()

    # 古埃及转写字符进一步归一化
    mapping = {
        "ꜣ": "a",
        "ꜥ": "a",
        "ȝ": "a",
        "ʾ": "a",
        "ḏ": "d",
        "ḥ": "h",
        "ḫ": "h",
        "ẖ": "h",
        "ḳ": "q",
        "š": "s",
        "ṯ": "t",
        "ṱ": "t",
        "ỉ": "i",
        "ī": "i",
        "ū": "u",
        "ꞽ": "i",
    }

    for old, new in mapping.items():
        term = term.replace(old, new)

    # 去除括号、特殊符号；保留英文数字、点、横杠、下划线
    term = re.sub(r"[^a-z0-9\.\-_]", "", term)

    return term


# =========================
# 4. 英文/德文译文分词
# =========================
def tokenize_translation(text: str):
    """
    对 translation 做简单词切分。
    """
    if not isinstance(text, str):
        return []

    raw_terms = re.split(r"[^A-Za-zÄÖÜäöüß0-9\-]+", text)

    terms = []
    for t in raw_terms:
        nt = normalize_term(t)

        if len(nt) <= 1:
            continue
        if nt in STOPWORDS:
            continue

        terms.append(nt)

    return terms


# =========================
# 5. 古埃及转写字段分词
# =========================
def tokenize_transliteration(text: str):
    """
    normalized_transliteration 已经较干净。
    这里按空格、连字符继续切分。
    """
    if not isinstance(text, str):
        return []

    # 同时保留完整片段和拆分片段
    raw_parts = re.split(r"\s+", text)

    terms = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        # 保留完整形式，如 htp-di-nswt
        full = normalize_term(part)
        if len(full) > 1 and full not in STOPWORDS:
            terms.append(full)

        # 再按连字符拆分，如 htp / di / nswt
        sub_parts = re.split(r"[\-_=\.]+", part)
        for sp in sub_parts:
            nt = normalize_term(sp)
            if len(nt) > 1 and nt not in STOPWORDS:
                terms.append(nt)

    return terms


# =========================
# 6. lemma_forms 分词
# =========================
def tokenize_lemmas(text: str):
    """
    lemma_forms 用 | 分隔。
    """
    if not isinstance(text, str):
        return []

    raw_terms = text.split("|")

    terms = []

    for t in raw_terms:
        nt = normalize_term(t)

        if len(nt) > 1 and nt not in STOPWORDS:
            terms.append(nt)

        # 对复合 lemma 进一步拆分
        for sp in re.split(r"[\-_=\.]+", t):
            nsp = normalize_term(sp)
            if len(nsp) > 1 and nsp not in STOPWORDS:
                terms.append(nsp)

    return terms


# =========================
# 7. mdc 分词
# =========================
def tokenize_mdc(text: str):
    """
    mdc 中有 Manuel de Codage 转写，保留完整形式和拆分形式。
    """
    if not isinstance(text, str):
        return []

    raw_parts = re.split(r"\s+", text)

    terms = []

    for part in raw_parts:
        full = normalize_term(part)
        if len(full) > 1 and full not in STOPWORDS:
            terms.append(full)

        for sp in re.split(r"[\-_=\.]+", part):
            nt = normalize_term(sp)
            if len(nt) > 1 and nt not in STOPWORDS:
                terms.append(nt)

    return terms


# =========================
# 8. 主程序
# =========================
def main():
    print("读取主文档：", MAIN_DOCS_CSV)
    df = pd.read_csv(MAIN_DOCS_CSV)

    print("主文档数量：", len(df))

    records = []

    for idx, row in df.iterrows():
        doc_id = row["doc_id"]

        field_tokenizers = {
            "translation": tokenize_translation,
            "normalized_transliteration": tokenize_transliteration,
            "lemma_forms": tokenize_lemmas,
            "mdc": tokenize_mdc
        }

        for field, tokenizer in field_tokenizers.items():
            text = row.get(field, "")
            tokens = tokenizer(text)

            for position, term in enumerate(tokens, start=1):
                records.append({
                    "doc_id": doc_id,
                    "term": term,
                    "field": field,
                    "position": position
                })

        # 每处理 10000 条提示一下，防止你以为卡住
        if (idx + 1) % 10000 == 0:
            print(f"已处理 {idx + 1} / {len(df)} 条主文档")

    token_df = pd.DataFrame(records)

    print("\n候选检索词记录生成完成")
    print("token_records 总行数：", len(token_df))

    # 删除空 term
    before = len(token_df)
    token_df = token_df[token_df["term"].fillna("").astype(str).str.strip() != ""]
    after = len(token_df)

    print("删除空 term 数：", before - after)
    print("剩余 token_records 数：", after)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    token_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n已保存：", OUTPUT_CSV)

    print("\n前 20 条 token_records：")
    print(token_df.head(20))

    print("\n出现频率最高的前 30 个 term：")
    print(token_df["term"].value_counts().head(30))


if __name__ == "__main__":
    main()