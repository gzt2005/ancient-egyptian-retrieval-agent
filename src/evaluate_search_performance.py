from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

DATA_DEMO_DIR = PROJECT_DIR / "data_demo"
SEMANTIC_DIR = PROJECT_DIR / "data_semantic_demo"
OUTPUT_DIR = PROJECT_DIR / "evaluation_results"

MAIN_DOCS_CSV = DATA_DEMO_DIR / "main_documents.csv"
TERM_DICTIONARY_CSV = DATA_DEMO_DIR / "term_dictionary.csv"
INVERTED_FILE_CSV = DATA_DEMO_DIR / "inverted_file.csv"
QUERY_EXPANSION_CSV = DATA_DEMO_DIR / "query_expansion.csv"

SEMANTIC_EMBEDDINGS_PATH = SEMANTIC_DIR / "semantic_embeddings.npy"
SEMANTIC_METADATA_PATH = SEMANTIC_DIR / "semantic_metadata.csv"

OUTPUT_CSV = OUTPUT_DIR / "evaluation_results.csv"

SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================
# 2. 工具函数
# =========================
def normalize_query_term(term: str) -> str:
    if not isinstance(term, str):
        return ""

    term = term.strip().lower()

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

    term = re.sub(r"[^a-z0-9\.\-_]", "", term)
    return term


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


# =========================
# 3. 加载关键词检索数据
# =========================
def load_keyword_data():
    print("加载关键词检索数据...")

    main_df = pd.read_csv(MAIN_DOCS_CSV, dtype=str).fillna("")
    term_dict = pd.read_csv(TERM_DICTIONARY_CSV, dtype=str).fillna("")
    inverted_df = pd.read_csv(INVERTED_FILE_CSV, dtype=str, low_memory=False).fillna("")
    query_expansion_df = pd.read_csv(QUERY_EXPANSION_CSV, dtype=str).fillna("")

    inverted_df["tf"] = pd.to_numeric(inverted_df["tf"], errors="coerce").fillna(0).astype(int)

    if "df" in term_dict.columns:
        term_dict["df"] = pd.to_numeric(term_dict["df"], errors="coerce").fillna(0).astype(int)

    if "total_tf" in term_dict.columns:
        term_dict["total_tf"] = pd.to_numeric(term_dict["total_tf"], errors="coerce").fillna(0).astype(int)

    print("主文档数量：", len(main_df))
    print("索引词条数量：", len(term_dict))
    print("倒排记录数量：", len(inverted_df))
    print("中文主题数量：", len(query_expansion_df))

    return main_df, term_dict, inverted_df, query_expansion_df


# =========================
# 4. 中文查询扩展
# =========================
def expand_chinese_query(query, query_expansion_df):
    query = query.strip()

    hit = query_expansion_df[query_expansion_df["query_zh"] == query]

    if len(hit) == 0:
        hit = query_expansion_df[
            query_expansion_df["query_zh"].apply(lambda x: x in query or query in x)
        ]

    if len(hit) == 0:
        return [], "未找到中文扩展词"

    row = hit.iloc[0]

    expanded_terms = [
        normalize_query_term(t)
        for t in str(row["expanded_terms"]).split(",")
        if normalize_query_term(t)
    ]

    explanation = row.get("explanation_zh", "")

    return expanded_terms, explanation


# =========================
# 5. 单词倒排检索
# =========================
def dialog_search_single_term(term, term_dict, inverted_df):
    term_norm = normalize_query_term(term)

    if not term_norm:
        return pd.DataFrame(), None

    term_info = term_dict[term_dict["term"] == term_norm]

    if len(term_info) == 0:
        return pd.DataFrame(), None

    term_info_row = term_info.iloc[0]
    term_id = term_info_row["term_id"]

    postings = inverted_df[inverted_df["term_id"] == term_id].copy()

    if len(postings) == 0:
        return pd.DataFrame(), term_info_row

    field_weights = {
        "lemma_forms": 5,
        "normalized_transliteration": 4,
        "mdc": 3,
        "translation": 2
    }

    postings["field_weight"] = postings["field"].map(field_weights).fillna(1)
    postings["weighted_tf"] = postings["tf"] * postings["field_weight"]

    doc_scores = (
        postings
        .groupby("doc_id")
        .agg(
            total_tf=("tf", "sum"),
            weighted_score=("weighted_tf", "sum"),
            matched_fields=("field", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
    )

    doc_scores["matched_term"] = term_norm

    return doc_scores, term_info_row


# =========================
# 6. 关键词检索
# =========================
def keyword_search(query, top_k, main_df, term_dict, inverted_df, query_expansion_df):
    query = query.strip()

    if contains_chinese(query):
        expanded_terms, _ = expand_chinese_query(query, query_expansion_df)

        if not expanded_terms:
            return pd.DataFrame()

        all_doc_scores = []

        for term in expanded_terms:
            doc_scores, _ = dialog_search_single_term(term, term_dict, inverted_df)

            if len(doc_scores) > 0:
                all_doc_scores.append(doc_scores)

        if not all_doc_scores:
            return pd.DataFrame()

        combined = pd.concat(all_doc_scores, ignore_index=True)

        combined_grouped = (
            combined
            .groupby("doc_id")
            .agg(
                total_tf=("total_tf", "sum"),
                weighted_score=("weighted_score", "sum"),
                matched_terms=("matched_term", lambda x: ", ".join(sorted(set(x)))),
                matched_fields=("matched_fields", lambda x: ", ".join(sorted(set(", ".join(x).split(", "))))),
            )
            .reset_index()
        )

        combined_grouped["matched_term_count"] = combined_grouped["matched_terms"].apply(
            lambda x: len(x.split(", ")) if isinstance(x, str) and x else 0
        )

        combined_grouped = combined_grouped.sort_values(
            by=["matched_term_count", "weighted_score", "total_tf"],
            ascending=[False, False, False]
        ).head(top_k)

        results = combined_grouped.merge(main_df, on="doc_id", how="left")

        return results

    else:
        term_norm = normalize_query_term(query)
        doc_scores, _ = dialog_search_single_term(term_norm, term_dict, inverted_df)

        if len(doc_scores) == 0:
            return pd.DataFrame()

        doc_scores = doc_scores.sort_values(
            by=["weighted_score", "total_tf", "doc_id"],
            ascending=[False, False, True]
        ).head(top_k)

        results = doc_scores.merge(main_df, on="doc_id", how="left")

        return results


# =========================
# 7. 加载语义检索资源
# =========================
def load_semantic_data():
    print("\n加载 AI 语义检索资源...")

    embeddings = np.load(SEMANTIC_EMBEDDINGS_PATH)
    metadata = pd.read_csv(SEMANTIC_METADATA_PATH, dtype=str).fillna("")

    print("语义向量 shape：", embeddings.shape)
    print("语义元数据数量：", len(metadata))

    print("加载语义模型：", SEMANTIC_MODEL_NAME)
    model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    return model, embeddings, metadata


# =========================
# 8. AI 语义检索
# =========================
def semantic_search(query, top_k, model, embeddings, metadata):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(embeddings, query_embedding[0])
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()
    results["semantic_score"] = scores[top_indices]

    return results


# =========================
# 9. 运行单次测试
# =========================
def run_single_eval(
    mode,
    query,
    top_k,
    main_df,
    term_dict,
    inverted_df,
    query_expansion_df,
    semantic_model,
    semantic_embeddings,
    semantic_metadata
):
    start_time = time.perf_counter()

    if mode == "关键词检索":
        results = keyword_search(
            query=query,
            top_k=top_k,
            main_df=main_df,
            term_dict=term_dict,
            inverted_df=inverted_df,
            query_expansion_df=query_expansion_df
        )
        score_col = "weighted_score"

    else:
        results = semantic_search(
            query=query,
            top_k=top_k,
            model=semantic_model,
            embeddings=semantic_embeddings,
            metadata=semantic_metadata
        )
        score_col = "semantic_score"

    elapsed_time = time.perf_counter() - start_time

    if len(results) > 0:
        top_row = results.iloc[0]
        top_doc_id = top_row.get("doc_id", "")
        top_score = top_row.get(score_col, "")
        top_corpus = top_row.get("corpus", "")
        top_translation = top_row.get("translation", "")
    else:
        top_doc_id = ""
        top_score = ""
        top_corpus = ""
        top_translation = ""

    return {
        "mode": mode,
        "query": query,
        "top_k": top_k,
        "elapsed_time_sec": round(elapsed_time, 4),
        "result_count": int(len(results)),
        "top_doc_id": top_doc_id,
        "top_score": top_score,
        "top_corpus": top_corpus,
        "top_translation_preview": str(top_translation)[:120]
    }


# =========================
# 10. 主函数
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top_k = 10

    test_queries = [
        ("关键词检索", "神"),
        ("关键词检索", "奥西里斯"),
        ("关键词检索", "太阳神"),
        ("关键词检索", "ntr"),
        ("关键词检索", "wsjr"),
        ("关键词检索", "osiris"),
        ("AI语义检索", "太阳神和国王"),
        ("AI语义检索", "Osiris and afterlife"),
        ("AI语义检索", "offering rituals"),
        ("AI语义检索", "texts about gods and kingship"),
        ("AI语义检索", "enemies of Osiris"),
    ]

    main_df, term_dict, inverted_df, query_expansion_df = load_keyword_data()
    semantic_model, semantic_embeddings, semantic_metadata = load_semantic_data()

    print("\n开始批量性能测评...")
    records = []

    # 第一轮可能包含模型预热影响，所以额外先跑一次 AI 查询作为 warm-up
    print("\n语义模型预热中...")
    _ = semantic_search(
        query="warm up query",
        top_k=5,
        model=semantic_model,
        embeddings=semantic_embeddings,
        metadata=semantic_metadata
    )
    print("预热完成。")

    for mode, query in test_queries:
        print(f"\n测试：{mode} | {query}")

        record = run_single_eval(
            mode=mode,
            query=query,
            top_k=top_k,
            main_df=main_df,
            term_dict=term_dict,
            inverted_df=inverted_df,
            query_expansion_df=query_expansion_df,
            semantic_model=semantic_model,
            semantic_embeddings=semantic_embeddings,
            semantic_metadata=semantic_metadata
        )

        records.append(record)

        print(
            f"耗时：{record['elapsed_time_sec']} 秒 | "
            f"返回：{record['result_count']} 条 | "
            f"Top-1：{record['top_doc_id']} | "
            f"Score：{record['top_score']}"
        )

    eval_df = pd.DataFrame(records)
    eval_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n批量性能测评完成！")
    print("结果文件：", OUTPUT_CSV)

    print("\n测评结果预览：")
    print(eval_df)

    print("\n按模式统计平均耗时：")
    print(eval_df.groupby("mode")["elapsed_time_sec"].mean())


if __name__ == "__main__":
    main()