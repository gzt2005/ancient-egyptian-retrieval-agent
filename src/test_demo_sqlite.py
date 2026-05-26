from pathlib import Path
import sqlite3
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
DB_PATH = PROJECT_DIR / "database_demo" / "egypt_demo.db"


# =========================
# 2. 字段权重
# =========================
FIELD_WEIGHTS = {
    "lemma_forms": 5,
    "normalized_transliteration": 4,
    "mdc": 3,
    "translation": 2
}


# =========================
# 3. 单词检索函数
# =========================
def sqlite_search_single_term(conn, term, top_k=10):
    """
    使用 SQLite 完成单个 term 的检索。
    """
    term = term.strip().lower()

    # 1. 查索引文档
    term_info = pd.read_sql_query(
        """
        SELECT term_id, term, df, total_tf, fields
        FROM term_dictionary
        WHERE term = ?
        LIMIT 1;
        """,
        conn,
        params=(term,)
    )

    if len(term_info) == 0:
        print(f"未在 term_dictionary 中找到：{term}")
        return pd.DataFrame()

    term_id = term_info.iloc[0]["term_id"]

    print("\n命中索引词：")
    print(term_info)

    # 2. 查倒排档
    postings = pd.read_sql_query(
        """
        SELECT term_id, term, doc_id, field, tf, positions
        FROM inverted_file
        WHERE term_id = ?;
        """,
        conn,
        params=(term_id,)
    )

    if len(postings) == 0:
        print(f"倒排档中没有记录：{term}")
        return pd.DataFrame()

    # 3. 字段加权
    postings["field_weight"] = postings["field"].map(FIELD_WEIGHTS).fillna(1)
    postings["weighted_tf"] = postings["tf"] * postings["field_weight"]

    # 4. 聚合 doc_id
    doc_scores = (
        postings
        .groupby("doc_id")
        .agg(
            total_tf=("tf", "sum"),
            weighted_score=("weighted_tf", "sum"),
            matched_fields=("field", lambda x: ", ".join(sorted(set(x)))),
            positions=("positions", lambda x: " | ".join(map(str, x)))
        )
        .reset_index()
    )

    doc_scores["matched_term"] = term

    doc_scores = doc_scores.sort_values(
        by=["weighted_score", "total_tf", "doc_id"],
        ascending=[False, False, True]
    ).head(top_k)

    # 5. 回主文档
    doc_ids = doc_scores["doc_id"].tolist()

    placeholders = ",".join(["?"] * len(doc_ids))

    main_docs = pd.read_sql_query(
        f"""
        SELECT *
        FROM main_documents
        WHERE doc_id IN ({placeholders});
        """,
        conn,
        params=doc_ids
    )

    results = doc_scores.merge(main_docs, on="doc_id", how="left")

    # 保持排序
    results["rank_order"] = range(1, len(results) + 1)

    return results


# =========================
# 4. 主程序
# =========================
def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"数据库不存在：{DB_PATH}")

    print("正在连接 SQLite 数据库：", DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    try:
        while True:
            query = input("\n请输入检索词，例如 ntr / wsjr / osiris / king，输入 q 退出：").strip()

            if query.lower() == "q":
                print("已退出。")
                break

            if not query:
                print("请输入有效检索词。")
                continue

            results = sqlite_search_single_term(conn, query, top_k=5)

            if len(results) == 0:
                print("没有检索结果。")
                continue

            print("\n" + "=" * 100)
            print(f"检索词：{query}")
            print(f"返回结果数量：{len(results)}")

            for _, row in results.iterrows():
                print("\n" + "-" * 100)
                print("rank:", row["rank_order"])
                print("doc_id:", row["doc_id"])
                print("weighted_score:", row["weighted_score"])
                print("total_tf:", row["total_tf"])
                print("matched_fields:", row["matched_fields"])
                print("corpus:", row.get("corpus", ""))
                print("date:", row.get("date", ""))
                print("findspot:", row.get("findspot", ""))
                print("translation:", row.get("translation", ""))
                print("transliteration:", row.get("transliteration", ""))
                print("lemma_forms:", row.get("lemma_forms", ""))
                print("mdc:", row.get("mdc", ""))

    finally:
        conn.close()


if __name__ == "__main__":
    main()