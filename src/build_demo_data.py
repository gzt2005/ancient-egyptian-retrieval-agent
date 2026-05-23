from pathlib import Path
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

DATA_DIR = PROJECT_DIR / "data_processed"
DEMO_DIR = PROJECT_DIR / "data_demo"

MAIN_DOCS_CSV = DATA_DIR / "main_documents.csv"
TERM_DICTIONARY_CSV = DATA_DIR / "term_dictionary.csv"
INVERTED_FILE_CSV = DATA_DIR / "inverted_file.csv"
QUERY_EXPANSION_CSV = DATA_DIR / "query_expansion.csv"

DEMO_MAIN_DOCS_CSV = DEMO_DIR / "main_documents.csv"
DEMO_TERM_DICTIONARY_CSV = DEMO_DIR / "term_dictionary.csv"
DEMO_INVERTED_FILE_CSV = DEMO_DIR / "inverted_file.csv"
DEMO_QUERY_EXPANSION_CSV = DEMO_DIR / "query_expansion.csv"


# =========================
# 2. 展示版参数
# =========================
DEMO_CORPUS = [
    "bbawpyramidentexte",
    "bbawtotenlit",
    "tb",
    "sawlit"
]

MAX_DOCS = 20000


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print("读取主文档：", MAIN_DOCS_CSV)
    main_df = pd.read_csv(MAIN_DOCS_CSV, dtype=str).fillna("")

    print("全量主文档数量：", len(main_df))

    # =========================
    # 3. 筛选展示 corpus
    # =========================
    demo_main = main_df[main_df["corpus"].isin(DEMO_CORPUS)].copy()

    print("筛选指定 corpus 后数量：", len(demo_main))

    # 如果超过 MAX_DOCS，则抽样
    if len(demo_main) > MAX_DOCS:
        demo_main = (
            demo_main
            .groupby("corpus", group_keys=False)
            .apply(lambda x: x.sample(
                min(len(x), max(1, MAX_DOCS // len(DEMO_CORPUS))),
                random_state=42
            ))
            .reset_index(drop=True)
        )

    # 再限制一下总数，防止略超
    demo_main = demo_main.head(MAX_DOCS).copy()

    print("展示版主文档数量：", len(demo_main))

    demo_doc_ids = set(demo_main["doc_id"])

    # 保存展示版主文档
    demo_main.to_csv(DEMO_MAIN_DOCS_CSV, index=False, encoding="utf-8-sig")
    print("已保存展示版主文档：", DEMO_MAIN_DOCS_CSV)

    # =========================
    # 4. 根据 doc_id 筛选倒排档
    # =========================
    print("\n读取倒排档：", INVERTED_FILE_CSV)
    inverted_df = pd.read_csv(INVERTED_FILE_CSV, dtype=str, low_memory=False).fillna("")

    print("全量倒排档数量：", len(inverted_df))

    demo_inverted = inverted_df[inverted_df["doc_id"].isin(demo_doc_ids)].copy()

    print("展示版倒排档数量：", len(demo_inverted))

    demo_inverted.to_csv(DEMO_INVERTED_FILE_CSV, index=False, encoding="utf-8-sig")
    print("已保存展示版倒排档：", DEMO_INVERTED_FILE_CSV)

    # =========================
    # 5. 根据展示版倒排档重建索引文档
    # =========================
    print("\n重建展示版索引文档...")

    demo_inverted["tf"] = pd.to_numeric(demo_inverted["tf"], errors="coerce").fillna(0).astype(int)

    term_tf = (
        demo_inverted
        .groupby(["term_id", "term"])["tf"]
        .sum()
        .reset_index(name="total_tf")
    )

    term_df = (
        demo_inverted
        .drop_duplicates(subset=["term_id", "term", "doc_id"])
        .groupby(["term_id", "term"])
        .size()
        .reset_index(name="df")
    )

    fields_df = (
        demo_inverted
        .drop_duplicates(subset=["term_id", "term", "field"])
        .groupby(["term_id", "term"])["field"]
        .apply(lambda x: ", ".join(sorted(x)))
        .reset_index(name="fields")
    )

    field_count_df = (
        demo_inverted
        .drop_duplicates(subset=["term_id", "term", "field"])
        .groupby(["term_id", "term"])
        .size()
        .reset_index(name="field_count")
    )

    demo_term_dict = term_tf.merge(term_df, on=["term_id", "term"], how="left")
    demo_term_dict = demo_term_dict.merge(fields_df, on=["term_id", "term"], how="left")
    demo_term_dict = demo_term_dict.merge(field_count_df, on=["term_id", "term"], how="left")

    demo_term_dict = demo_term_dict.sort_values(
        by=["df", "total_tf"],
        ascending=[False, False]
    ).reset_index(drop=True)

    demo_term_dict = demo_term_dict[
        ["term_id", "term", "df", "total_tf", "fields", "field_count"]
    ]

    print("展示版索引词条数量：", len(demo_term_dict))

    demo_term_dict.to_csv(DEMO_TERM_DICTIONARY_CSV, index=False, encoding="utf-8-sig")
    print("已保存展示版索引文档：", DEMO_TERM_DICTIONARY_CSV)

    # =========================
    # 6. 复制中文扩展表
    # =========================
    query_expansion_df = pd.read_csv(QUERY_EXPANSION_CSV, dtype=str).fillna("")
    query_expansion_df.to_csv(DEMO_QUERY_EXPANSION_CSV, index=False, encoding="utf-8-sig")
    print("已保存展示版中文扩展表：", DEMO_QUERY_EXPANSION_CSV)

    # =========================
    # 7. 输出统计
    # =========================
    print("\n展示版数据生成完成！")
    print("data_demo 文件夹：", DEMO_DIR)

    print("\n展示版 corpus 分布：")
    print(demo_main["corpus"].value_counts())

    print("\n检查几个重要词是否还存在：")
    important_terms = ["ntr", "wsjr", "nswt", "htp", "king", "god", "osiris"]
    for term in important_terms:
        hit = demo_term_dict[demo_term_dict["term"] == term]
        if len(hit) == 0:
            print(term, ": 不存在")
        else:
            row = hit.iloc[0]
            print(term, f": df={row['df']}, total_tf={row['total_tf']}, fields={row['fields']}")


if __name__ == "__main__":
    main()