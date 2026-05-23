from pathlib import Path
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
TOKEN_RECORDS_CSV = PROJECT_DIR / "data_processed" / "token_records.csv"
OUTPUT_CSV = PROJECT_DIR / "data_processed" / "term_dictionary.csv"


# =========================
# 2. 判断无效 term
# =========================
def is_valid_term(term: str) -> bool:
    """
    过滤掉不适合进入索引文档的 term。
    """
    if not isinstance(term, str):
        return False

    term = term.strip().lower()

    if term == "":
        return False

    # 去掉纯省略号、纯点号
    if set(term) <= {"."}:
        return False

    # 去掉纯横线
    if set(term) <= {"-"}:
        return False

    # 去掉过短且没有意义的符号
    if len(term) == 1 and not term.isdigit():
        return False

    # 去掉明显无意义的值
    bad_terms = {
        "nan",
        "none",
        "unknown",
        "...",
        "..",
        ".",
        "-",
        "--",
        "---"
    }

    if term in bad_terms:
        return False

    return True


# =========================
# 3. 主程序
# =========================
def main():
    print("正在读取 token_records：", TOKEN_RECORDS_CSV)

    token_df = pd.read_csv(TOKEN_RECORDS_CSV)

    print("原始 token_records 行数：", len(token_df))

    # 基础清洗
    token_df["term"] = token_df["term"].fillna("").astype(str).str.strip().str.lower()
    token_df["doc_id"] = token_df["doc_id"].fillna("").astype(str)
    token_df["field"] = token_df["field"].fillna("").astype(str)

    # 过滤无效 term
    before = len(token_df)
    token_df = token_df[token_df["term"].apply(is_valid_term)].copy()
    after = len(token_df)

    print("过滤无效 term 记录数：", before - after)
    print("有效 token_records 行数：", after)

    # =========================
    # 4. 计算 total_tf
    # =========================
    tf_df = (
        token_df
        .groupby("term")
        .size()
        .reset_index(name="total_tf")
    )

    # =========================
    # 5. 计算 df：出现该词的不同文档数
    # =========================
    df_df = (
        token_df
        .drop_duplicates(subset=["term", "doc_id"])
        .groupby("term")
        .size()
        .reset_index(name="df")
    )

    # =========================
    # 6. 统计来源字段
    # =========================
    fields_df = (
        token_df
        .drop_duplicates(subset=["term", "field"])
        .groupby("term")["field"]
        .apply(lambda x: ", ".join(sorted(x)))
        .reset_index(name="fields")
    )

    field_count_df = (
        token_df
        .drop_duplicates(subset=["term", "field"])
        .groupby("term")
        .size()
        .reset_index(name="field_count")
    )

    # =========================
    # 7. 合并成索引文档
    # =========================
    term_dict = tf_df.merge(df_df, on="term", how="left")
    term_dict = term_dict.merge(fields_df, on="term", how="left")
    term_dict = term_dict.merge(field_count_df, on="term", how="left")

    # 按文档频率从高到低排序
    term_dict = term_dict.sort_values(
        by=["df", "total_tf"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # 生成 term_id
    term_dict.insert(
        0,
        "term_id",
        [f"T{i:06d}" for i in range(1, len(term_dict) + 1)]
    )

    # 字段顺序
    term_dict = term_dict[
        ["term_id", "term", "df", "total_tf", "fields", "field_count"]
    ]

    # =========================
    # 8. 保存
    # =========================
    term_dict.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n索引文档生成完成！")
    print("term_dictionary 词条数：", len(term_dict))
    print("已保存：", OUTPUT_CSV)

    print("\n前 30 个高频检索词：")
    print(term_dict.head(30))

    print("\n检查几个重要词：")
    important_terms = ["ntr", "wsjr", "nswt", "htp", "dd", "king", "god", "osiris"]
    for t in important_terms:
        hit = term_dict[term_dict["term"] == t]
        print("\nterm:", t)
        if len(hit) == 0:
            print("  未找到")
        else:
            print(hit)


if __name__ == "__main__":
    main()