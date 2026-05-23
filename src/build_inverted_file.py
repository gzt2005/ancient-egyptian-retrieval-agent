from pathlib import Path
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

TOKEN_RECORDS_CSV = PROJECT_DIR / "data_processed" / "token_records.csv"
TERM_DICTIONARY_CSV = PROJECT_DIR / "data_processed" / "term_dictionary.csv"
OUTPUT_CSV = PROJECT_DIR / "data_processed" / "inverted_file.csv"


# =========================
# 2. 无效 term 判断
# 要和 build_term_dictionary.py 保持一致
# =========================
def is_valid_term(term: str) -> bool:
    if not isinstance(term, str):
        return False

    term = term.strip().lower()

    if term == "":
        return False

    if set(term) <= {"."}:
        return False

    if set(term) <= {"-"}:
        return False

    if len(term) == 1 and not term.isdigit():
        return False

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

    print("正在读取 term_dictionary：", TERM_DICTIONARY_CSV)
    term_dict = pd.read_csv(TERM_DICTIONARY_CSV)

    print("term_dictionary 词条数：", len(term_dict))

    # =========================
    # 4. 基础清洗
    # =========================
    token_df["term"] = token_df["term"].fillna("").astype(str).str.strip().str.lower()
    token_df["doc_id"] = token_df["doc_id"].fillna("").astype(str)
    token_df["field"] = token_df["field"].fillna("").astype(str)
    token_df["position"] = token_df["position"].fillna(0).astype(int)

    before = len(token_df)
    token_df = token_df[token_df["term"].apply(is_valid_term)].copy()
    after = len(token_df)

    print("过滤无效 term 记录数：", before - after)
    print("有效 token_records 行数：", after)

    # =========================
    # 5. 合并 term_id
    # =========================
    term_map = term_dict[["term_id", "term"]].copy()

    token_df = token_df.merge(term_map, on="term", how="left")

    missing_term_id = token_df["term_id"].isna().sum()
    print("未匹配到 term_id 的记录数：", missing_term_id)

    token_df = token_df.dropna(subset=["term_id"]).copy()

    # =========================
    # 6. 生成倒排档
    # 按 term_id + term + doc_id + field 聚合
    # tf = 在该文档该字段中出现次数
    # positions = 出现位置
    # =========================
    print("\n开始生成倒排档，这一步可能需要一点时间...")

    inverted_df = (
        token_df
        .groupby(["term_id", "term", "doc_id", "field"])["position"]
        .apply(lambda x: ",".join(map(str, sorted(x.tolist()))))
        .reset_index(name="positions")
    )

    inverted_df["tf"] = inverted_df["positions"].apply(
        lambda x: len(x.split(",")) if isinstance(x, str) and x else 0
    )

    # 调整字段顺序
    inverted_df = inverted_df[
        ["term_id", "term", "doc_id", "field", "tf", "positions"]
    ]

    # 排序：方便查看和检索
    inverted_df = inverted_df.sort_values(
        by=["term_id", "doc_id", "field"]
    ).reset_index(drop=True)

    # =========================
    # 7. 保存倒排档
    # =========================
    inverted_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n倒排档生成完成！")
    print("inverted_file 行数：", len(inverted_df))
    print("已保存：", OUTPUT_CSV)

    print("\n前 20 行倒排档预览：")
    print(inverted_df.head(20))

    # =========================
    # 8. 检查重要词倒排情况
    # =========================
    important_terms = ["ntr", "wsjr", "nswt", "htp", "dd", "king", "god", "osiris"]

    print("\n重要词倒排记录数量检查：")
    for t in important_terms:
        count = len(inverted_df[inverted_df["term"] == t])
        print(t, ":", count)


if __name__ == "__main__":
    main()