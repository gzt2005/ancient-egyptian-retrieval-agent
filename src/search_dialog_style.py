from pathlib import Path
import pandas as pd
import re


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

MAIN_DOCS_CSV = PROJECT_DIR / "data_processed" / "main_documents.csv"
TERM_DICTIONARY_CSV = PROJECT_DIR / "data_processed" / "term_dictionary.csv"
INVERTED_FILE_CSV = PROJECT_DIR / "data_processed" / "inverted_file.csv"


# =========================
# 2. 归一化用户输入
# =========================
def normalize_query_term(term: str) -> str:
    """
    将用户输入的检索词归一化。
    例如：
    nṯr -> ntr
    ḏd -> dd
    Wsjr -> wsjr
    """
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


# =========================
# 3. 加载数据
# =========================
print("正在加载主文档...")
main_df = pd.read_csv(MAIN_DOCS_CSV)

print("正在加载索引文档...")
term_dict = pd.read_csv(TERM_DICTIONARY_CSV)

print("正在加载倒排档...")
inverted_df = pd.read_csv(INVERTED_FILE_CSV)

print("加载完成！")
print("主文档数量：", len(main_df))
print("索引词条数量：", len(term_dict))
print("倒排档记录数：", len(inverted_df))


# =========================
# 4. DIALOG 风格检索函数
# =========================
def dialog_search(query, top_k=10):
    """
    DIALOG 风格检索：
    1. 归一化用户输入
    2. 查索引文档 term_dictionary
    3. 查倒排档 inverted_file
    4. 回主文档 main_documents
    """

    query_norm = normalize_query_term(query)

    if not query_norm:
        return {
            "query": query,
            "query_norm": query_norm,
            "term_info": None,
            "results": pd.DataFrame()
        }

    # Step 1：查索引文档
    term_info = term_dict[term_dict["term"] == query_norm]

    if len(term_info) == 0:
        return {
            "query": query,
            "query_norm": query_norm,
            "term_info": None,
            "results": pd.DataFrame()
        }

    term_info_row = term_info.iloc[0]
    term_id = term_info_row["term_id"]

    # Step 2：查倒排档
    postings = inverted_df[inverted_df["term_id"] == term_id].copy()

    if len(postings) == 0:
        return {
            "query": query,
            "query_norm": query_norm,
            "term_info": term_info_row,
            "results": pd.DataFrame()
        }

    # Step 3：按照 doc_id 聚合
    # 一个词可能在同一个 doc 的多个字段中出现
    doc_scores = (
        postings
        .groupby("doc_id")
        .agg(
            total_tf=("tf", "sum"),
            matched_fields=("field", lambda x: ", ".join(sorted(set(x)))),
            positions=("positions", lambda x: " | ".join(map(str, x)))
        )
        .reset_index()
    )

    # 排序规则：
    # 1. total_tf 高的优先
    # 2. 同等情况下 doc_id 靠前
    doc_scores = doc_scores.sort_values(
        by=["total_tf", "doc_id"],
        ascending=[False, True]
    ).head(top_k)

    # Step 4：回主文档
    results = doc_scores.merge(main_df, on="doc_id", how="left")

    return {
        "query": query,
        "query_norm": query_norm,
        "term_info": term_info_row,
        "results": results
    }


# =========================
# 5. 打印结果函数
# =========================
def print_search_result(search_output):
    query = search_output["query"]
    query_norm = search_output["query_norm"]
    term_info = search_output["term_info"]
    results = search_output["results"]

    print("\n" + "#" * 90)
    print("原始检索词：", query)
    print("归一化检索词：", query_norm)

    if term_info is None:
        print("索引文档中未找到该检索词。")
        return

    print("\n【索引文档信息】")
    print("term_id:", term_info["term_id"])
    print("term:", term_info["term"])
    print("df:", term_info["df"])
    print("total_tf:", term_info["total_tf"])
    print("fields:", term_info["fields"])
    print("field_count:", term_info["field_count"])

    print("\n【检索结果】")
    print("返回文档数：", len(results))

    if len(results) == 0:
        print("倒排档中没有找到对应文档。")
        return

    for _, row in results.iterrows():
        print("\n" + "=" * 90)
        print("doc_id:", row["doc_id"])
        print("sentence_id:", row["sentence_id"])
        print("匹配字段:", row["matched_fields"])
        print("total_tf:", row["total_tf"])
        print("corpus:", row["corpus"])
        print("date:", row["date"])
        print("findspot:", row["findspot"])
        print("translation:", row["translation"])
        print("transliteration:", row["transliteration"])
        print("normalized:", row["normalized_transliteration"])
        print("lemma_forms:", row["lemma_forms"])
        print("mdc:", row["mdc"])


# =========================
# 6. 交互式运行
# =========================
def main():
    print("\nDIALOG 风格古埃及文字检索系统已启动")
    print("你可以输入：ntr / nṯr / wsjr / Wsjr / nswt / king / god / osiris")
    print("输入 q 退出")

    while True:
        query = input("\n请输入检索词：").strip()

        if query.lower() == "q":
            print("已退出。")
            break

        output = dialog_search(query, top_k=10)
        print_search_result(output)


if __name__ == "__main__":
    main()