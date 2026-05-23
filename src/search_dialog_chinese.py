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
QUERY_EXPANSION_CSV = PROJECT_DIR / "data_processed" / "query_expansion.csv"


# =========================
# 2. 归一化函数
# =========================
def normalize_query_term(term: str) -> str:
    """
    将用户输入或扩展词归一化。
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


def contains_chinese(text: str) -> bool:
    """
    判断用户输入中是否包含中文。
    """
    return bool(re.search(r"[\u4e00-\u9fff]", text))


# =========================
# 3. 加载数据
# =========================
print("正在加载主文档...")
main_df = pd.read_csv(MAIN_DOCS_CSV)

print("正在加载索引文档...")
term_dict = pd.read_csv(TERM_DICTIONARY_CSV)

print("正在加载倒排档...")
inverted_df = pd.read_csv(INVERTED_FILE_CSV)

print("正在加载中文查询扩展表...")
query_expansion_df = pd.read_csv(QUERY_EXPANSION_CSV)

print("加载完成！")
print("主文档数量：", len(main_df))
print("索引词条数量：", len(term_dict))
print("倒排档记录数：", len(inverted_df))
print("中文主题数量：", len(query_expansion_df))


# =========================
# 4. 中文查询扩展
# =========================
def expand_chinese_query(query: str):
    """
    如果用户输入中文，则查 query_expansion.csv。
    返回：
    - expanded_terms: 扩展后的英文/德文/古埃及词项列表
    - explanation: 中文说明
    """
    query = query.strip()

    # 精确匹配
    hit = query_expansion_df[query_expansion_df["query_zh"] == query]

    # 如果精确匹配失败，尝试包含匹配
    if len(hit) == 0:
        hit = query_expansion_df[
            query_expansion_df["query_zh"].apply(lambda x: x in query or query in x)
        ]

    if len(hit) == 0:
        return [], "未在中文查询扩展表中找到该主题。"

    row = hit.iloc[0]

    expanded_terms = [
        normalize_query_term(t)
        for t in str(row["expanded_terms"]).split(",")
        if normalize_query_term(t)
    ]

    explanation = row["explanation_zh"]

    return expanded_terms, explanation


# =========================
# 5. 单词 DIALOG 检索
# =========================
def dialog_search_single_term(term, top_k=20):
    """
    对单个 term 执行 DIALOG 风格检索。
    返回 doc_scores。
    """
    term_norm = normalize_query_term(term)

    if not term_norm:
        return pd.DataFrame(), None

    # Step 1: 查索引文档
    term_info = term_dict[term_dict["term"] == term_norm]

    if len(term_info) == 0:
        return pd.DataFrame(), None

    term_info_row = term_info.iloc[0]
    term_id = term_info_row["term_id"]

    # Step 2: 查倒排档
    postings = inverted_df[inverted_df["term_id"] == term_id].copy()

    if len(postings) == 0:
        return pd.DataFrame(), term_info_row

    # Step 3: 按 doc_id 聚合
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

    doc_scores["matched_term"] = term_norm

    return doc_scores, term_info_row


# =========================
# 6. 中文/英文统一检索
# =========================
def search(query, top_k=10):
    """
    支持：
    - 中文主题检索：神 / 奥西里斯 / 国王
    - 英文检索：god / osiris / king
    - 古埃及转写检索：ntr / wsjr / nswt
    """

    query = query.strip()

    if contains_chinese(query):
        expanded_terms, explanation = expand_chinese_query(query)

        if not expanded_terms:
            return {
                "query": query,
                "mode": "中文检索",
                "expanded_terms": [],
                "explanation": explanation,
                "term_infos": [],
                "results": pd.DataFrame()
            }

        all_doc_scores = []
        term_infos = []

        for term in expanded_terms:
            doc_scores, term_info = dialog_search_single_term(term, top_k=50)

            if term_info is not None:
                term_infos.append(term_info)

            if len(doc_scores) > 0:
                all_doc_scores.append(doc_scores)

        if not all_doc_scores:
            return {
                "query": query,
                "mode": "中文检索",
                "expanded_terms": expanded_terms,
                "explanation": explanation,
                "term_infos": term_infos,
                "results": pd.DataFrame()
            }

        combined = pd.concat(all_doc_scores, ignore_index=True)

        # 合并多个扩展词的结果
        combined_grouped = (
            combined
            .groupby("doc_id")
            .agg(
                total_tf=("total_tf", "sum"),
                matched_terms=("matched_term", lambda x: ", ".join(sorted(set(x)))),
                matched_fields=("matched_fields", lambda x: ", ".join(sorted(set(", ".join(x).split(", "))))),
                positions=("positions", lambda x: " || ".join(map(str, x)))
            )
            .reset_index()
        )

        # 排序：命中词越多、词频越高越靠前
        combined_grouped["matched_term_count"] = combined_grouped["matched_terms"].apply(
            lambda x: len(x.split(", ")) if isinstance(x, str) and x else 0
        )

        combined_grouped = combined_grouped.sort_values(
            by=["matched_term_count", "total_tf"],
            ascending=[False, False]
        ).head(top_k)

        results = combined_grouped.merge(main_df, on="doc_id", how="left")

        return {
            "query": query,
            "mode": "中文检索",
            "expanded_terms": expanded_terms,
            "explanation": explanation,
            "term_infos": term_infos,
            "results": results
        }

    else:
        # 非中文：按单 term 检索
        term_norm = normalize_query_term(query)
        doc_scores, term_info = dialog_search_single_term(term_norm, top_k=top_k)

        if len(doc_scores) == 0:
            return {
                "query": query,
                "mode": "普通检索",
                "expanded_terms": [term_norm],
                "explanation": "",
                "term_infos": [term_info] if term_info is not None else [],
                "results": pd.DataFrame()
            }

        doc_scores = doc_scores.sort_values(
            by=["total_tf", "doc_id"],
            ascending=[False, True]
        ).head(top_k)

        results = doc_scores.merge(main_df, on="doc_id", how="left")

        return {
            "query": query,
            "mode": "普通检索",
            "expanded_terms": [term_norm],
            "explanation": "",
            "term_infos": [term_info] if term_info is not None else [],
            "results": results
        }


# =========================
# 7. 输出结果
# =========================
def print_results(output):
    print("\n" + "#" * 100)
    print("原始查询：", output["query"])
    print("检索模式：", output["mode"])
    print("扩展词：", ", ".join(output["expanded_terms"]))

    if output["explanation"]:
        print("中文说明：", output["explanation"])

    if output["term_infos"]:
        print("\n【命中的索引词】")
        for info in output["term_infos"]:
            if info is None:
                continue
            print(
                f"- {info['term']} | term_id={info['term_id']} | "
                f"df={info['df']} | total_tf={info['total_tf']} | fields={info['fields']}"
            )

    results = output["results"]

    print("\n【检索结果】")
    print("返回文档数：", len(results))

    if len(results) == 0:
        print("没有找到结果。")
        return

    for _, row in results.iterrows():
        print("\n" + "=" * 100)
        print("doc_id:", row["doc_id"])
        print("sentence_id:", row["sentence_id"])
        print("命中词:", row.get("matched_terms", row.get("matched_term", "")))
        print("匹配字段:", row["matched_fields"])
        print("total_tf:", row["total_tf"])
        print("corpus:", row["corpus"])
        print("date:", row["date"])
        print("findspot:", row["findspot"])
        print("原始译文:", row["translation"])
        print("古埃及转写:", row["transliteration"])
        print("归一化转写:", row["normalized_transliteration"])
        print("lemma_forms:", row["lemma_forms"])
        print("mdc:", row["mdc"])

        # 简单中文提示，不做强行翻译
        print("中文提示: 该结果为原始语料证据，可结合命中词和译文判断其主题相关性。")


# =========================
# 8. 主程序
# =========================
def main():
    print("\n支持中文检索的古埃及 DIALOG 风格检索系统已启动")
    print("中文示例：神 / 奥西里斯 / 国王 / 太阳神 / 供奉 / 来世 / 死者")
    print("英文示例：god / osiris / king")
    print("转写示例：ntr / wsjr / nswt / htp")
    print("输入 q 退出")

    while True:
        query = input("\n请输入检索词：").strip()

        if query.lower() == "q":
            print("已退出。")
            break

        output = search(query, top_k=10)
        print_results(output)


if __name__ == "__main__":
    main()