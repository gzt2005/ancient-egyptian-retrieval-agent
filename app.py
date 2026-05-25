from pathlib import Path
import re
import pandas as pd
import streamlit as st


# =========================
# 1. 页面设置
# =========================
st.set_page_config(
    page_title="古埃及文字检索智能体",
    page_icon="𓂀",
    layout="wide"
)


# =========================
# 2. 路径设置
# =========================
PROJECT_DIR = Path(__file__).parent

MAIN_DOCS_CSV = PROJECT_DIR / "data_demo" / "main_documents.csv"
TERM_DICTIONARY_CSV = PROJECT_DIR / "data_demo" / "term_dictionary.csv"
INVERTED_FILE_CSV = PROJECT_DIR / "data_demo" / "inverted_file.csv"
QUERY_EXPANSION_CSV = PROJECT_DIR / "data_demo" / "query_expansion.csv"


# =========================
# 3. 工具函数
# =========================
def normalize_query_term(term: str) -> str:
    """
    将用户输入或扩展词归一化。
    例如：
    nṯr -> ntr
    ḏd -> dd
    ꜥnḫ -> anh
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


def contains_chinese(text: str) -> bool:
    """
    判断是否包含中文。
    """
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def generate_chinese_hint(query, matched_terms, matched_fields):
    """
    根据查询词、命中词和命中字段生成简单中文解释。
    """
    if not matched_terms:
        matched_terms = "相关词项"

    if not matched_fields:
        matched_fields = "相关字段"

    if contains_chinese(query):
        return (
            f"该结果与“{query}”主题相关，系统通过扩展词 {matched_terms} 命中原始语料。"
            f"命中字段包括 {matched_fields}，可结合原始译文、古埃及转写、lemma 和 MDC 判断其主题相关性。"
        )

    return (
        f"该结果命中检索词 {matched_terms}，命中字段包括 {matched_fields}。"
        f"可结合原始译文、古埃及转写、lemma 和 MDC 判断其文本证据价值。"
    )


# =========================
# 4. 缓存加载数据
# =========================
@st.cache_data(show_spinner="正在加载主文档、索引文档和倒排档，请稍等...")
def load_data():
    main_df = pd.read_csv(MAIN_DOCS_CSV, dtype=str).fillna("")
    term_dict = pd.read_csv(TERM_DICTIONARY_CSV, dtype=str).fillna("")
    inverted_df = pd.read_csv(INVERTED_FILE_CSV, dtype=str, low_memory=False).fillna("")
    query_expansion_df = pd.read_csv(QUERY_EXPANSION_CSV, dtype=str).fillna("")

    # 数值字段转回来，方便排序和展示
    if "tf" in inverted_df.columns:
        inverted_df["tf"] = pd.to_numeric(
            inverted_df["tf"], errors="coerce"
        ).fillna(0).astype(int)

    if "token_count" in main_df.columns:
        main_df["token_count"] = pd.to_numeric(
            main_df["token_count"], errors="coerce"
        ).fillna(0).astype(int)

    if "df" in term_dict.columns:
        term_dict["df"] = pd.to_numeric(
            term_dict["df"], errors="coerce"
        ).fillna(0).astype(int)

    if "total_tf" in term_dict.columns:
        term_dict["total_tf"] = pd.to_numeric(
            term_dict["total_tf"], errors="coerce"
        ).fillna(0).astype(int)

    return main_df, term_dict, inverted_df, query_expansion_df


main_df, term_dict, inverted_df, query_expansion_df = load_data()


# =========================
# 5. 中文查询扩展
# =========================
def expand_chinese_query(query: str):
    """
    如果用户输入中文，则查询 query_expansion.csv，
    将中文主题词扩展为英文、德文和古埃及转写词。
    """
    query = query.strip()

    # 精确匹配
    hit = query_expansion_df[query_expansion_df["query_zh"] == query]

    # 包含匹配
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

    explanation = row.get("explanation_zh", "")

    return expanded_terms, explanation


# =========================
# 6. 单词 DIALOG 检索
# =========================
def dialog_search_single_term(term):
    """
    对单个 term 执行 DIALOG 风格检索，并加入字段加权排序。
    """
    term_norm = normalize_query_term(term)

    if not term_norm:
        return pd.DataFrame(), None

    # Step 1：查索引文档
    term_info = term_dict[term_dict["term"] == term_norm]

    if len(term_info) == 0:
        return pd.DataFrame(), None

    term_info_row = term_info.iloc[0]
    term_id = term_info_row["term_id"]

    # Step 2：查倒排档
    postings = inverted_df[inverted_df["term_id"] == term_id].copy()

    if len(postings) == 0:
        return pd.DataFrame(), term_info_row

    # Step 3：字段加权
    # 让古埃及语核心字段优先于普通译文字段
    field_weights = {
        "lemma_forms": 5,
        "normalized_transliteration": 4,
        "mdc": 3,
        "translation": 2
    }

    postings["field_weight"] = postings["field"].map(field_weights).fillna(1)
    postings["weighted_tf"] = postings["tf"] * postings["field_weight"]

    # Step 4：按 doc_id 聚合
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

    doc_scores["matched_term"] = term_norm

    return doc_scores, term_info_row


# =========================
# 7. 中文/普通统一检索
# =========================
def search(query, top_k=10):
    """
    支持中文主题词、英文关键词、古埃及转写词检索。
    中文检索：先扩展词，再分别查倒排档，最后合并排序。
    普通检索：直接查倒排档。
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
            doc_scores, term_info = dialog_search_single_term(term)

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

        combined_grouped = (
            combined
            .groupby("doc_id")
            .agg(
                total_tf=("total_tf", "sum"),
                weighted_score=("weighted_score", "sum"),
                matched_terms=("matched_term", lambda x: ", ".join(sorted(set(x)))),
                matched_fields=("matched_fields", lambda x: ", ".join(sorted(set(", ".join(x).split(", "))))),
                positions=("positions", lambda x: " || ".join(map(str, x)))
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

        # 回主文档，取完整文本信息
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
        term_norm = normalize_query_term(query)
        doc_scores, term_info = dialog_search_single_term(term_norm)

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
            by=["weighted_score", "total_tf", "doc_id"],
            ascending=[False, False, True]
        ).head(top_k)

        # 回主文档，取完整文本信息
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
# 8. 页面主体
# =========================
st.title("𓂀 古埃及文字检索智能体")
st.caption(
    "基于 DIALOG 风格的“主文档—索引文档—倒排档”结构，"
    "支持中文主题扩展、古埃及转写检索与原始语料证据返回。"
)

with st.sidebar:
    st.header("系统信息")
    st.write("主文档数量：", len(main_df))
    st.write("索引词条数量：", len(term_dict))
    st.write("倒排记录数量：", len(inverted_df))
    st.write("中文主题数量：", len(query_expansion_df))

    st.divider()

    st.header("示例查询")
    st.markdown("""
    **中文：**
    - 神
    - 奥西里斯
    - 国王
    - 太阳神
    - 供奉
    - 来世

    **转写/英文：**
    - ntr
    - wsjr
    - nswt
    - osiris
    - king
    """)


query = st.text_input(
    "请输入检索词",
    placeholder="例如：神、奥西里斯、国王、ntr、wsjr、osiris"
)

top_k = st.slider("返回结果数量", min_value=3, max_value=20, value=10)

search_button = st.button("开始检索", type="primary")


if search_button:
    if not query.strip():
        st.warning("请输入检索词。")
    else:
        output = search(query, top_k=top_k)

        st.subheader("检索概览")

        col1, col2, col3 = st.columns(3)
        col1.metric("检索模式", output["mode"])
        col2.metric("扩展词数量", len(output["expanded_terms"]))
        col3.metric("返回结果数", len(output["results"]))

        st.write("**原始查询：**", output["query"])
        st.write(
            "**扩展词：**",
            ", ".join(output["expanded_terms"]) if output["expanded_terms"] else "无"
        )

        if output["explanation"]:
            st.info(output["explanation"])

        if output["term_infos"]:
            with st.expander("查看命中的索引词信息"):
                for info in output["term_infos"]:
                    if info is None:
                        continue
                    st.markdown(
                        f"""
                        - **term**: `{info['term']}`
                        - **term_id**: `{info['term_id']}`
                        - **df**: {info['df']}
                        - **total_tf**: {info['total_tf']}
                        - **fields**: {info['fields']}
                        """
                    )

        results = output["results"]

        if len(results) == 0:
            st.error("没有找到结果。")
        else:
            st.subheader("检索结果")

            for idx, row in results.iterrows():
                with st.container(border=True):
                    st.markdown(f"### 结果 {idx + 1}｜{row['doc_id']}")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.write("**corpus**")
                    c1.write(row["corpus"])

                    c2.write("**date**")
                    c2.write(row["date"])

                    c3.write("**findspot**")
                    c3.write(row["findspot"])

                    c4.write("**加权分数**")
                    c4.write(row.get("weighted_score", "未计算"))

                    matched_terms = row.get("matched_terms", row.get("matched_term", ""))
                    matched_fields = row.get("matched_fields", "")

                    st.write("**命中词：**", matched_terms)
                    st.write("**匹配字段：**", matched_fields)
                    st.write("**原始词频 total_tf：**", row["total_tf"])

                    chinese_hint = generate_chinese_hint(
                        query=output["query"],
                        matched_terms=matched_terms,
                        matched_fields=matched_fields
                    )

                    st.info(chinese_hint)

                    st.markdown("**原始译文：**")
                    st.write(row["translation"])

                    st.markdown("**古埃及转写：**")
                    st.code(row["transliteration"], language="text")

                    with st.expander("查看 lemma / mdc / 归一化转写"):
                        st.markdown("**归一化转写：**")
                        st.code(row["normalized_transliteration"], language="text")

                        st.markdown("**lemma_forms：**")
                        st.code(row["lemma_forms"], language="text")

                        st.markdown("**mdc：**")
                        st.code(row["mdc"], language="text")
