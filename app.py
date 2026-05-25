from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


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

SEMANTIC_DIR = PROJECT_DIR / "data_semantic_demo"
SEMANTIC_EMBEDDINGS_PATH = SEMANTIC_DIR / "semantic_embeddings.npy"
SEMANTIC_METADATA_PATH = SEMANTIC_DIR / "semantic_metadata.csv"

SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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
# 4. 缓存加载倒排检索数据
# =========================
@st.cache_data(show_spinner="正在加载主文档、索引文档和倒排档，请稍等...")
def load_data():
    main_df = pd.read_csv(MAIN_DOCS_CSV, dtype=str).fillna("")
    term_dict = pd.read_csv(TERM_DICTIONARY_CSV, dtype=str).fillna("")
    inverted_df = pd.read_csv(INVERTED_FILE_CSV, dtype=str, low_memory=False).fillna("")
    query_expansion_df = pd.read_csv(QUERY_EXPANSION_CSV, dtype=str).fillna("")

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
# 4.1 缓存加载 AI 语义检索资源
# =========================
@st.cache_resource(show_spinner="正在加载 AI 语义检索模型，请稍等...")
def load_semantic_model():
    return SentenceTransformer(SEMANTIC_MODEL_NAME)


@st.cache_data(show_spinner="正在加载语义向量索引，请稍等...")
def load_semantic_index():
    embeddings = np.load(SEMANTIC_EMBEDDINGS_PATH)
    metadata = pd.read_csv(SEMANTIC_METADATA_PATH, dtype=str).fillna("")
    return embeddings, metadata


# =========================
# 5. 中文查询扩展
# =========================
def expand_chinese_query(query: str):
    """
    如果用户输入中文，则查询 query_expansion.csv，
    将中文主题词扩展为英文、德文和古埃及转写词。
    """
    query = query.strip()

    hit = query_expansion_df[query_expansion_df["query_zh"] == query]

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
            positions=("positions", lambda x: " | ".join(map(str, x)))
        )
        .reset_index()
    )

    doc_scores["matched_term"] = term_norm

    return doc_scores, term_info_row


# =========================
# 7. 关键词检索：中文/普通统一检索
# =========================
def keyword_search(query, top_k=10):
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
                "mode": "关键词检索",
                "sub_mode": "中文检索",
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
                "mode": "关键词检索",
                "sub_mode": "中文检索",
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

        results = combined_grouped.merge(main_df, on="doc_id", how="left")

        return {
            "query": query,
            "mode": "关键词检索",
            "sub_mode": "中文检索",
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
                "mode": "关键词检索",
                "sub_mode": "普通检索",
                "expanded_terms": [term_norm],
                "explanation": "",
                "term_infos": [term_info] if term_info is not None else [],
                "results": pd.DataFrame()
            }

        doc_scores = doc_scores.sort_values(
            by=["weighted_score", "total_tf", "doc_id"],
            ascending=[False, False, True]
        ).head(top_k)

        results = doc_scores.merge(main_df, on="doc_id", how="left")

        return {
            "query": query,
            "mode": "关键词检索",
            "sub_mode": "普通检索",
            "expanded_terms": [term_norm],
            "explanation": "",
            "term_infos": [term_info] if term_info is not None else [],
            "results": results
        }


# =========================
# 8. AI 语义检索
# =========================
def semantic_search(query, top_k=10):
    """
    基于 sentence-transformers 的 AI 语义检索。
    """
    query = query.strip()

    if not query:
        return {
            "query": query,
            "mode": "AI语义检索",
            "sub_mode": "语义向量检索",
            "expanded_terms": [],
            "explanation": "请输入有效查询。",
            "term_infos": [],
            "results": pd.DataFrame()
        }

    model = load_semantic_model()
    embeddings, metadata = load_semantic_index()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(embeddings, query_embedding[0])
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()
    results["semantic_score"] = scores[top_indices]

    return {
        "query": query,
        "mode": "AI语义检索",
        "sub_mode": "语义向量检索",
        "expanded_terms": [],
        "explanation": "系统基于语义向量计算用户查询与古埃及文本记录之间的相似度，返回语义最接近的原始语料证据。",
        "term_infos": [],
        "results": results
    }


# =========================
# 9. 页面主体
# =========================
st.title("𓂀 古埃及文字检索智能体")
st.caption(
    "基于 DIALOG 风格的“主文档—索引文档—倒排档”结构，"
    "支持中文主题扩展、古埃及转写检索、字段加权排序与 AI 语义检索。"
)

with st.sidebar:
    st.header("系统信息")
    st.write("主文档数量：", len(main_df))
    st.write("索引词条数量：", len(term_dict))
    st.write("倒排记录数量：", len(inverted_df))
    st.write("中文主题数量：", len(query_expansion_df))

    if SEMANTIC_EMBEDDINGS_PATH.exists() and SEMANTIC_METADATA_PATH.exists():
        try:
            semantic_metadata_preview = pd.read_csv(SEMANTIC_METADATA_PATH, dtype=str, nrows=5)
            st.write("语义索引状态：", "已加载")
            st.write("语义索引规模：", "8000 条")
        except Exception:
            st.write("语义索引状态：", "存在但未检查")
    else:
        st.write("语义索引状态：", "未找到")

    st.divider()

    st.header("示例查询")
    st.markdown("""
    **关键词检索：**
    - 神
    - 奥西里斯
    - 国王
    - ntr
    - wsjr
    - osiris

    **AI语义检索：**
    - 太阳神和国王
    - Osiris and afterlife
    - offering rituals
    - texts about gods and kingship
    - enemies of Osiris
    """)


search_mode = st.radio(
    "请选择检索模式",
    ["关键词检索", "AI语义检索"],
    horizontal=True
)

query = st.text_input(
    "请输入检索词或自然语言问题",
    placeholder="例如：神、奥西里斯、ntr、太阳神和国王、Osiris and afterlife"
)

top_k = st.slider("返回结果数量", min_value=3, max_value=20, value=10)

search_button = st.button("开始检索", type="primary")


if search_button:
    if not query.strip():
        st.warning("请输入检索词或自然语言问题。")
    else:
        if search_mode == "关键词检索":
            output = keyword_search(query, top_k=top_k)
        else:
            output = semantic_search(query, top_k=top_k)

        st.subheader("检索概览")

        col1, col2, col3 = st.columns(3)
        col1.metric("检索模式", output["mode"])
        col2.metric("子模式", output.get("sub_mode", ""))
        col3.metric("返回结果数", len(output["results"]))

        st.write("**原始查询：**", output["query"])

        if output["expanded_terms"]:
            st.write("**扩展词：**", ", ".join(output["expanded_terms"]))

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

            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                with st.container(border=True):
                    st.markdown(f"### 结果 {rank}｜{row.get('doc_id', '')}")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.write("**corpus**")
                    c1.write(row.get("corpus", ""))

                    c2.write("**date**")
                    c2.write(row.get("date", ""))

                    c3.write("**findspot**")
                    c3.write(row.get("findspot", ""))

                    if output["mode"] == "AI语义检索":
                        c4.write("**语义分数**")
                        c4.write(round(float(row.get("semantic_score", 0)), 4))
                    else:
                        c4.write("**加权分数**")
                        c4.write(row.get("weighted_score", "未计算"))

                    if output["mode"] == "AI语义检索":
                        st.write("**检索方式：** AI 语义相似度匹配")
                        st.write(
                            "**语义分数 semantic_score：**",
                            round(float(row.get("semantic_score", 0)), 4)
                        )

                        st.info(
                            f"该结果与查询“{output['query']}”在语义上较为接近。"
                            "系统根据文本译文、古埃及转写和元数据生成语义向量，并按相似度返回相关古埃及文本证据。"
                        )
                    else:
                        matched_terms = row.get("matched_terms", row.get("matched_term", ""))
                        matched_fields = row.get("matched_fields", "")

                        st.write("**命中词：**", matched_terms)
                        st.write("**匹配字段：**", matched_fields)
                        st.write("**原始词频 total_tf：**", row.get("total_tf", ""))

                        chinese_hint = generate_chinese_hint(
                            query=output["query"],
                            matched_terms=matched_terms,
                            matched_fields=matched_fields
                        )

                        st.info(chinese_hint)

                    st.markdown("**原始译文：**")
                    st.write(row.get("translation", ""))

                    st.markdown("**古埃及转写：**")
                    st.code(row.get("transliteration", ""), language="text")

                    with st.expander("查看 lemma / mdc / 归一化转写"):
                        st.markdown("**归一化转写：**")
                        st.code(row.get("normalized_transliteration", ""), language="text")

                        st.markdown("**lemma_forms：**")
                        st.code(row.get("lemma_forms", ""), language="text")

                        st.markdown("**mdc：**")
                        st.code(row.get("mdc", ""), language="text")