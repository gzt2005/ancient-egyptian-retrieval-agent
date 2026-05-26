from pathlib import Path
import re
import time
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# =========================
# 1. 页面设置
# =========================
st.set_page_config(
    page_title="古埃及文字智能检索系统",
    page_icon="𓂀",
    layout="wide"
)


# =========================
# 2. 路径设置
# =========================
PROJECT_DIR = Path(__file__).parent

DB_PATH = PROJECT_DIR / "database_demo" / "egypt_demo.db"

SEMANTIC_DIR = PROJECT_DIR / "data_semantic_demo"
SEMANTIC_EMBEDDINGS_PATH = SEMANTIC_DIR / "semantic_embeddings.npy"
SEMANTIC_METADATA_PATH = SEMANTIC_DIR / "semantic_metadata.csv"
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EVALUATION_DIR = PROJECT_DIR / "evaluation_results"
EVALUATION_RESULTS_CSV = EVALUATION_DIR / "evaluation_results.csv"


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


def get_sqlite_connection():
    """
    获取 SQLite 数据库连接。
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"数据库文件不存在：{DB_PATH}")
    return sqlite3.connect(DB_PATH)


# =========================
# 4. SQLite 系统信息加载
# =========================
@st.cache_data(show_spinner="正在读取 SQLite 数据库信息...")
def load_sqlite_counts():
    """
    从 SQLite 数据库读取各表记录数量。
    """
    if not DB_PATH.exists():
        return {
            "main_documents": 0,
            "term_dictionary": 0,
            "inverted_file": 0,
            "query_expansion": 0
        }

    conn = sqlite3.connect(DB_PATH)

    try:
        counts = {}
        for table in ["main_documents", "term_dictionary", "inverted_file", "query_expansion"]:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table};")
            counts[table] = cursor.fetchone()[0]
        return counts
    finally:
        conn.close()


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
# 4.2 缓存加载系统测评结果
# =========================
@st.cache_data(show_spinner="正在加载系统性能测评结果...")
def load_evaluation_results():
    if not EVALUATION_RESULTS_CSV.exists():
        return pd.DataFrame()

    eval_df = pd.read_csv(EVALUATION_RESULTS_CSV, dtype=str).fillna("")

    if "elapsed_time_sec" in eval_df.columns:
        eval_df["elapsed_time_sec"] = pd.to_numeric(
            eval_df["elapsed_time_sec"],
            errors="coerce"
        ).fillna(0)

    if "result_count" in eval_df.columns:
        eval_df["result_count"] = pd.to_numeric(
            eval_df["result_count"],
            errors="coerce"
        ).fillna(0).astype(int)

    return eval_df


# =========================
# 5. 中文查询扩展：SQLite 版
# =========================
def expand_chinese_query_sqlite(conn, query: str):
    """
    从 SQLite 的 query_expansion 表中读取中文查询扩展词。
    """
    query = query.strip()

    exact_hit = pd.read_sql_query(
        """
        SELECT *
        FROM query_expansion
        WHERE query_zh = ?
        LIMIT 1;
        """,
        conn,
        params=(query,)
    )

    if len(exact_hit) > 0:
        row = exact_hit.iloc[0]
    else:
        all_expansion = pd.read_sql_query(
            """
            SELECT *
            FROM query_expansion;
            """,
            conn
        ).fillna("")

        fuzzy_hit = all_expansion[
            all_expansion["query_zh"].apply(lambda x: x in query or query in x)
        ]

        if len(fuzzy_hit) == 0:
            return [], "未在中文查询扩展表中找到该主题。"

        row = fuzzy_hit.iloc[0]

    expanded_terms = [
        normalize_query_term(t)
        for t in str(row.get("expanded_terms", "")).split(",")
        if normalize_query_term(t)
    ]

    explanation = row.get("explanation_zh", "")

    return expanded_terms, explanation


# =========================
# 6. 单词 DIALOG 检索：SQLite 版
# =========================
def dialog_search_single_term_sqlite(conn, term):
    """
    对单个 term 执行 SQLite-backed DIALOG 风格检索，并加入字段加权排序。
    """
    term_norm = normalize_query_term(term)

    if not term_norm:
        return pd.DataFrame(), None

    # 1. 查索引文档 term_dictionary
    term_info = pd.read_sql_query(
        """
        SELECT term_id, term, df, total_tf, fields
        FROM term_dictionary
        WHERE term = ?
        LIMIT 1;
        """,
        conn,
        params=(term_norm,)
    )

    if len(term_info) == 0:
        return pd.DataFrame(), None

    term_info_row = term_info.iloc[0]
    term_id = term_info_row["term_id"]

    # 2. 查倒排档 inverted_file
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
        return pd.DataFrame(), term_info_row

    postings["tf"] = pd.to_numeric(postings["tf"], errors="coerce").fillna(0).astype(int)

    # 3. 字段加权
    field_weights = {
        "lemma_forms": 5,
        "normalized_transliteration": 4,
        "mdc": 3,
        "translation": 2
    }

    postings["field_weight"] = postings["field"].map(field_weights).fillna(1)
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

    doc_scores["matched_term"] = term_norm

    return doc_scores, term_info_row


def fetch_main_documents_sqlite(conn, doc_scores):
    """
    根据 doc_scores 中的 doc_id 回主文档表取完整文本信息。
    """
    if len(doc_scores) == 0:
        return pd.DataFrame()

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
    ).fillna("")

    results = doc_scores.merge(main_docs, on="doc_id", how="left")

    return results


# =========================
# 7. 关键词检索：SQLite 版
# =========================
def keyword_search(query, top_k=10):
    """
    SQLite-backed 关键词检索。
    支持中文主题词、英文关键词、古埃及转写词检索。
    """
    query = query.strip()

    conn = get_sqlite_connection()

    try:
        if contains_chinese(query):
            expanded_terms, explanation = expand_chinese_query_sqlite(conn, query)

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
                doc_scores, term_info = dialog_search_single_term_sqlite(conn, term)

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

            results = fetch_main_documents_sqlite(conn, combined_grouped)

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

            doc_scores, term_info = dialog_search_single_term_sqlite(conn, term_norm)

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

            results = fetch_main_documents_sqlite(conn, doc_scores)

            return {
                "query": query,
                "mode": "关键词检索",
                "sub_mode": "普通检索",
                "expanded_terms": [term_norm],
                "explanation": "",
                "term_infos": [term_info] if term_info is not None else [],
                "results": results
            }

    finally:
        conn.close()


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
st.title("𓂀 古埃及文字智能检索系统")
st.caption(
    "Version 1.0｜基于 DIALOG 风格的“主文档—索引文档—倒排档”结构，"
    "支持 SQLite 关键词检索、中文主题扩展、字段加权排序、AI 语义检索与系统性能测评。"
)

system_counts = load_sqlite_counts()

with st.sidebar:
    st.header("系统信息")
    st.write("系统版本：", "Version 1.0")
    st.write("数据库状态：", "已连接" if DB_PATH.exists() else "未找到")
    st.write("主文档数量：", system_counts.get("main_documents", 0))
    st.write("索引词条数量：", system_counts.get("term_dictionary", 0))
    st.write("倒排记录数量：", system_counts.get("inverted_file", 0))
    st.write("中文主题数量：", system_counts.get("query_expansion", 0))

    if SEMANTIC_EMBEDDINGS_PATH.exists() and SEMANTIC_METADATA_PATH.exists():
        st.write("语义索引状态：", "已加载")
        st.write("语义索引规模：", "8000 条")
    else:
        st.write("语义索引状态：", "未找到")

    if EVALUATION_RESULTS_CSV.exists():
        st.write("性能测评状态：", "已生成")
    else:
        st.write("性能测评状态：", "未生成")

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
        start_time = time.perf_counter()

        if search_mode == "关键词检索":
            output = keyword_search(query, top_k=top_k)
        else:
            output = semantic_search(query, top_k=top_k)

        elapsed_time = time.perf_counter() - start_time

        st.subheader("检索概览")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("检索模式", output["mode"])
        col2.metric("子模式", output.get("sub_mode", ""))
        col3.metric("返回结果数", len(output["results"]))
        col4.metric("检索耗时", f"{elapsed_time:.3f} 秒")

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


# =========================
# 10. 系统性能测评展示
# =========================
st.divider()

with st.expander("系统性能测评", expanded=False):
    st.markdown(
        """
        本模块用于展示系统批量性能测评结果，主要比较关键词检索与 AI 语义检索在响应时间、
        返回结果数量和 Top-1 文档等方面的表现。
        """
    )

    eval_df = load_evaluation_results()

    if len(eval_df) == 0:
        st.warning(
            "尚未找到性能测评结果文件。请先运行 "
            "`src/evaluate_search_performance.py` 生成 evaluation_results.csv。"
        )
    else:
        st.subheader("测评数据概览")

        c1, c2, c3 = st.columns(3)
        c1.metric("测试查询数量", len(eval_df))
        c2.metric("检索模式数量", eval_df["mode"].nunique())
        c3.metric("平均返回结果数", round(eval_df["result_count"].mean(), 2))

        st.subheader("不同检索模式平均耗时")

        avg_time_df = (
            eval_df
            .groupby("mode", as_index=False)["elapsed_time_sec"]
            .mean()
            .rename(columns={"elapsed_time_sec": "avg_elapsed_time_sec"})
        )

        avg_time_df["avg_elapsed_time_sec"] = avg_time_df["avg_elapsed_time_sec"].round(4)

        st.dataframe(avg_time_df, use_container_width=True)

        st.subheader("详细测评结果")

        display_cols = [
            "mode",
            "query",
            "top_k",
            "elapsed_time_sec",
            "result_count",
            "top_doc_id",
            "top_score",
            "top_corpus",
            "top_translation_preview"
        ]

        display_cols = [c for c in display_cols if c in eval_df.columns]

        st.dataframe(
            eval_df[display_cols],
            use_container_width=True
        )

        st.info(
            "说明：AI 语义检索的批量测评结果为模型预热后的热启动检索耗时；"
            "网页端首次语义检索可能包含模型加载和语义索引加载时间，因此首次耗时会更长。"
        )