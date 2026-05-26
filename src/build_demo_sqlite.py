from pathlib import Path
import sqlite3
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

DATA_DEMO_DIR = PROJECT_DIR / "data_demo"
DATABASE_DEMO_DIR = PROJECT_DIR / "database_demo"
DATABASE_DEMO_DIR.mkdir(parents=True, exist_ok=True)

MAIN_DOCS_CSV = DATA_DEMO_DIR / "main_documents.csv"
TERM_DICTIONARY_CSV = DATA_DEMO_DIR / "term_dictionary.csv"
INVERTED_FILE_CSV = DATA_DEMO_DIR / "inverted_file.csv"
QUERY_EXPANSION_CSV = DATA_DEMO_DIR / "query_expansion.csv"

DB_PATH = DATABASE_DEMO_DIR / "egypt_demo.db"


# =========================
# 2. 工具函数
# =========================
def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    """
    安全读取 CSV 文件，并统一把空值填充为空字符串。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在：{csv_path}")

    print(f"正在读取：{csv_path}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False).fillna("")
    print(f"读取完成：{csv_path.name}，行数：{len(df)}，字段数：{len(df.columns)}")
    return df


def create_indexes(conn: sqlite3.Connection):
    """
    为常用检索字段建立索引。
    """
    cursor = conn.cursor()

    print("\n正在创建索引...")

    index_sql_list = [
        # 主文档表索引
        "CREATE INDEX IF NOT EXISTS idx_main_documents_doc_id ON main_documents(doc_id);",
        "CREATE INDEX IF NOT EXISTS idx_main_documents_corpus ON main_documents(corpus);",
        "CREATE INDEX IF NOT EXISTS idx_main_documents_date ON main_documents(date);",

        # 索引文档表索引
        "CREATE INDEX IF NOT EXISTS idx_term_dictionary_term ON term_dictionary(term);",
        "CREATE INDEX IF NOT EXISTS idx_term_dictionary_term_id ON term_dictionary(term_id);",

        # 倒排档索引
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_term_id ON inverted_file(term_id);",
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_term ON inverted_file(term);",
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_doc_id ON inverted_file(doc_id);",
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_term_doc ON inverted_file(term_id, doc_id);",

        # 中文扩展表索引
        "CREATE INDEX IF NOT EXISTS idx_query_expansion_query_zh ON query_expansion(query_zh);"
    ]

    for sql in index_sql_list:
        cursor.execute(sql)

    conn.commit()
    print("索引创建完成。")


def check_database(conn: sqlite3.Connection):
    """
    检查数据库表数量和核心记录数量。
    """
    cursor = conn.cursor()

    print("\n==============================")
    print("数据库检查")
    print("==============================")

    tables = [
        "main_documents",
        "term_dictionary",
        "inverted_file",
        "query_expansion"
    ]

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} 条记录")

    print("\n检查重要检索词：")
    important_terms = ["ntr", "wsjr", "nswt", "htp", "king", "god", "osiris"]

    for term in important_terms:
        cursor.execute(
            """
            SELECT term_id, term, df, total_tf, fields
            FROM term_dictionary
            WHERE term = ?
            LIMIT 1;
            """,
            (term,)
        )
        row = cursor.fetchone()

        if row is None:
            print(f"{term}: 未找到")
        else:
            print(f"{term}: term_id={row[0]}, df={row[2]}, total_tf={row[3]}, fields={row[4]}")

    print("\n检查倒排记录示例：")
    cursor.execute(
        """
        SELECT term_id, term, doc_id, field, tf, positions
        FROM inverted_file
        WHERE term = 'wsjr'
        LIMIT 5;
        """
    )
    rows = cursor.fetchall()

    for row in rows:
        print(row)


# =========================
# 3. 主流程
# =========================
def main():
    print("==============================")
    print("开始构建展示版 SQLite 数据库")
    print("==============================")

    print("数据库路径：", DB_PATH)

    main_df = read_csv_safely(MAIN_DOCS_CSV)
    term_df = read_csv_safely(TERM_DICTIONARY_CSV)
    inverted_df = read_csv_safely(INVERTED_FILE_CSV)
    query_expansion_df = read_csv_safely(QUERY_EXPANSION_CSV)

    # 数值字段尽量转换，方便后续排序和统计
    for col in ["token_count"]:
        if col in main_df.columns:
            main_df[col] = pd.to_numeric(main_df[col], errors="coerce").fillna(0).astype(int)

    for col in ["df", "total_tf", "field_count"]:
        if col in term_df.columns:
            term_df[col] = pd.to_numeric(term_df[col], errors="coerce").fillna(0).astype(int)

    for col in ["tf"]:
        if col in inverted_df.columns:
            inverted_df[col] = pd.to_numeric(inverted_df[col], errors="coerce").fillna(0).astype(int)

    # 如果旧数据库存在，先删除，避免旧表残留
    if DB_PATH.exists():
        print("\n发现旧数据库，正在删除：", DB_PATH)
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    try:
        print("\n正在写入 SQLite 表...")

        main_df.to_sql(
            "main_documents",
            conn,
            if_exists="replace",
            index=False
        )
        print("main_documents 写入完成。")

        term_df.to_sql(
            "term_dictionary",
            conn,
            if_exists="replace",
            index=False
        )
        print("term_dictionary 写入完成。")

        inverted_df.to_sql(
            "inverted_file",
            conn,
            if_exists="replace",
            index=False
        )
        print("inverted_file 写入完成。")

        query_expansion_df.to_sql(
            "query_expansion",
            conn,
            if_exists="replace",
            index=False
        )
        print("query_expansion 写入完成。")

        create_indexes(conn)
        check_database(conn)

        print("\n==============================")
        print("展示版 SQLite 数据库构建完成！")
        print("==============================")
        print("数据库文件：", DB_PATH)

    finally:
        conn.close()


if __name__ == "__main__":
    main()