from pathlib import Path
import sqlite3


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
DB_PATH = PROJECT_DIR / "database" / "egypt_agent.db"


# =========================
# 2. 数据库检索函数
# =========================
def search_sentences(query, limit=10):
    """
    在 SQLite 数据库中检索古埃及句子。
    支持 translation / transliteration / normalized_transliteration / lemma_forms / mdc。
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    q = f"%{query.lower().strip()}%"

    sql = """
    SELECT 
        sentence_id,
        text_id,
        corpus,
        date,
        findspot,
        sentence_translation,
        transliteration,
        normalized_transliteration,
        lemma_forms,
        mdc,
        token_count
    FROM sentences
    WHERE lower(sentence_translation) LIKE ?
       OR lower(transliteration) LIKE ?
       OR lower(normalized_transliteration) LIKE ?
       OR lower(lemma_forms) LIKE ?
       OR lower(mdc) LIKE ?
    ORDER BY token_count ASC
    LIMIT ?;
    """

    cursor.execute(sql, (q, q, q, q, q, limit))
    rows = cursor.fetchall()

    conn.close()

    return rows


# =========================
# 3. 交互式检索
# =========================
def main():
    print("SQLite 数据库检索系统已启动")
    print("数据库路径：", DB_PATH)
    print("可输入英文关键词、古埃及转写归一化词、lemma 或 mdc")
    print("例如：king / god / ntr / dd / htp / Wsjr")
    print("输入 q 退出")

    while True:
        query = input("\n请输入检索词：").strip()

        if query.lower() == "q":
            print("已退出。")
            break

        if not query:
            print("请输入有效检索词。")
            continue

        results = search_sentences(query, limit=10)

        print(f"\n检索词：{query}")
        print(f"返回结果数：{len(results)}")

        if len(results) == 0:
            print("没有找到结果。")
            continue

        for row in results:
            print("\n" + "=" * 80)
            print("sentence_id:", row["sentence_id"])
            print("text_id:", row["text_id"])
            print("corpus:", row["corpus"])
            print("date:", row["date"])
            print("findspot:", row["findspot"])
            print("token_count:", row["token_count"])
            print("translation:", row["sentence_translation"])
            print("transliteration:", row["transliteration"])
            print("normalized:", row["normalized_transliteration"])
            print("lemma_forms:", row["lemma_forms"])
            print("mdc:", row["mdc"])


if __name__ == "__main__":
    main()