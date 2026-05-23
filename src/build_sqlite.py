from pathlib import Path
import pandas as pd
import sqlite3


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
CSV_PATH = PROJECT_DIR / "data_processed" / "sentences.csv"
DB_PATH = PROJECT_DIR / "database" / "egypt_agent.db"


# =========================
# 2. 读取 CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("CSV 读取成功")
print("数据行数：", len(df))
print("字段数量：", len(df.columns))


# =========================
# 3. 写入 SQLite
# =========================
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

df.to_sql("sentences", conn, if_exists="replace", index=False)

cursor = conn.cursor()

# =========================
# 4. 建立索引，提高查询速度
# =========================
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentence_id ON sentences(sentence_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_id ON sentences(text_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_corpus ON sentences(corpus);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON sentences(date);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_findspot ON sentences(findspot);")

conn.commit()


# =========================
# 5. 检查数据库写入结果
# =========================
cursor.execute("SELECT COUNT(*) FROM sentences;")
count = cursor.fetchone()[0]

cursor.execute("SELECT corpus, COUNT(*) AS cnt FROM sentences GROUP BY corpus ORDER BY cnt DESC LIMIT 10;")
corpus_counts = cursor.fetchall()

conn.close()

print("\n数据库已生成：", DB_PATH)
print("sentences 表记录数：", count)

print("\n前 10 个 corpus 数量：")
for corpus, cnt in corpus_counts:
    print(corpus, ":", cnt)
    