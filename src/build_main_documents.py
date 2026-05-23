from pathlib import Path
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
INPUT_CSV = PROJECT_DIR / "data_processed" / "sentences.csv"
OUTPUT_CSV = PROJECT_DIR / "data_processed" / "main_documents.csv"


# =========================
# 2. 读取 sentences.csv
# =========================
df = pd.read_csv(INPUT_CSV)

print("原始 sentences.csv 读取成功")
print("原始数据行数：", len(df))
print("原始字段数：", len(df.columns))


# =========================
# 3. 选择主文档需要的字段
# =========================
selected_columns = [
    "sentence_id",
    "text_id",
    "corpus",
    "date",
    "findspot",
    "sentence_translation",
    "transliteration",
    "normalized_transliteration",
    "mdc",
    "lemma_forms",
    "pos_tags",
    "hiero_inventar",
    "source_file",
    "token_count"
]

main_df = df[selected_columns].copy()


# =========================
# 4. 字段重命名
# =========================
main_df = main_df.rename(columns={
    "sentence_translation": "translation"
})


# =========================
# 5. 生成 doc_id
# =========================
main_df.insert(
    0,
    "doc_id",
    [f"D{i:06d}" for i in range(1, len(main_df) + 1)]
)


# =========================
# 6. 基础清洗
# =========================
# 填充空值，避免后面建索引时报错
text_columns = [
    "sentence_id",
    "text_id",
    "corpus",
    "date",
    "findspot",
    "translation",
    "transliteration",
    "normalized_transliteration",
    "mdc",
    "lemma_forms",
    "pos_tags",
    "hiero_inventar",
    "source_file"
]

for col in text_columns:
    main_df[col] = main_df[col].fillna("").astype(str)

main_df["token_count"] = main_df["token_count"].fillna(0).astype(int)


# =========================
# 7. 删除完全无效文档
# =========================
# 如果一条记录既没有译文，也没有转写，就没有检索价值
before_count = len(main_df)

main_df = main_df[
    (main_df["translation"].str.strip() != "") |
    (main_df["transliteration"].str.strip() != "") |
    (main_df["lemma_forms"].str.strip() != "")
].copy()

after_count = len(main_df)

print("删除无效文档数：", before_count - after_count)
print("主文档记录数：", after_count)


# =========================
# 8. 保存主文档
# =========================
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
main_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\n主文档已生成：", OUTPUT_CSV)


# =========================
# 9. 输出预览
# =========================
print("\n主文档字段：")
print(main_df.columns.tolist())

print("\n前 5 条主文档预览：")
print(main_df.head())

print("\ncorpus 数量前 10：")
print(main_df["corpus"].value_counts().head(10))