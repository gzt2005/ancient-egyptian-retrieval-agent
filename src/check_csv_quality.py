from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
CSV_PATH = PROJECT_DIR / "data_processed" / "sentences.csv"

df = pd.read_csv(CSV_PATH)

print("CSV 是否成功读取：是")
print("数据总行数：", len(df))
print("字段数量：", len(df.columns))

print("\n字段列表：")
for col in df.columns:
    print("-", col)

print("\n==============================")
print("关键字段缺失情况")
print("==============================")

key_cols = [
    "sentence_translation",
    "transliteration",
    "normalized_transliteration",
    "lemma_forms",
    "mdc",
    "hiero_inventar",
    "hiero_unicode",
    "hiero"
]

for col in key_cols:
    empty_count = df[col].fillna("").astype(str).str.strip().eq("").sum()
    empty_rate = empty_count / len(df) * 100
    print(f"{col}: 空值 {empty_count} 条，占比 {empty_rate:.2f}%")

print("\n==============================")
print("随机抽取 5 条非空样本")
print("==============================")

sample_df = df[
    (df["sentence_translation"].fillna("").astype(str).str.len() > 5) &
    (df["transliteration"].fillna("").astype(str).str.len() > 3)
].sample(5, random_state=42)

for idx, row in sample_df.iterrows():
    print("\n" + "=" * 80)
    print("sentence_id:", row["sentence_id"])
    print("corpus:", row["corpus"])
    print("date:", row["date"])
    print("findspot:", row["findspot"])
    print("translation:", row["sentence_translation"])
    print("transliteration:", row["transliteration"])
    print("normalized:", row["normalized_transliteration"])
    print("lemma_forms:", row["lemma_forms"])
    print("hiero_inventar:", row["hiero_inventar"])
    print("hiero:", row["hiero"])