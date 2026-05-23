from pathlib import Path
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
CSV_PATH = PROJECT_DIR / "data_processed" / "sentences.csv"


# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(CSV_PATH)

print("数据读取成功！")
print("总句子数：", len(df))
print("可检索字段：sentence_translation / transliteration / normalized_transliteration / lemma_forms / mdc")


# =========================
# 3. 检索函数
# =========================
def keyword_search(query, top_k=10):
    """
    在多个字段中进行关键词检索。
    """

    q = query.lower().strip()

    if not q:
        return pd.DataFrame()

    search_fields = [
        "sentence_translation",
        "transliteration",
        "normalized_transliteration",
        "lemma_forms",
        "mdc"
    ]

    mask = False

    for field in search_fields:
        field_mask = df[field].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        mask = mask | field_mask

    results = df[mask].copy()

    # 简单排序：优先展示 token_count 不太长的句子，方便阅读
    results = results.sort_values(by="token_count", ascending=True)

    return results.head(top_k)


# =========================
# 4. 交互式检索
# =========================
while True:
    query = input("\n请输入检索词，输入 q 退出：").strip()

    if query.lower() == "q":
        print("已退出检索。")
        break

    results = keyword_search(query, top_k=10)

    print(f"\n检索词：{query}")
    print(f"返回结果数：{len(results)}")

    if len(results) == 0:
        print("没有找到结果。")
        continue

    for i, row in results.iterrows():
        print("\n" + "=" * 80)
        print("sentence_id:", row["sentence_id"])
        print("corpus:", row["corpus"])
        print("date:", row["date"])
        print("findspot:", row["findspot"])
        print("translation:", row["sentence_translation"])
        print("transliteration:", row["transliteration"])
        print("normalized:", row["normalized_transliteration"])
        print("lemma_forms:", row["lemma_forms"])
        print("mdc:", row["mdc"])