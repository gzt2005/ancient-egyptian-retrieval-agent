from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SEMANTIC_DIR = PROJECT_DIR / "data_semantic_demo"
EMBEDDINGS_PATH = SEMANTIC_DIR / "semantic_embeddings.npy"
METADATA_PATH = SEMANTIC_DIR / "semantic_metadata.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================
# 2. 加载数据
# =========================
print("加载语义向量：", EMBEDDINGS_PATH)
embeddings = np.load(EMBEDDINGS_PATH)

print("加载语义元数据：", METADATA_PATH)
metadata = pd.read_csv(METADATA_PATH, dtype=str).fillna("")

print("embedding shape:", embeddings.shape)
print("metadata shape:", metadata.shape)


# =========================
# 3. 加载模型
# =========================
print("加载模型：", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)


# =========================
# 4. 语义检索函数
# =========================
def semantic_search(query, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 因为 embedding 已经归一化，所以点积就是 cosine similarity
    scores = np.dot(embeddings, query_embedding[0])

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()
    results["semantic_score"] = scores[top_indices]

    return results


# =========================
# 5. 交互测试
# =========================
def main():
    print("\n语义检索测试系统已启动")
    print("示例：")
    print("- texts about gods and kingship")
    print("- Osiris and afterlife")
    print("- offering rituals")
    print("- enemies of Osiris")
    print("- 太阳神和国王")
    print("输入 q 退出")

    while True:
        query = input("\n请输入语义查询：").strip()

        if query.lower() == "q":
            print("已退出。")
            break

        if not query:
            print("请输入有效查询。")
            continue

        results = semantic_search(query, top_k=5)

        print("\n" + "=" * 100)
        print("查询：", query)
        print("返回结果数：", len(results))

        for i, row in results.iterrows():
            print("\n" + "-" * 100)
            print("semantic_score:", round(float(row["semantic_score"]), 4))
            print("doc_id:", row["doc_id"])
            print("corpus:", row["corpus"])
            print("date:", row["date"])
            print("findspot:", row["findspot"])
            print("translation:", row["translation"])
            print("transliteration:", row["transliteration"])
            print("lemma_forms:", row["lemma_forms"])
            print("mdc:", row["mdc"])


if __name__ == "__main__":
    main()