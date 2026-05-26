from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

MAIN_DOCS_CSV = PROJECT_DIR / "data_demo" / "main_documents.csv"

OUTPUT_DIR = PROJECT_DIR / "data_semantic_demo"
EMBEDDINGS_PATH = OUTPUT_DIR / "semantic_embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "semantic_metadata.csv"
CONFIG_PATH = OUTPUT_DIR / "semantic_config.json"


# =========================
# 2. 模型设置
# =========================
# 这个模型较轻量，适合本地和 Streamlit demo
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 为了先稳定跑通，展示版最多取 8000 条做语义索引
# 后面确认没问题可以改成 12000 或 19358
MAX_SEMANTIC_DOCS = 8000


# =========================
# 3. 构造语义文本
# =========================
def build_semantic_text(row):
    """
    将每条主文档转换成适合做语义向量的文本。
    这里主要使用 translation，同时拼接少量元数据。
    """
    translation = str(row.get("translation", "")).strip()
    corpus = str(row.get("corpus", "")).strip()
    date = str(row.get("date", "")).strip()
    findspot = str(row.get("findspot", "")).strip()
    transliteration = str(row.get("transliteration", "")).strip()

    text_parts = []

    if translation:
        text_parts.append(f"Translation: {translation}")

    if transliteration:
        text_parts.append(f"Transliteration: {transliteration}")

    if corpus:
        text_parts.append(f"Corpus: {corpus}")

    if date:
        text_parts.append(f"Date: {date}")

    if findspot:
        text_parts.append(f"Findspot: {findspot}")

    return " | ".join(text_parts)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取展示版主文档：", MAIN_DOCS_CSV)
    df = pd.read_csv(MAIN_DOCS_CSV, dtype=str).fillna("")

    print("展示版主文档数量：", len(df))

    # 只保留有 translation 或 transliteration 的记录
    df["semantic_text"] = df.apply(build_semantic_text, axis=1)

    df = df[df["semantic_text"].str.strip() != ""].copy()

    print("可用于语义索引的记录数：", len(df))

    # 为了第一次构建速度稳定，先抽取部分文档
    if len(df) > MAX_SEMANTIC_DOCS:
        df = df.sample(MAX_SEMANTIC_DOCS, random_state=42).reset_index(drop=True)

    print("本次构建语义索引记录数：", len(df))

    print("\n加载语义模型：", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = df["semantic_text"].tolist()

    print("\n开始生成 embedding，这一步可能需要几分钟...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("embedding shape:", embeddings.shape)

    # 保存向量
    np.save(EMBEDDINGS_PATH, embeddings)

    # 保存元数据
    metadata_cols = [
        "doc_id",
        "sentence_id",
        "corpus",
        "date",
        "findspot",
        "translation",
        "transliteration",
        "normalized_transliteration",
        "lemma_forms",
        "mdc",
        "semantic_text"
    ]

    metadata_cols = [c for c in metadata_cols if c in df.columns]
    metadata_df = df[metadata_cols].copy()
    metadata_df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    # 保存配置
    config = {
        "model_name": MODEL_NAME,
        "embedding_file": str(EMBEDDINGS_PATH),
        "metadata_file": str(METADATA_PATH),
        "num_docs": int(len(metadata_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n语义索引生成完成！")
    print("向量文件：", EMBEDDINGS_PATH)
    print("元数据文件：", METADATA_PATH)
    print("配置文件：", CONFIG_PATH)

    print("\n前 5 条语义索引元数据：")
    print(metadata_df[["doc_id", "corpus", "translation"]].head())


if __name__ == "__main__":
    main()