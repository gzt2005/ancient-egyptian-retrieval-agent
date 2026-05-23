from pathlib import Path
import json
import re
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

AES_JSON_DIR = PROJECT_DIR / "data_raw" / "aes" / "files" / "aes"

OUTPUT_DIR = PROJECT_DIR / "data_processed"
OUTPUT_CSV = OUTPUT_DIR / "sentences.csv"


# =========================
# 2. 古埃及语转写归一化函数
# =========================
def normalize_transliteration(text: str) -> str:
    """
    将古埃及语转写中的特殊字符简化，方便后续普通键盘检索。
    例如：
    nṯr -> ntr
    ꜥnḫ -> anh
    ḏd -> dd

    注意：原始转写仍然会保存在 transliteration 字段里。
    """
    if not isinstance(text, str):
        return ""

    mapping = {
        "ꜣ": "a",
        "ꜥ": "a",
        "ȝ": "a",
        "ʾ": "a",
        "ḏ": "d",
        "Ḏ": "D",
        "ḥ": "h",
        "Ḥ": "H",
        "ḫ": "h",
        "ẖ": "h",
        "ḳ": "q",
        "š": "s",
        "Š": "S",
        "ṯ": "t",
        "Ṯ": "T",
        "ṱ": "t",
        "ỉ": "i",
        "ī": "i",
        "ū": "u",
        "ꞽ": "i",
    }

    for old, new in mapping.items():
        text = text.replace(old, new)

    # 去掉一些标注符号，保留字母、数字、空格、连字符、点号、下划线
    text = re.sub(r"[^A-Za-z0-9\s\-_.]", "", text)

    # 多空格合并
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


# =========================
# 3. 处理一个 sentence
# =========================
def parse_sentence(sentence_id: str, sentence_data: dict, source_file: str) -> dict:
    """
    将一个句子节点解析成一行结构化数据。
    """

    text_id = sentence_data.get("text", "")
    owner = sentence_data.get("owner", "")
    corpus = sentence_data.get("corpus", "")
    date = sentence_data.get("date", "")
    findspot = sentence_data.get("findspot", "")
    sentence_translation = sentence_data.get("sentence_translation", "")

    tokens = sentence_data.get("token", [])

    written_forms = []
    mdc_forms = []
    lemma_forms = []
    lemma_ids = []
    pos_tags = []
    cotext_translations = []
    hiero_inventars = []
    hiero_unicodes = []
    hieros = []

    if isinstance(tokens, list):
        for token in tokens:
            if not isinstance(token, dict):
                continue

            written_form = token.get("written_form", "")
            mdc = token.get("mdc", "")
            lemma_form = token.get("lemma_form", "")
            lemma_id = token.get("lemmaID", "")
            pos = token.get("pos", "")
            cotext_translation = token.get("cotext_translation", "")
            hiero_inventar = token.get("hiero_inventar", "")
            hiero_unicode = token.get("hiero_unicode", "")
            hiero = token.get("hiero", "")

            if written_form:
                written_forms.append(str(written_form))

            if mdc:
                mdc_forms.append(str(mdc))

            if lemma_form:
                lemma_forms.append(str(lemma_form))

            if lemma_id:
                lemma_ids.append(str(lemma_id))

            if pos:
                pos_tags.append(str(pos))

            if cotext_translation:
                cotext_translations.append(str(cotext_translation))

            if hiero_inventar:
                hiero_inventars.append(str(hiero_inventar))

            if hiero_unicode:
                hiero_unicodes.append(str(hiero_unicode))

            if hiero:
                hieros.append(str(hiero))

    transliteration = " ".join(written_forms)
    normalized_transliteration = normalize_transliteration(transliteration)

    mdc_text = " ".join(mdc_forms)
    lemma_forms_text = " | ".join(lemma_forms)
    lemma_ids_text = " | ".join(lemma_ids)
    pos_tags_text = " | ".join(pos_tags)
    cotext_translations_text = " | ".join(cotext_translations)
    hiero_inventar_text = " | ".join(hiero_inventars)
    hiero_unicode_text = " | ".join(hiero_unicodes)
    hiero_text = " | ".join(hieros)

    return {
        "sentence_id": sentence_id,
        "text_id": text_id,
        "owner": owner,
        "corpus": corpus,
        "date": date,
        "findspot": findspot,
        "sentence_translation": sentence_translation,
        "transliteration": transliteration,
        "normalized_transliteration": normalized_transliteration,
        "mdc": mdc_text,
        "lemma_forms": lemma_forms_text,
        "lemma_ids": lemma_ids_text,
        "pos_tags": pos_tags_text,
        "cotext_translations": cotext_translations_text,
        "hiero_inventar": hiero_inventar_text,
        "hiero_unicode": hiero_unicode_text,
        "hiero": hiero_text,
        "source_file": source_file,
        "token_count": len(tokens) if isinstance(tokens, list) else 0,
    }


# =========================
# 4. 主程序：遍历所有 JSON
# =========================
def main():
    print("AES JSON 文件夹：", AES_JSON_DIR)
    print("文件夹是否存在：", AES_JSON_DIR.exists())

    json_files = list(AES_JSON_DIR.glob("*.json"))

    # 排除 schema 文件
    json_files = [f for f in json_files if f.name != "aesschema.json"]

    print("待处理语料 JSON 文件数量：", len(json_files))

    rows = []

    for json_file in json_files:
        print("正在处理：", json_file.name)

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("读取失败：", json_file.name)
            print("错误信息：", e)
            continue

        if not isinstance(data, dict):
            print("跳过：最外层不是 dict", json_file.name)
            continue

        for sentence_id, sentence_data in data.items():
            if not isinstance(sentence_data, dict):
                continue

            row = parse_sentence(
                sentence_id=sentence_id,
                sentence_data=sentence_data,
                source_file=json_file.name
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    print("\n解析完成！")
    print("总句子数量：", len(df))

    if len(df) > 0:
        print("\n字段列表：")
        print(df.columns.tolist())

        print("\n前 5 行预览：")
        print(df.head())

        print("\n各 corpus 数量：")
        print(df["corpus"].value_counts().head(20))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n已保存 CSV：", OUTPUT_CSV)


if __name__ == "__main__":
    main()