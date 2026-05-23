from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")
OUTPUT_CSV = PROJECT_DIR / "data_processed" / "query_expansion.csv"


query_expansions = [
    {
        "query_zh": "神",
        "topic_en": "god / deity",
        "expanded_terms": "god,deity,gott,ntr",
        "explanation_zh": "nṯr / ntr 是古埃及语中常见的“神”相关词项，也可对应英文 god、deity 和德文 Gott。"
    },
    {
        "query_zh": "大神",
        "topic_en": "great god",
        "expanded_terms": "great,god,ntr,aa,ntr-aa",
        "explanation_zh": "nṯr-ꜥꜣ / ntr-aa 常可理解为“大神”。"
    },
    {
        "query_zh": "奥西里斯",
        "topic_en": "Osiris",
        "expanded_terms": "osiris,wsjr",
        "explanation_zh": "Wsjr 是奥西里斯 Osiris 的常见古埃及语转写形式。"
    },
    {
        "query_zh": "国王",
        "topic_en": "king / ruler",
        "expanded_terms": "king,ruler,nswt,koenig,könig",
        "explanation_zh": "nswt 是古埃及语中与“国王”相关的重要词项。"
    },
    {
        "query_zh": "太阳神",
        "topic_en": "sun god / Ra",
        "expanded_terms": "sun,solar,ra,re,raw,sonnengott",
        "explanation_zh": "Rꜥw / raw 通常与太阳神 Ra / Re 相关。"
    },
    {
        "query_zh": "拉神",
        "topic_en": "Ra / Re",
        "expanded_terms": "ra,re,raw,sun,solar",
        "explanation_zh": "拉神通常对应 Ra / Re，在转写中可见 Rꜥw / raw。"
    },
    {
        "query_zh": "来世",
        "topic_en": "afterlife / underworld",
        "expanded_terms": "afterlife,underworld,jenseits,duat,dwat",
        "explanation_zh": "该主题用于检索与死后世界、冥界、来世相关的文本。"
    },
    {
        "query_zh": "冥界",
        "topic_en": "underworld / Duat",
        "expanded_terms": "underworld,duat,dwat,jenseits",
        "explanation_zh": "Duat / dwat 通常与古埃及冥界概念相关。"
    },
    {
        "query_zh": "供奉",
        "topic_en": "offering",
        "expanded_terms": "offering,opfer,htp",
        "explanation_zh": "ḥtp / htp 常与供奉、祭献、安宁等意义相关。"
    },
    {
        "query_zh": "祭品",
        "topic_en": "offering",
        "expanded_terms": "offering,opfer,htp",
        "explanation_zh": "祭品和供奉主题可通过 offering、Opfer、htp 等词检索。"
    },
    {
        "query_zh": "死亡",
        "topic_en": "death",
        "expanded_terms": "death,dead,tote,mwt",
        "explanation_zh": "该主题用于检索与死亡、死者相关的文本。"
    },
    {
        "query_zh": "死者",
        "topic_en": "deceased / dead",
        "expanded_terms": "dead,deceased,tote,mwt",
        "explanation_zh": "该主题用于检索与死者、亡者相关的表达。"
    },
    {
        "query_zh": "天空",
        "topic_en": "sky / heaven",
        "expanded_terms": "sky,heaven,himmel,pt",
        "explanation_zh": "p.t / pt 常与天空相关，英文可对应 sky、heaven，德文可对应 Himmel。"
    },
    {
        "query_zh": "神庙",
        "topic_en": "temple",
        "expanded_terms": "temple,tempel,pr",
        "explanation_zh": "该主题用于检索与神庙、庙宇建筑相关的文本。"
    },
    {
        "query_zh": "墓葬",
        "topic_en": "tomb / grave",
        "expanded_terms": "tomb,grave,grab",
        "explanation_zh": "该主题用于检索与墓葬、坟墓相关的文本。"
    },
    {
        "query_zh": "敌人",
        "topic_en": "enemy",
        "expanded_terms": "enemy,enemies,feind,feinde,hfti",
        "explanation_zh": "该主题用于检索与敌人、对抗者相关的文本。"
    },
    {
        "query_zh": "真理",
        "topic_en": "truth / Maat",
        "expanded_terms": "truth,maat,maa",
        "explanation_zh": "mꜣꜥ / maa 常与真实、正义、玛阿特相关。"
    },
    {
        "query_zh": "书吏",
        "topic_en": "scribe",
        "expanded_terms": "scribe,schreiber,ss",
        "explanation_zh": "sš / ss 通常与书吏、书写相关。"
    },
    {
        "query_zh": "生命",
        "topic_en": "life",
        "expanded_terms": "life,leben,ankh,anh",
        "explanation_zh": "ꜥnḫ / ankh / anh 常与生命相关。"
    },
    {
        "query_zh": "赫鲁斯",
        "topic_en": "Horus",
        "expanded_terms": "horus,hr",
        "explanation_zh": "Horus 常对应古埃及转写 Ḥr / hr。"
    }
]


df = pd.DataFrame(query_expansions)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("中文查询扩展表已生成：", OUTPUT_CSV)
print("记录数：", len(df))
print(df[["query_zh", "topic_en", "expanded_terms"]])