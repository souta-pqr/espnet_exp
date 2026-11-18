#!/usr/bin/env python
"""waseda_soma の完了の区間のうち，CEJC のラベルの決め方では完了にならない 3 種を数えない tag を作る（2026-11-18 に silver13 でその場で作ったものを，スクリプトに書き起こした）．
  0.5 秒未満の応答だけの区間（対話行為が re だけ）・言い終わりの合図だけの区間（eom だけ）・フィラーだけの区間（fil・u だけ）
soma_prep.py の後に流す．<出力>/data_cejc/{segments,wav.scp,tag} を作る．

  python soma_cejc_tag.py
"""
import collections
import glob
import os
import re
import shutil
import xml.etree.ElementTree as ET

S = "/autofs/diamond2/share/corpus/waseda_soma/2022/eaf"
O = "/autofs/diamond5/share/users/kobori/waseda_soma"
os.makedirs(f"{O}/data_cejc", exist_ok=True)
for f in ("segments", "wav.scp"):
    shutil.copy(f"{O}/data/{f}", f"{O}/data_cejc/{f}")
E = {}
for f in sorted(glob.glob(S + "/*.eaf")):
    rid = os.path.basename(f)[:-4]
    if re.search(r"_\d{2}_\d$", rid):
        continue
    E.setdefault(re.sub(r"^(\d{8})_", r"\1-", rid), f)


def tiers(f):
    r = ET.parse(f).getroot()
    ts = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE", -1)) for t in r.iter("TIME_SLOT")}
    return {t.get("TIER_ID"): [(ts[a.get("TIME_SLOT_REF1")], ts[a.get("TIME_SLOT_REF2")], (a.find("ANNOTATION_VALUE").text or "").strip())
                               for a in t.iter("ALIGNABLE_ANNOTATION")] for t in r.iter("TIER")}


C = {}
c = collections.Counter()
with open(f"{O}/data_cejc/tag", "w") as fo:
    for l in open(f"{O}/data/tag"):
        u, lab = l.split()
        if lab == "1":
            m = re.match(r"(.+)_([UA])_(\d+)_(\d+)$", u)
            rid, s, b, e = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            if rid not in C:
                C[rid] = tiers(E[rid])
            who = "user" if s == "U" else "agent"
            T = C[rid]
            d = {(x, y): v for x, y, v in T.get(f"dialog_acts_{who}", [])}
            md = {(x, y): v for x, y, v in T.get(f"manual_dialog_acts_{who}", []) if v}
            a = {re.sub(r"\s*\(.*?\)", "", x).strip() for x in (md.get((b, e)) or d.get((b, e), "")).split("/")}
            k = "短い応答" if a == {"re"} and e - b < 500 else "言い終わりの合図" if a == {"eom"} else "フィラー" if a and a <= {"fil", "u"} else None
            if k:
                c[k] += 1
                continue
        fo.write(l)
        c["残す" + lab] += 1
print(dict(c))
