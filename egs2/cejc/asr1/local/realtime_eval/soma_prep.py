"""waseda_soma（人がターンのラベルを付けた 95 対話）を，CEJC と同じ形（話者ごとの音声・区間・3 つのラベル）にする（2026-11-18）．
区間は ELAN の音声区間（IPU）．ラベルは人のターンの注釈から作る：
  相槌(2)：対話行為が相槌（bc）か了解（ack）だけの区間
  継続(0)：同じ話者の次の区間が「前のターンの継続（turn_id=1）」，または次の区間と結合（+f）
  完了(1)：同じ話者の次の区間が「ターンの開始（turn_id=0）」，またはその後に区間が無い
  同じ対話に 2 人目の注釈（_2）があるものは 1 人目だけを使う．turn_id をターンの通し番号で付けた注釈は，同じ番号が続けば継続とみなす．雑音（turn_id=-1）の区間は数えない．次の区間の turn_id が 0・1 以外なら，その区間は数えない
音声は話者ごとに 1 チャンネルを取り出し，相手の区間（自分の区間の前後 0.1 秒を除く）を無音にする（マイクへの漏れ込みが大きいため）．"""
import glob, os, re, wave, collections
import xml.etree.ElementTree as ET
import numpy as np
S = "/autofs/diamond2/share/corpus/waseda_soma"; O = "/autofs/diamond5/share/users/kobori/waseda_soma"
os.makedirs(f"{O}/wav", mode=0o700, exist_ok=True); os.makedirs(f"{O}/data", exist_ok=True); os.chmod(O, 0o700)
def tiers(f):
    r = ET.parse(f).getroot(); ts = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE", -1)) for t in r.iter("TIME_SLOT")}
    return {t.get("TIER_ID"): [(ts[a.get("TIME_SLOT_REF1")] / 1000, ts[a.get("TIME_SLOT_REF2")] / 1000, (a.find("ANNOTATION_VALUE").text or "").strip())
                               for a in t.iter("ALIGNABLE_ANNOTATION")] for t in r.iter("TIER")}
def acts(v):
    return {re.sub(r"\s*\(.*?\)", "", x).strip() for x in v.split("/") if x.strip()}
fs, ft, fw = (open(f"{O}/data/{n}", "w") for n in ("segments", "tag", "wav.scp"))
cnt = collections.Counter(); done = set()


def to01(tid):
    """turn_id をターンの通し番号で付けた注釈（同じ番号＝同じターン）は，0（ターンの開始）・1（継続）に直す．-7888 のような値は不明"""
    v = sorted(tid)
    if all(x[2] in ("0", "1", "-1") for x in v): return v
    out, prev = [], None
    for b, e, x in v:
        if x == "-1": out.append((b, e, x)); continue
        if not x.isdigit(): out.append((b, e, "?")); prev = None; continue
        out.append((b, e, "1" if x == prev else "0")); prev = x
    cnt["通し番号の注釈"] += 1
    return out


for f in sorted(glob.glob(f"{S}/2022/eaf/*.eaf")):
    rid = os.path.basename(f)[:-4]
    if re.search(r"_\d{2}_\d$", rid): cnt["2 人目の注釈（使わない）"] += 1; continue
    rid = re.sub(r"^(\d{8})_", r"\1-", rid)
    if rid in done: cnt["同じ対話の 2 つ目の注釈（使わない）"] += 1; continue
    done.add(rid); T = tiers(f)
    w = wave.open(f"{S}/wav/{rid}.wav"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)
    for spk, ch, oth in (("user", 0, "agent"), ("agent", 1, "user")):
        tid, da = to01(T.get(f"turn_id_{spk}", [])), T.get(f"dialog_acts_{spk}", [])
        man = {(b, e): v for b, e, v in T.get(f"manual_dialog_acts_{spk}", []) if v}
        dam = {(b, e): v for b, e, v in da}
        ipu = sorted((b, e, v, man.get((b, e)) or dam.get((b, e), "")) for b, e, v in tid if v != "-1")
        if not ipu: cnt["話者の区間なし"] += 1; continue
        rec = f"{rid}_{'U' if spk == 'user' else 'A'}"
        y = x[:, ch].copy(); keep = np.zeros(len(y), bool)
        for b, e, *_ in sorted(T.get(f"turn_id_{spk}", [])):
            keep[max(0, int((b - 0.1) * 16000)):int((e + 0.1) * 16000)] = True
        for b, e, *_ in T.get(f"turn_id_{oth}", []):
            s0, s1 = int(b * 16000), int(e * 16000); y[s0:s1][~keep[s0:s1]] = 0
        o = wave.open(f"{O}/wav/{rec}.wav", "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000); o.writeframes(y.tobytes()); o.close()
        fw.write(f"{rec} {O}/wav/{rec}.wav\n")
        for j, (b, e, v, d) in enumerate(ipu):
            a = acts(d); nxt = ipu[j + 1][2] if j + 1 < len(ipu) else "0"
            if v not in ("0", "1"): cnt["自分の turn_id が不明"] += 1; continue
            if "+f" in a: lab = 0
            elif a and a <= {"bc", "ack"}: lab = 2
            elif nxt == "1": lab = 0
            elif nxt == "0": lab = 1
            else: cnt["次の turn_id が不明"] += 1; continue
            u = f"{rec}_{int(b * 1000):07d}_{int(e * 1000):07d}"
            fs.write(f"{u} {rec} {b:.3f} {e:.3f}\n"); ft.write(f"{u} {lab}\n"); cnt[(spk, lab)] += 1
for k, v in sorted(cnt.items(), key=str): print(k, v)
