"""
evaluate.py — Évaluation qualité STT
Métriques : WER normalisé + CER + BLEU
Sources   : Whisper vs CC YouTube (alignés par timestamps réels)

Usage : python3 evaluate.py
Génère : ../outputs/eval_results.json
"""

import json, re, os
from jiwer import wer, cer
import sacrebleu

BASE          = os.path.dirname(os.path.abspath(__file__))
DATA          = os.path.join(BASE, "..", "data")
OUT           = os.path.join(BASE, "..", "outputs")
TRANSCRIPTION = os.path.join(DATA, "transcription.json")
REFERENCE     = os.path.join(DATA, "reference.txt")
VTT           = os.path.join(DATA, "france24.fr.srt")
os.makedirs(OUT, exist_ok=True)

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("📂 Chargement des données...")
with open(TRANSCRIPTION, encoding="utf-8") as f:
    whisper_data = json.load(f)
whisper_segs = whisper_data["segments"]

with open(REFERENCE, encoding="utf-8") as f:
    ref_lines = [l.strip() for l in f if l.strip()]

n = min(len(whisper_segs), len(ref_lines))
whisper_segs = whisper_segs[:n]
ref_lines    = ref_lines[:n]
print(f"   ✅ {n} segments chargés")

# ── Extraction CC YouTube — texte linéaire complet ───────────────
# On reconstruit le texte CC dans l'ordre sans répétitions,
# puis on l'aligne MOT À MOT sur la référence (même logique que Whisper).

print("\n🕐 Alignement CC YouTube...")

with open(VTT, encoding="utf-8") as f:
    vtt_content = f.read()

# 1. Extraire tous les blocs utiles (dur > 50ms)
raw_blocks = []
for block in re.split(r'\n\n+', vtt_content.strip()):
    lines = block.strip().split('\n')
    ts = next((l for l in lines if re.match(r'\d{2}:\d{2}:\d{2}.*-->', l)), None)
    if not ts: continue
    m = re.match(r'(\d{2}:\d{2}:\d{2})[.,](\d+)\s*-->\s*(\d{2}:\d{2}:\d{2})[.,](\d+)', ts)
    if not m: continue
    def to_s(h, ms): 
        a,b,c = map(int,h.split(':')); return a*3600+b*60+c+int(ms)/1000
    dur = to_s(m.group(3),m.group(4)) - to_s(m.group(1),m.group(2))
    if dur < 0.05: continue  # ignorer transitions ~10ms
    txt_parts = []
    for l in lines:
        lc = re.sub(r'<[^>]+>', '', l).strip()
        if not lc: continue
        if re.match(r'^\d+$', lc): continue
        if re.match(r'\d{2}:\d{2}:\d{2}', lc): continue
        txt_parts.append(lc)
    txt = " ".join(txt_parts).strip()
    if txt:
        raw_blocks.append(txt)

# 2. Dédupliquer : retirer le préfixe répété entre blocs consécutifs
cc_words_all = []
prev_words = []
for txt in raw_blocks:
    cur = txt.split()
    overlap = 0
    for k in range(min(len(prev_words), len(cur)), 0, -1):
        if prev_words[-k:] == cur[:k]:
            overlap = k
            break
    cc_words_all.extend(cur[overlap:])
    prev_words = cur

print(f"   ✅ {len(cc_words_all)} mots CC extraits")
print(f"   Début : {' '.join(cc_words_all[:10])}")

# 3. Alignement séquentiel sur la référence :
#    Pour chaque segment, on prend exactement autant de mots CC
#    que le segment Whisper correspondant en contient.
#    → même découpage que Whisper, même longueur, alignement parfait.
# Alignement dynamique : pour chaque segment, on cherche dans le flux CC
# la fenêtre de mots qui ressemble le plus à la REF.
# On part de la position courante et on teste des fenêtres de taille
# variable autour de len(ref) pour trouver le meilleur score de similarité.

from difflib import SequenceMatcher

def best_window(cc_words, start, ref_words, slack=8):
    """
    Prend len(ref) mots par défaut.
    Si le dernier mot de la fenêtre ne ressemble pas du tout au dernier
    mot de la REF, on essaie d'étendre jusqu'à trouver un meilleur match
    ou d'atteindre slack mots supplémentaires.
    """
    n = len(ref_words)
    ref_last = ref_words[-1].lower().strip('.,;:!?«»') if ref_words else ""

    # Fenêtre de base
    base = cc_words[start: start + n]
    if not base:
        return n

    # Score du dernier mot de base vs dernier mot REF
    base_last = base[-1].lower().strip('.,;:!?«»') if base else ""
    base_score = SequenceMatcher(None, ref_last, base_last).ratio()

    # Si le dernier mot est déjà bon (>0.5), on garde la fenêtre de base
    if base_score >= 0.5:
        return n

    # Sinon on cherche dans les slack mots suivants un meilleur dernier mot
    best_k     = n
    best_score = base_score
    for extra in range(1, slack + 1):
        k = n + extra
        window = cc_words[start: start + k]
        if not window or start + k > len(cc_words):
            break
        w_last = window[-1].lower().strip('.,;:!?«»')
        score  = SequenceMatcher(None, ref_last, w_last).ratio()
        if score > best_score:
            best_score = score
            best_k     = k
        if score >= 0.8:  # assez bon, on s'arrête
            break

    return best_k

cursor = 0
cc_segs = []
for ref in ref_lines:
    ref_words = ref.split()
    ref0 = ref_words[0].lower().strip('.,;:!?«»') if ref_words else ""

    # Si le mot courant du flux ne ressemble pas au 1er mot de la REF,
    # on saute jusqu'à 3 mots pour trouver le bon point de départ.
    skip = 0
    for s in range(1, 4):
        if cursor + s >= len(cc_words_all):
            break
        w = cc_words_all[cursor + s].lower().strip('.,;:!?«»')
        w0 = cc_words_all[cursor].lower().strip('.,;:!?«»')
        # Skipp si le mot courant ne ressemble pas du tout à ref0
        # mais que le mot suivant ressemble mieux
        from difflib import SequenceMatcher as SM
        score_cur  = SM(None, ref0, w0).ratio()
        score_next = SM(None, ref0, w).ratio()
        if score_next > score_cur + 0.3:
            skip = s
            break

    cursor += skip
    k = best_window(cc_words_all, cursor, ref_words, slack=6)
    cc_segs.append(" ".join(cc_words_all[cursor: cursor + k]))
    cursor += k

print("\n   Vérification alignement (seg 01-06) :")
for i, (ref, cc) in enumerate(zip(ref_lines[:6], cc_segs[:6])):
    print(f"   Seg.{i+1:02d} REF : {ref[:65]}")
    print(f"          CC  : {cc[:65]}")
print(f"\n   ✅ CC alignés sur {n} segments")

# ── Métriques ─────────────────────────────────────────────────────
print("\n🔬 Calcul des métriques...")

def fmt(s): return f"{int(s)//60:02d}:{int(s)%60:02d}"

def quality_label(w):
    if w == 0:   return "Parfait"
    elif w < 10: return "Bon"
    elif w < 25: return "Acceptable"
    else:        return "À corriger"

def bleu_score(ref, hyp):
    try:
        return round(sacrebleu.sentence_bleu(hyp, [ref]).score, 1)
    except:
        return 0.0

records = []
for i, (seg, ref, cc) in enumerate(zip(whisper_segs, ref_lines, cc_segs)):
    stt   = seg["text"].strip()
    ref_n = normalize(ref)
    stt_n = normalize(stt)
    cc_n  = normalize(cc) if cc else ""

    records.append({
        "ID":          i + 1,
        "Début":       fmt(seg["start"]),
        "Fin":         fmt(seg["end"]),
        "start_s":     seg["start"],
        "end_s":       seg["end"],
        "Référence":   ref,
        "STT Whisper": stt,
        "WER brut W":  round(wer(ref, stt)       * 100, 1),
        "WER norm W":  round(wer(ref_n, stt_n)   * 100, 1),
        "CER W":       round(cer(ref_n, stt_n)   * 100, 1),
        "BLEU W":      bleu_score(ref, stt),
        "Qualité W":   quality_label(round(wer(ref_n, stt_n) * 100, 1)),
        "CC YouTube":  cc,
        "WER brut CC": round(wer(ref, cc)        * 100, 1) if cc else 0.0,
        "WER norm CC": round(wer(ref_n, cc_n)    * 100, 1) if cc_n else 0.0,
        "CER CC":      round(cer(ref_n, cc_n)    * 100, 1) if cc_n else 0.0,
        "BLEU CC":     bleu_score(ref, cc) if cc else 0.0,
        "Qualité CC":  quality_label(round(wer(ref_n, cc_n) * 100, 1) if cc_n else 0.0),
        "Wildmoka":    ("🔴 À corriger" if round(wer(ref_n, stt_n)*100,1) >= 25
                        else ("⚠️ Vérifier" if round(wer(ref_n, stt_n)*100,1) > 0
                              else "✅ OK")),
    })

ref_full = " ".join(ref_lines)
stt_full = " ".join(seg["text"].strip() for seg in whisper_segs)
cc_full  = " ".join(cc_segs)

global_stats = {
    "whisper": {
        "wer_brut": round(wer(ref_full, stt_full) * 100, 2),
        "wer_norm": round(wer(normalize(ref_full), normalize(stt_full)) * 100, 2),
        "cer":      round(cer(normalize(ref_full), normalize(stt_full)) * 100, 2),
        "bleu":     bleu_score(ref_full, stt_full),
    },
    "cc_youtube": {
        "wer_brut": round(wer(ref_full, cc_full) * 100, 2),
        "wer_norm": round(wer(normalize(ref_full), normalize(cc_full)) * 100, 2),
        "cer":      round(cer(normalize(ref_full), normalize(cc_full)) * 100, 2),
        "bleu":     bleu_score(ref_full, cc_full),
    }
}

print(f"\n{'='*62}")
print(f"  SubQuality — Résultats d'évaluation STT")
print(f"{'='*62}")
print(f"\n  {'Métrique':<28} {'Whisper':>10} {'CC YouTube':>12}")
print(f"  {'-'*50}")
for label, kw, kc in [
    ("WER brut (%)",            "wer_brut","wer_brut"),
    ("WER normalisé (%)",       "wer_norm","wer_norm"),
    ("CER (%)",                 "cer",     "cer"),
    ("BLEU (0-100, +haut=+bon)","bleu",   "bleu"),
]:
    print(f"  {label:<28} {global_stats['whisper'][kw]:>10} {global_stats['cc_youtube'][kc]:>12}")
print(f"{'='*62}\n")

output = {
    "global":       global_stats,
    "segments":     records,
    "n_segments":   n,
    "n_a_corriger": sum(1 for r in records if r["WER norm W"] >= 25),
    "n_parfaits":   sum(1 for r in records if r["WER norm W"] == 0),
}
out_path = os.path.join(OUT, "eval_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  💾 Résultats sauvegardés → outputs/eval_results.json\n")
