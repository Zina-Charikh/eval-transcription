"""
SubQuality — Application Streamlit
Usage : streamlit run app.py (depuis le dossier scripts/)
"""

import json, re, os, difflib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="SubQuality — France 24 × Wildmoka",
    page_icon="🎬", layout="wide",
    initial_sidebar_state="collapsed",
)

BASE       = os.path.dirname(os.path.abspath(__file__))
OUT        = os.path.join(BASE, "..", "outputs")
VIDEO_PATH = os.path.join(BASE, "..", "france24.mp4")
EVAL_PATH  = os.path.join(OUT,  "eval_results.json")
FRAMES_W   = os.path.join(BASE, "..", "frames", "whisper")
FRAMES_C   = os.path.join(BASE, "..", "frames", "cc")

COLORS_Q = {
    "Parfait":    "#10b981",
    "Bon":        "#f59e0b",
    "Acceptable": "#f97316",
    "À corriger": "#ef4444",
}

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top:1rem; padding-bottom:1rem; }
  .kpi {
    background:white; border-radius:12px; padding:16px 10px 12px;
    border-top:4px solid; box-shadow:0 2px 8px rgba(0,0,0,.07);
    text-align:center;
  }
  .kpi-val { font-size:1.8rem; font-weight:700; margin:4px 0; }
  .kpi-lbl { font-size:.76rem; color:#64748b; margin:0; }
  .seg-card {
    background:white; border-radius:10px; padding:12px 16px;
    margin-bottom:8px; border-left:5px solid;
    box-shadow:0 1px 4px rgba(0,0,0,.06);
  }
  .tag {
    display:inline-block; padding:2px 9px;
    border-radius:20px; font-size:.74rem; font-weight:600;
  }
  .wm-card {
    background:white; border-radius:10px; padding:14px 18px;
    margin-bottom:12px; border-left:6px solid;
    box-shadow:0 2px 6px rgba(0,0,0,.07);
  }
  .stTabs [data-baseweb="tab"] {
    font-size:.95rem; font-weight:600; padding:10px 22px;
  }
</style>
""", unsafe_allow_html=True)

# ── Données ───────────────────────────────────────────────────────
@st.cache_data
def load_eval():
    if not os.path.exists(EVAL_PATH):
        return None
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)

ev = load_eval()
if ev is None:
    st.error("⚠️ Lance d'abord `python3 evaluate.py`.")
    st.stop()

df        = pd.DataFrame(ev["segments"])
n         = ev["n_segments"]
nbc       = ev["n_a_corriger"]
nbp       = ev["n_parfaits"]
gw        = ev["global"]["whisper"]["wer_norm"]
gc        = ev["global"]["whisper"]["cer"]
gb        = ev["global"]["whisper"]["bleu"]
cc_wer    = ev["global"]["cc_youtube"]["wer_norm"]
cc_cer    = ev["global"]["cc_youtube"]["cer"]
cc_bleu   = ev["global"]["cc_youtube"]["bleu"]

# ── Utilitaires ───────────────────────────────────────────────────
def mark_errors(ref, hyp):
    if not hyp or not hyp.strip():
        return []
    ref_w   = re.sub(r"[^\w\s]", "", ref.lower()).split()
    hyp_w   = hyp.split()
    hyp_cln = [re.sub(r"[^\w\s]", "", w.lower()) for w in hyp_w]
    result  = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, ref_w, hyp_cln).get_opcodes():
        if op == "equal":
            result += [(w, False) for w in hyp_w[j1:j2]]
        elif op in ("replace", "insert"):
            result += [(w, True) for w in hyp_w[j1:j2]]
    return result

def errors_html(ref, hyp):
    parts = []
    for w, err in mark_errors(ref, hyp):
        if err:
            parts.append(f"<span style='background:#fee2e2;color:#ef4444;"
                         f"border-radius:4px;padding:1px 6px;font-weight:600'>{w}</span>")
        else:
            parts.append(w)
    return " ".join(parts)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#1e3a5f;margin-bottom:4px'>🎬 SubQuality</h1>
<p style='color:#64748b;font-size:1rem;margin-top:0'>
  Analyse automatique de la qualité des sous-titres —
  <b>France 24</b> × <b>Whisper</b> × <b>CC YouTube</b> × <b>Wildmoka</b>
</p>
""", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎥 Lecteur vidéo", "🎬 Wildmoka"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab1:

    # KPIs
    cols = st.columns(6)
    for col, (val, lbl, color) in zip(cols, [
        (f"{gw}%",    "WER Whisper",       "#0ea5e9"),
        (f"{gc}%",    "CER Whisper",       "#8b5cf6"),
        (f"{gb}",     "BLEU Whisper /100", "#0ea5e9"),
        (f"{cc_wer}%","WER CC YouTube",    "#10b981"),
        (f"{nbp}/{n}","Segments parfaits", "#10b981"),
        (str(nbc),    "Clips Wildmoka ⚠️","#ef4444"),
    ]):
        with col:
            st.markdown(f"""<div class="kpi" style="border-top-color:{color}">
              <div class="kpi-val" style="color:{color}">{val}</div>
              <p class="kpi-lbl">{lbl}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("ℹ️ Comprendre WER normalisé, CER et BLEU"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**WER normalisé** — Word Error Rate sans ponctuation ni majuscules. "
                        "*Ex :* \"Île-de-France\" vs \"île de france\" → WER brut 33% → normalisé **0%**.")
        with c2:
            st.markdown("**CER** — Character Error Rate lettre par lettre. "
                        "*Ex :* \"Tabarot\" vs \"Tabaro\" → WER 100% mais CER **14%** (1 lettre manquante).")
        with c3:
            st.markdown(f"**BLEU** (0–100) — Qualité des séquences de mots. "
                        f"**>60** = bon, **>80** = excellent. Whisper : **{gb}** · CC YT : **{cc_bleu}**.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique WER par segment
    st.subheader("WER normalisé par segment — Whisper")
    fig_wer = go.Figure()
    for _, r in df.iterrows():
        fig_wer.add_trace(go.Bar(
            x=[f"S{r['ID']:02d}"], y=[r["WER norm W"]],
            marker_color=COLORS_Q[r["Qualité W"]], showlegend=False,
            hovertemplate=(f"<b>Seg.{r['ID']:02d}</b> · {r['Début']}→{r['Fin']}<br>"
                           f"WER : <b>{r['WER norm W']}%</b> · CER : {r['CER W']}%"
                           f" · BLEU : {r['BLEU W']}<br>"
                           f"<i>{r['Référence'][:60]}</i><extra></extra>")
        ))
    fig_wer.add_hline(y=gw, line_dash="dash", line_color="#0ea5e9",
                      annotation_text=f"Moy. {gw}%", annotation_position="top right")
    fig_wer.add_hline(y=25, line_dash="dot", line_color="#f59e0b",
                      annotation_text="Seuil Wildmoka 25%", annotation_position="top left")
    fig_wer.update_layout(height=320, paper_bgcolor="white", plot_bgcolor="#f8fafc",
                          margin=dict(l=20,r=20,t=20,b=20), yaxis_title="WER (%)")
    st.plotly_chart(fig_wer, use_container_width=True)

    col_a, col_b, col_c = st.columns([1.2, 1.8, 1.5])
    with col_a:
        st.subheader("Distribution qualité")
        qc = df["Qualité W"].value_counts()
        fig_d = go.Figure(go.Pie(labels=qc.index, values=qc.values, hole=0.55,
                                  marker_colors=[COLORS_Q[q] for q in qc.index],
                                  textinfo="label+percent", textfont_size=11))
        fig_d.update_layout(height=250, paper_bgcolor="white",
                            margin=dict(l=5,r=5,t=5,b=5), showlegend=False)
        st.plotly_chart(fig_d, use_container_width=True)

    with col_b:
        st.subheader("Whisper vs CC YouTube")
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(name="Whisper", x=["WER","CER","BLEU"],
                               y=[gw, gc, gb], marker_color="#0ea5e9",
                               text=[gw, gc, gb], textposition="outside"))
        fig_c.add_trace(go.Bar(name="CC YouTube", x=["WER","CER","BLEU"],
                               y=[cc_wer, cc_cer, cc_bleu], marker_color="#10b981",
                               text=[cc_wer, cc_cer, cc_bleu], textposition="outside"))
        fig_c.update_layout(height=250, barmode="group", paper_bgcolor="white",
                            plot_bgcolor="#f8fafc", margin=dict(l=5,r=5,t=5,b=5),
                            legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_c, use_container_width=True)

    with col_c:
        st.subheader("WER vs CER")
        fig_s = px.scatter(df, x="WER norm W", y="CER W", color="Qualité W",
                           color_discrete_map=COLORS_Q, text="ID",
                           hover_data={"Référence":True,"WER norm W":True,"CER W":True})
        fig_s.update_traces(textposition="top center", textfont_size=9)
        fig_s.update_layout(height=250, paper_bgcolor="white", plot_bgcolor="#f8fafc",
                            margin=dict(l=5,r=5,t=5,b=5), showlegend=False,
                            xaxis_title="WER (%)", yaxis_title="CER (%)")
        st.plotly_chart(fig_s, use_container_width=True)

    st.subheader("BLEU score par segment")
    fig_bleu = go.Figure()
    fig_bleu.add_trace(go.Bar(x=[f"S{r['ID']:02d}" for _,r in df.iterrows()],
                              y=df["BLEU W"],
                              marker_color=[COLORS_Q[q] for q in df["Qualité W"]],
                              hovertemplate="Seg.%{x}<br>BLEU : %{y}<extra></extra>"))
    fig_bleu.add_hline(y=60, line_dash="dot", line_color="#10b981", annotation_text="Seuil bon (60)")
    fig_bleu.update_layout(height=200, paper_bgcolor="white", plot_bgcolor="#f8fafc",
                           margin=dict(l=20,r=20,t=20,b=20),
                           yaxis=dict(title="BLEU", range=[0,105]), showlegend=False)
    st.plotly_chart(fig_bleu, use_container_width=True)

    # Tableau filtrable
    st.subheader("Détail par segment")
    f1, f2 = st.columns([1, 2])
    with f1:
        source = st.radio("Source :", ["Whisper","CC YouTube"], horizontal=True)
    with f2:
        filtre = st.selectbox("Qualité :", ["Tous","Parfait","Bon","Acceptable","À corriger"])

    col_q    = "Qualité W"   if source=="Whisper" else "Qualité CC"
    col_stt  = "STT Whisper" if source=="Whisper" else "CC YouTube"
    col_wer2 = "WER norm W"  if source=="Whisper" else "WER norm CC"
    col_cer2 = "CER W"       if source=="Whisper" else "CER CC"
    col_bl2  = "BLEU W"      if source=="Whisper" else "BLEU CC"

    dv = df if filtre=="Tous" else df[df[col_q]==filtre]
    for _, r in dv.iterrows():
        cb = COLORS_Q[r[col_q]]
        st.markdown(f"""
        <div class="seg-card" style="border-left-color:{cb}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:700;color:{cb}">
              Seg.{r['ID']:02d} &nbsp;·&nbsp; {r['Début']} → {r['Fin']}
            </span>
            <span>
              <span class="tag" style="background:{cb}22;color:{cb}">{r[col_q]}</span> &nbsp;
              <span class="tag" style="background:#f1f5f9;color:#475569">WER {r[col_wer2]}%</span> &nbsp;
              <span class="tag" style="background:#f5f3ff;color:#8b5cf6">CER {r[col_cer2]}%</span> &nbsp;
              <span class="tag" style="background:#f0fdf4;color:#10b981">BLEU {r[col_bl2]}</span>
            </span>
          </div>
          <p style="color:#16a34a;font-size:.85rem;margin:6px 0 3px">
            📝 <b>REF :</b> {r['Référence']}
          </p>
          <p style="color:#374151;font-size:.85rem;margin:0">
            🤖 <b>{source} :</b> {r[col_stt]}
          </p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — LECTEUR VIDÉO
# ══════════════════════════════════════════════════════════════════
with tab2:

    # Lire le segment actif depuis query params (mis à jour par le JS)
    qp = st.query_params
    auto_seg = int(qp.get("seg", 0))
    auto_seg = max(0, min(auto_seg, n - 1))

    # Sélecteurs en haut
    ctrl1, ctrl2 = st.columns([1, 3])
    with ctrl1:
        src_v = st.radio("Source STT :", ["Whisper","CC YouTube"],
                         horizontal=False, key="vs")
    col_stt_v = "STT Whisper" if src_v=="Whisper" else "CC YouTube"
    col_wer_v = "WER norm W"  if src_v=="Whisper" else "WER norm CC"
    col_cer_v = "CER W"       if src_v=="Whisper" else "CER CC"
    col_bl_v  = "BLEU W"      if src_v=="Whisper" else "BLEU CC"
    col_q_v   = "Qualité W"   if src_v=="Whisper" else "Qualité CC"

    with ctrl2:
        labels = [f"Seg.{r['ID']:02d}  ·  {r['Début']} → {r['Fin']}  ·  "
                  f"WER {r[col_wer_v]}%  ·  {r[col_q_v]}"
                  for _, r in df.iterrows()]

        nav1, nav2, nav3 = st.columns([1, 8, 1])
        with nav1:
            if st.button("◀", key="prev_seg") and auto_seg > 0:
                st.query_params["seg"] = auto_seg - 1
                st.rerun()
        with nav2:
            sel = st.selectbox("Segment :", labels, index=auto_seg, key="seg_sel")
        with nav3:
            if st.button("▶", key="next_seg") and auto_seg < n - 1:
                st.query_params["seg"] = auto_seg + 1
                st.rerun()

    seg_idx = labels.index(sel)
    if seg_idx != auto_seg:
        st.query_params["seg"] = seg_idx
    row     = df.iloc[seg_idx]
    cb      = COLORS_Q[row[col_q_v]]
    stt_txt = str(row[col_stt_v]) if row[col_stt_v] else ""

    # Timestamps de tous les segments pour le JS
    seg_times = [(float(r["start_s"]), float(r["end_s"])) for _, r in df.iterrows()]
    seg_times_js = str(seg_times)

    st.divider()

    vid_col, info_col = st.columns([1, 3])

    with vid_col:
        if os.path.exists(VIDEO_PATH):
            import base64
            with open(VIDEO_PATH, "rb") as vf:
                vid_b64 = base64.b64encode(vf.read()).decode()

            # Player HTML5 custom avec sync segment → selectbox via query params
            player_html = f"""
            <video id="vp" controls style="width:100%;border-radius:10px;background:#000;max-height:300px"
                   src="data:video/mp4;base64,{vid_b64}"
                   currentTime="{int(row['start_s'])}">
            </video>
            <script>
            const segs = {seg_times_js};
            const vp   = document.getElementById('vp');
            vp.currentTime = {int(row['start_s'])};
            let lastSeg = {seg_idx};
            vp.addEventListener('timeupdate', () => {{
                const t = vp.currentTime;
                for (let i = 0; i < segs.length; i++) {{
                    if (t >= segs[i][0] && t < segs[i][1]) {{
                        if (i !== lastSeg) {{
                            lastSeg = i;
                            // Met à jour l'URL query param → Streamlit relit
                            const url = new URL(window.parent.location.href);
                            url.searchParams.set('seg', i);
                            window.parent.history.replaceState(null, '', url.toString());
                            // Force le rechargement Streamlit
                            window.parent.postMessage({{type:'streamlit:setComponentValue', value: i}}, '*');
                        }}
                        break;
                    }}
                }}
            }});
            </script>
            """
            import streamlit.components.v1 as components
            components.html(player_html, height=320)
            st.caption(f"▶ Seg.{row['ID']:02d} · {row['Début']} → {row['Fin']}")
        else:
            st.warning("Vidéo introuvable")

    with info_col:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap">
          <span style="font-size:1.05rem;font-weight:800;color:{cb}">
            Seg.{row['ID']:02d} &nbsp; {row['Début']} → {row['Fin']}
          </span>
          <span style="background:{cb};color:white;border-radius:20px;
                       padding:3px 14px;font-size:.8rem;font-weight:700">
            {row[col_q_v]}
          </span>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("WER",  f"{row[col_wer_v]}%")
        m2.metric("CER",  f"{row[col_cer_v]}%")
        m3.metric("BLEU", f"{row[col_bl_v]}/100")

        st.divider()

        st.markdown(f"""
        <div style="background:#f0fdf4;border-radius:8px;padding:10px 14px;margin-bottom:8px">
          <div style="color:#16a34a;font-weight:700;font-size:.78rem;margin-bottom:4px">📝 RÉFÉRENCE</div>
          <div style="color:#166534;font-size:.9rem;line-height:1.6">{row['Référence']}</div>
        </div>""", unsafe_allow_html=True)

        h = errors_html(row["Référence"], stt_txt) if stt_txt else "<i style='color:#94a3b8'>(vide)</i>"
        st.markdown(f"""
        <div style="background:#fffbeb;border-radius:8px;padding:10px 14px">
          <div style="color:#92400e;font-weight:700;font-size:.78rem;margin-bottom:4px">🤖 {src_v.upper()}</div>
          <div style="font-size:.9rem;line-height:1.9">{h}</div>
        </div>""", unsafe_allow_html=True)

    # Timeline
    st.divider()
    st.caption("Timeline des segments")
    fig_tl = go.Figure()
    for _, r in df.iterrows():
        fig_tl.add_trace(go.Bar(
            x=[r["end_s"]-r["start_s"]], y=[""], base=[r["start_s"]],
            orientation="h", marker_color=COLORS_Q[r[col_q_v]],
            opacity=0.85, width=0.4, showlegend=False,
            hovertemplate=(f"<b>Seg.{r['ID']:02d}</b><br>"
                           f"{r['Début']} → {r['Fin']}<br>"
                           f"WER {r[col_wer_v]}% · {r[col_q_v]}<extra></extra>"),
        ))
    fig_tl.add_vline(x=row["start_s"], line_dash="dash", line_color="#1e3a5f",
                     line_width=2, annotation_text=f"▶ {row['Début']}",
                     annotation_position="top")
    fig_tl.update_layout(
        height=80, paper_bgcolor="white", plot_bgcolor="#f8fafc",
        margin=dict(l=10,r=10,t=20,b=10), barmode="overlay",
        xaxis=dict(title="", range=[0, df["end_s"].max()+1]),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    lc = st.columns(4)
    for i, (q, c) in enumerate(COLORS_Q.items()):
        with lc[i]:
            st.markdown(f"<span style='background:{c};color:white;border-radius:4px;"
                        f"padding:2px 10px;font-size:.76rem;font-weight:600'>{q}</span>",
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — WILDMOKA
# ══════════════════════════════════════════════════════════════════
with tab3:

    st.subheader("🎬 Wildmoka — Clips à corriger")

    with st.expander("ℹ️ C'est quoi Wildmoka ?"):
        st.markdown("""
**Wildmoka** est une plateforme de clipping et diffusion média utilisée par **France 24, Arte, BFM TV**.

SubQuality génère les **timestamps des segments WER ≥ 25%** au format JSON Wildmoka.
Un éditeur peut ouvrir le clip au bon timestamp, corriger le sous-titre, puis diffuser.

**Seuil retenu :** WER normalisé ≥ 25%
        """)

    bad_df = df[df["WER norm W"] >= 25].copy()
    bad_df["Priorité"] = bad_df["WER norm W"].apply(lambda w: "HAUTE" if w>=40 else "MOYENNE")
    high   = int((bad_df["Priorité"]=="HAUTE").sum())
    medium = int((bad_df["Priorité"]=="MOYENNE").sum())

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl, color in [
        (k1, str(nbc),   "Clips à réviser",  "#ef4444"),
        (k2, str(high),  "Priorité HAUTE",   "#ef4444"),
        (k3, str(medium),"Priorité MOYENNE", "#f97316"),
        (k4, str(gb),    "BLEU Whisper",     "#0ea5e9"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi" style="border-top-color:{color}">
              <div class="kpi-val" style="color:{color}">{val}</div>
              <p class="kpi-lbl">{lbl}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    prio = st.radio("Filtrer :", ["Tous","HAUTE (≥40%)","MOYENNE (25-40%)"], horizontal=True)
    clips = bad_df.copy()
    if prio == "HAUTE (≥40%)":        clips = clips[clips["Priorité"]=="HAUTE"]
    elif prio == "MOYENNE (25-40%)":  clips = clips[clips["Priorité"]=="MOYENNE"]

    st.markdown("<br>", unsafe_allow_html=True)

    for _, cr in clips.reset_index(drop=True).iterrows():
        color     = "#ef4444" if cr["Priorité"]=="HAUTE" else "#f97316"
        wer_pct   = min(int(cr["WER norm W"]), 100)
        stt_h     = errors_html(cr["Référence"], cr["STT Whisper"])



        with st.container():
            st.markdown(f"""
            <div class="wm-card" style="border-left-color:{color}">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px">
                <span style="font-weight:800;font-size:1rem;color:{color}">
                  🎬 F24_SEG_{cr['ID']:02d}
                </span>
                <span style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
                  <span class="tag" style="background:{color};color:white">{cr['Priorité']}</span>
                  <span class="tag" style="background:#f1f5f9;color:#475569">
                    ⏱ {cr['Début']} → {cr['Fin']}
                  </span>
                </span>
              </div>
              <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;
                            font-size:.74rem;color:#64748b;margin-bottom:3px">
                  <span>WER normalisé</span>
                  <span style="font-weight:700;color:{color}">{cr['WER norm W']}%</span>
                </div>
                <div style="background:#f1f5f9;border-radius:6px;height:6px;overflow:hidden">
                  <div style="width:{wer_pct}%;height:100%;background:{color};border-radius:6px"></div>
                </div>
              </div>
              <div style="display:flex;gap:20px;margin-bottom:10px;font-size:.8rem">
                <span>📊 <b>CER</b> <span style="color:{color}">{cr['CER W']}%</span></span>
                <span>✨ <b>BLEU</b> <span style="color:#10b981">{cr['BLEU W']}/100</span></span>
              </div>
              <div style="background:#f0fdf4;border-radius:6px;padding:8px 10px;margin-bottom:6px;font-size:.84rem">
                <span style="color:#16a34a;font-weight:700">REF</span>
                <span style="color:#166534;margin-left:8px">{cr['Référence']}</span>
              </div>
              <div style="background:#fffbeb;border-radius:6px;padding:8px 10px;font-size:.84rem;line-height:1.8">
                <span style="color:#92400e;font-weight:700">STT</span>
                <span style="margin-left:8px">{stt_h}</span>
              </div>
              <p style="color:#94a3b8;font-size:.71rem;margin-top:8px;margin-bottom:0">
                🏷️ subtitle_quality_alert &nbsp;·&nbsp; REVIEW_REQUIRED
              </p>
            </div>""", unsafe_allow_html=True)


    # Export JSON
    st.divider()
    wm_json = {
        "source": "France 24 — JT Épisode Neige",
        "pipeline": "SubQuality v1.0",
        "model_stt": "openai/whisper-base",
        "metriques": {"wer_norm_whisper": gw, "cer_whisper": gc, "bleu_whisper": gb},
        "total_segments": n,
        "clips_to_review": nbc,
        "clips": [{
            "clip_id":      f"F24_SEG_{r['ID']:02d}",
            "timestamp_in": r["Début"], "timestamp_out": r["Fin"],
            "start_s": r["start_s"], "end_s": r["end_s"],
            "wer_norm": r["WER norm W"], "cer": r["CER W"], "bleu": r["BLEU W"],
            "priority": "HIGH" if r["WER norm W"]>=40 else "MEDIUM",
            "action": "REVIEW_REQUIRED",
            "wildmoka_tag": "subtitle_quality_alert",
            "reference": r["Référence"], "stt_output": r["STT Whisper"],
        } for _, r in bad_df.iterrows()]
    }
    col_dl, _ = st.columns([1, 2])
    with col_dl:
        st.download_button("⬇️ Télécharger wildmoka_clips.json",
                           data=json.dumps(wm_json, ensure_ascii=False, indent=2),
                           file_name="wildmoka_clips.json", mime="application/json")
    with st.expander("Aperçu JSON"):
        st.json(wm_json)
