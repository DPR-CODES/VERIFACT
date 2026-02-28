"""
app.py — VERIFACT Frontend (All-in-One Streamlit)
Cyberpunk intelligence-terminal aesthetic.
Run: streamlit run app.py
"""

import os
import time
import requests
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Support both local development and remote deployment
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 8001))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")

st.set_page_config(
    page_title="VERIFACT",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "bg":      "#060912",
    "surface": "#0d1117",
    "panel":   "#111827",
    "border":  "#1f2d3d",
    "cyan":    "#00e5ff",
    "green":   "#00ff88",
    "red":     "#ff2d55",
    "amber":   "#ffa500",
    "purple":  "#b060ff",
    "muted":   "#4a5568",
    "text":    "#e2eaf5",
    "subtext": "#8899aa",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #060912 !important;
    color: #e2eaf5 !important;
    font-family: 'Exo 2', sans-serif !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060912; }
::-webkit-scrollbar-thumb { background: #00e5ff33; border-radius: 2px; }

[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1f2d3d !important;
}
[data-testid="stSidebar"] * { color: #e2eaf5 !important; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

h1, h2, h3, h4 { font-family: 'Exo 2', sans-serif !important; color: #e2eaf5 !important; }
label, p, div { color: #e2eaf5 !important; }

.stTextArea textarea {
    background: #111827 !important;
    color: #e2eaf5 !important;
    border: 1px solid #1f2d3d !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: #00e5ff88 !important;
    box-shadow: 0 0 0 3px #00e5ff18 !important;
    outline: none !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid #00e5ff88 !important;
    color: #00e5ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.1rem !important;
    border-radius: 4px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #00e5ff18 !important;
    border-color: #00e5ff !important;
    box-shadow: 0 0 16px #00e5ff44 !important;
}
.stButton > button:disabled {
    border-color: #4a5568 !important;
    color: #4a5568 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-bottom: 1px solid #1f2d3d !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8899aa !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.65rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom-color: #00e5ff !important;
    background: #00e5ff0a !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #060912 !important;
    padding: 1.5rem 0 !important;
}

.stNumberInput input, .stTextInput input {
    background: #111827 !important;
    color: #e2eaf5 !important;
    border-color: #1f2d3d !important;
    font-family: 'Share Tech Mono', monospace !important;
}

hr { border-color: #1f2d3d !important; }
.stAlert { background: #111827 !important; border-color: #1f2d3d !important; }

body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,229,255,0.010) 2px,
        rgba(0,229,255,0.010) 4px
    );
    pointer-events: none;
    z-index: 9999;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ── HTML BUILDERS ────────────────────────────────────────────────────────────

def render_header():
    st.markdown(f"""
    <div style="text-align:center; padding:2.2rem 0 1.5rem;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                    color:{C['cyan']}88; letter-spacing:0.3em; text-transform:uppercase; margin-bottom:0.35rem;">
            INTELLIGENCE VERIFICATION SYSTEM · v2.0
        </div>
        <div style="font-family:'Exo 2',sans-serif; font-size:3.6rem; font-weight:900;
                    letter-spacing:0.04em; line-height:1;
                    background:linear-gradient(135deg,{C['cyan']},{C['purple']});
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    margin-bottom:0.4rem;">
            VERIFACT
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                    color:{C['subtext']}; letter-spacing:0.1em;">
            BERT · CLAIM EXTRACTION · MULTI-SOURCE EVIDENCE · AI VERDICT
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(data):
    label  = data.get("label","UNCERTAIN")
    conf   = data.get("confidence_percent",0)
    expln  = data.get("explanation","")
    fs     = data.get("final_score",0.5)
    bs     = data.get("bert_score",0.5)
    es     = data.get("evidence_score",0.5)
    method = data.get("bert_method","?")

    cfg = {
        "REAL":      {"color":C["green"], "bg":"#00ff881a","border":"#00ff8866","icon":"[ ✓ ]","glow":"#00ff8844"},
        "FAKE":      {"color":C["red"],   "bg":"#ff2d551a","border":"#ff2d5566","icon":"[ ✗ ]","glow":"#ff2d5544"},
        "UNCERTAIN": {"color":C["amber"], "bg":"#ffa5001a","border":"#ffa50066","icon":"[ ? ]","glow":"#ffa50044"},
    }
    c = cfg.get(label, cfg["UNCERTAIN"])
    col = c["color"]

    st.markdown(f"""
    <div style="background:{c['bg']};border:1px solid {c['border']};border-radius:8px;
                padding:1.6rem 2rem;margin:0.8rem 0;box-shadow:0 0 40px {c['glow']};">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1.5rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:200px;">
          <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.6rem;">
            <span style="font-size:1.6rem;color:{col};font-weight:900;
                         font-family:'Share Tech Mono',monospace;
                         text-shadow:0 0 20px {col}88;">{c['icon']}</span>
            <div>
              <div style="font-size:1.6rem;font-weight:900;color:{col};
                          letter-spacing:0.1em;font-family:'Exo 2',sans-serif;">{label}</div>
              <div style="font-size:0.68rem;color:{C['subtext']};
                          font-family:'Share Tech Mono',monospace;letter-spacing:0.04em;">
                ML: {method.upper()} · BERT 60% · EVIDENCE 40%
              </div>
            </div>
          </div>
          <div style="font-size:0.87rem;color:{C['text']}cc;line-height:1.65;
                      border-left:2px solid {col}55;padding-left:0.75rem;">{expln}</div>
        </div>
        <div style="min-width:210px;">
          <div style="text-align:center;margin-bottom:0.9rem;">
            <div style="font-size:2.8rem;font-weight:900;color:{col};
                        font-family:'Exo 2',sans-serif;text-shadow:0 0 25px {col};">{conf}%</div>
            <div style="font-size:0.65rem;color:{C['subtext']};font-family:'Share Tech Mono',monospace;
                        letter-spacing:0.12em;">OVERALL CONFIDENCE</div>
          </div>
          <div style="display:flex;flex-direction:column;gap:0.45rem;">
            {"".join([
              f'<div><div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
              f'<span style="font-size:0.65rem;color:{C["subtext"]};font-family:\'Share Tech Mono\',monospace;">{lbl}</span>'
              f'<span style="font-size:0.65rem;color:{bar_c};font-family:\'Share Tech Mono\',monospace;">{pct}%</span>'
              f'</div><div style="height:4px;background:{C["border"]};border-radius:2px;">'
              f'<div style="width:{pct}%;height:100%;background:{bar_c};border-radius:2px;'
              f'box-shadow:0 0 6px {bar_c};"></div></div></div>'
              for lbl, pct, bar_c in [
                ("FINAL",    int(fs*100), col),
                ("BERT",     int(bs*100), C["cyan"]),
                ("EVIDENCE", int(es*100), C["purple"]),
              ]
            ])}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats(data):
    items = [
        ("CLAIMS",   str(data.get("claims_checked",0)),              C["cyan"]),
        ("ARTICLES", str(data.get("articles_found",0)),              C["purple"]),
        ("SOURCES",  str(len(data.get("evidence_summary",[]))),      C["green"]),
        ("TIME",     f"{data.get('processing_time_ms',0)}ms",        C["amber"]),
    ]
    cols = st.columns(4)
    for col, (label, val, color) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div style="background:{C['panel']};border:1px solid {C['border']};
                        border-top:2px solid {color};border-radius:6px;
                        padding:0.85rem 1rem;text-align:center;">
              <div style="font-size:1.55rem;font-weight:800;color:{color};
                          font-family:'Exo 2',sans-serif;">{val}</div>
              <div style="font-size:0.6rem;color:{C['subtext']};
                          font-family:'Share Tech Mono',monospace;
                          letter-spacing:0.12em;margin-top:2px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_claim(i, claim):
    st.markdown(f"""
    <div style="background:{C['panel']};border:1px solid {C['border']};
                border-left:3px solid {C['cyan']};border-radius:0 6px 6px 0;
                padding:0.85rem 1.1rem;margin-bottom:0.5rem;">
      <div style="display:flex;gap:0.8rem;align-items:flex-start;">
        <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                     color:{C['cyan']};background:{C['cyan']}18;padding:2px 7px;
                     border-radius:3px;min-width:28px;text-align:center;
                     flex-shrink:0;margin-top:2px;">{i:02d}</span>
        <span style="font-size:0.91rem;color:{C['text']};line-height:1.55;">{claim}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def cred_badge(w):
    if w >= 0.95: return f'<span style="color:{C["green"]};font-size:0.7rem;font-family:\'Share Tech Mono\',monospace;">◆ HIGHLY CREDIBLE</span>'
    if w >= 0.80: return f'<span style="color:{C["cyan"]};font-size:0.7rem;font-family:\'Share Tech Mono\',monospace;">◆ TRUSTED</span>'
    if w >= 0.60: return f'<span style="color:{C["purple"]};font-size:0.7rem;font-family:\'Share Tech Mono\',monospace;">◆ MODERATE</span>'
    return f'<span style="color:{C["amber"]};font-size:0.7rem;font-family:\'Share Tech Mono\',monospace;">◆ LOW TRUST</span>'


def render_evidence(i, e):
    w   = e.get("source_weight",0.5)
    sim = int(e.get("similarity",0)*100)
    snip = (e.get("snippet") or e.get("description",""))[:150]
    sc  = C["green"] if sim >= 60 else C["amber"] if sim >= 30 else C["red"]
    st.markdown(f"""
    <div style="background:{C['panel']};border:1px solid {C['border']};
                border-radius:6px;padding:0.95rem 1.15rem;margin-bottom:0.75rem;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:160px;">
          <a href="{e.get('url','#')}" target="_blank"
             style="color:{C['cyan']};text-decoration:none;font-size:0.9rem;
                    font-weight:600;line-height:1.4;display:block;margin-bottom:0.25rem;">
            {i}. {(e.get('title','Untitled'))[:85]}
          </a>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                      color:{C['subtext']};margin-bottom:0.35rem;">
            {e.get('source','Unknown').upper()}
          </div>
          <div style="font-size:0.8rem;color:{C['subtext']};line-height:1.5;">{snip}</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.35rem;flex-shrink:0;">
          {cred_badge(w)}
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;
                      color:{sc};background:{sc}15;padding:2px 8px;
                      border:1px solid {sc}44;border-radius:3px;">MATCH {sim}%</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:{C['subtext']};">
            CRED {int(w*100)}%
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_backend_status(ok, msg):
    if ok:
        st.markdown(f"""
        <div style="background:#00ff8812;border:1px solid #00ff8844;border-radius:5px;
                    padding:0.55rem 0.85rem;margin-bottom:0.5rem;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.73rem;
                      color:{C['green']};letter-spacing:0.04em;">● BACKEND ONLINE</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                      color:{C['subtext']};margin-top:2px;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#ff2d5512;border:1px solid #ff2d5544;border-radius:5px;
                    padding:0.55rem 0.85rem;margin-bottom:0.5rem;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.73rem;
                      color:{C['red']};letter-spacing:0.04em;">● BACKEND OFFLINE</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                      color:{C['subtext']};margin-top:2px;">{msg}</div>
        </div>
        <div style="background:{C['panel']};border:1px solid {C['border']};border-radius:5px;
                    padding:0.65rem 0.85rem;font-size:0.72rem;color:{C['subtext']};
                    font-family:'Share Tech Mono',monospace;line-height:1.9;">
          TO START:<br>
          <span style="color:{C['cyan']};">› python main.py</span><br><br>
          WAIT FOR:<br>
          <span style="color:{C['green']};">✅ All components ready</span>
        </div>
        """, unsafe_allow_html=True)


# ── CHARTS ────────────────────────────────────────────────────────────────────

def _bg_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    return fig, ax


def chart_gauge(score, label):
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(theta_bg, [1]*200, lw=14, color=C["border"], solid_capstyle="round")
    fill_c = C["green"] if label == "REAL" else C["red"] if label == "FAKE" else C["amber"]
    end    = np.pi - score * np.pi
    theta_f = np.linspace(np.pi, end, 200)
    ax.plot(theta_f, [1]*200, lw=14, color=fill_c, solid_capstyle="round", alpha=0.9)
    ax.plot(theta_f, [1]*200, lw=20, color=fill_c, solid_capstyle="round", alpha=0.12)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    ax.text(np.pi/2, 0.32, f"{int(score*100)}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color=fill_c, fontfamily="monospace")
    ax.text(np.pi/2, 0.02, label, ha="center", va="center",
            fontsize=9, color=C["subtext"], fontfamily="monospace")
    plt.tight_layout(pad=0.1)
    return fig


def chart_breakdown(bs, es, fs):
    import numpy as np
    fig, ax = _bg_fig(5.5, 3.2)
    labels = ["BERT Score", "Evidence Score", "Final Score"]
    values = [bs, es, fs]
    colors = [C["cyan"], C["purple"], C["green"]]
    bars   = ax.barh(labels, values, color=colors, alpha=0.85, height=0.4)
    for bar, val in zip(bars, values):
        ax.text(min(val+0.02, 1.0), bar.get_y()+bar.get_height()/2,
                f"{int(val*100)}%", va="center", ha="left",
                color=C["text"], fontsize=9, fontfamily="monospace", fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.axvline(0.65, color=C["green"], lw=0.8, linestyle="--", alpha=0.45)
    ax.axvline(0.40, color=C["red"],   lw=0.8, linestyle="--", alpha=0.45)
    ax.tick_params(axis="x", colors=C["muted"], labelsize=7.5)
    ax.set_yticklabels(labels, fontfamily="monospace", color=C["subtext"], fontsize=8)
    ax.set_xlabel("Score (0–1)", color=C["subtext"], fontsize=8, fontfamily="monospace")
    for sp in ax.spines.values(): sp.set_edgecolor(C["border"])
    ax.grid(axis="x", color=C["border"], linestyle="--", alpha=0.35)
    ax.set_title("SCORE BREAKDOWN", color=C["subtext"], fontsize=8.5, fontfamily="monospace", pad=6)
    plt.tight_layout()
    return fig


def chart_source_pie(evidence):
    import numpy as np
    fig, ax = _bg_fig(4.8, 3.8)
    if not evidence:
        ax.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                color=C["muted"], fontsize=9, transform=ax.transAxes, fontfamily="monospace")
        ax.axis("off")
        return fig
    sources = {}
    for e in evidence:
        s = e.get("source","Unknown")
        sources[s] = sources.get(s,0) + 1
    palette = [C["cyan"],C["purple"],C["green"],C["amber"],C["red"],"#ff6b6b","#4ecdc4"]
    wedges, texts, autos = ax.pie(
        sources.values(), labels=None, autopct="%1.0f%%",
        colors=palette[:len(sources)], startangle=90,
        wedgeprops={"edgecolor":C["bg"],"linewidth":2}, pctdistance=0.75,
    )
    for at in autos: at.set(color=C["bg"], fontsize=8, fontweight="bold")
    patches = [mpatches.Patch(color=palette[i], label=k) for i,k in enumerate(sources)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5,-0.12),
              ncol=min(len(sources),3), framealpha=0, labelcolor=C["subtext"], fontsize=7.5)
    ax.set_title("EVIDENCE SOURCES", color=C["subtext"], fontsize=8.5, fontfamily="monospace", pad=6)
    plt.tight_layout()
    return fig


def chart_credibility(evidence):
    import numpy as np
    fig, ax = _bg_fig(6.5, 3.8)
    if not evidence:
        ax.axis("off")
        return fig
    top    = sorted(evidence, key=lambda x: x.get("source_weight",0), reverse=True)[:6]
    labels = [e.get("source","?")[:16] for e in top]
    cred   = [e.get("source_weight",0) for e in top]
    match  = [e.get("similarity",0) for e in top]
    x = np.arange(len(labels))
    w = 0.37
    ax.bar(x-w/2, cred,  w, color=C["cyan"],   alpha=0.85, label="Credibility")
    ax.bar(x+w/2, match, w, color=C["purple"], alpha=0.85, label="Match Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=7.5,
                        color=C["subtext"], fontfamily="monospace")
    ax.tick_params(axis="y", colors=C["muted"], labelsize=7.5)
    ax.axhline(0.7, color=C["green"], lw=0.7, linestyle="--", alpha=0.4)
    ax.grid(axis="y", color=C["border"], linestyle="--", alpha=0.35)
    for sp in ax.spines.values(): sp.set_edgecolor(C["border"])
    ax.legend(labelcolor=C["subtext"], facecolor=C["panel"],
              edgecolor=C["border"], fontsize=7.5)
    ax.set_title("CREDIBILITY vs MATCH SCORE", color=C["subtext"],
                 fontsize=8.5, fontfamily="monospace", pad=6)
    plt.tight_layout()
    return fig


# ── BACKEND HELPERS ───────────────────────────────────────────────────────────

def get_base():
    # Use API_URL if set (for production), otherwise build from host/port (for local dev)
    if API_URL and "localhost" not in API_URL and "127.0.0.1" not in API_URL:
        return API_URL
    return f"http://{st.session_state.get('api_host', API_HOST)}:{st.session_state.get('api_port', API_PORT)}"


def check_health():
    try:
        r = requests.get(f"{get_base()}/health", timeout=4)
        if r.status_code == 200:
            d = r.json()
            m = d.get("components",{}).get("bert_method","?")
            b = "✓" if d.get("components",{}).get("bert") else "⚠"
            return True, f"BERT {b} [{m.upper()}] · ALL SYSTEMS GO"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "CONNECTION REFUSED — RUN: python main.py"
    except requests.exceptions.Timeout:
        return False, "HEALTH CHECK TIMEOUT"


def call_analyze(text, max_retries=3):
    for attempt in range(1, max_retries+1):
        try:
            r = requests.post(f"{get_base()}/analyze", json={"text": text}, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                wait = attempt * 2.0
                st.warning(f"⚠ Attempt {attempt}/{max_retries} — retrying in {wait:.0f}s…")
                time.sleep(wait)
            else:
                raise ConnectionError(f"Backend unreachable after {max_retries} attempts. Run: python main.py") from e
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                st.warning(f"⚠ Timeout on attempt {attempt}/{max_retries}, retrying…")
                time.sleep(2)
            else:
                raise TimeoutError("Backend timed out. Models may be loading. Try again or check backend logs.")
        except requests.exceptions.HTTPError as e:
            raise e


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    for k, v in [("api_host", API_HOST), ("api_port", API_PORT),
                 ("text_input",""), ("last_result", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    render_header()

    # SIDEBAR
    with st.sidebar:
        st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:{C["cyan"]};letter-spacing:0.2em;padding:0.4rem 0 0.7rem;">▸ BACKEND CONFIG</div>', unsafe_allow_html=True)
        # Only show HOST/PORT inputs in local development mode
        if "localhost" in API_URL or "127.0.0.1" in API_URL:
            nh = st.text_input("HOST", value=st.session_state["api_host"])
            np_ = st.number_input("PORT", value=st.session_state["api_port"], min_value=1, max_value=65535, step=1)
            if nh != st.session_state["api_host"]:   st.session_state["api_host"] = nh
            if int(np_) != st.session_state["api_port"]: st.session_state["api_port"] = int(np_)
        else:
            st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:{C["green"]};padding:0.5rem;border-left:2px solid {C["green"]};">✓ Connected to{API_URL.replace("https://","").replace("http://","")}</div>', unsafe_allow_html=True)

        is_ok, hmsg = check_health()
        render_backend_status(is_ok, hmsg)
        if st.button("↻  RECHECK"):
            st.rerun()

        st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:{C["cyan"]};letter-spacing:0.2em;padding:1rem 0 0.6rem;">▸ EXAMPLES</div>', unsafe_allow_html=True)
        for ex in [
            "NASA confirmed evidence of liquid water on Mars subsurface.",
            "Quantum computers can now solve any problem in seconds.",
            "WHO declared major breakthrough in malaria treatment.",
            "5G towers cause widespread health damage per researchers.",
            "Vaccines contain microchips designed to track individuals.",
        ]:
            if st.button((ex[:52]+"…") if len(ex)>52 else ex, key=f"ex_{ex[:15]}"):
                st.session_state["text_input"] = ex
                st.rerun()

        st.markdown(f"""
        <div style="margin-top:1.2rem;padding-top:0.9rem;border-top:1px solid {C['border']};
                    font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                    color:{C['muted']};line-height:2;">
          REAL  ·  SCORE ≥ 65%<br>
          UNCERTAIN  ·  40–65%<br>
          FAKE  ·  SCORE ≤ 40%<br><br>
          BERT WEIGHT  ·  60%<br>
          EVIDENCE  ·  40%
        </div>
        """, unsafe_allow_html=True)

    # MAIN INPUT
    st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:{C["subtext"]};letter-spacing:0.1em;margin-bottom:0.3rem;">INPUT · PASTE ARTICLE OR CLAIM TO ANALYSE</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "input", value=st.session_state.get("text_input",""), height=175,
        placeholder="Paste a news article, headline, or factual claim…",
        label_visibility="collapsed",
    )

    btn_label = "◈  ANALYSE CLAIM" if is_ok else "◈  ANALYSE CLAIM  [ BACKEND OFFLINE ]"
    clicked   = st.button(btn_label, use_container_width=True, disabled=not is_ok)

    if not is_ok:
        st.markdown(f"""
        <div style="background:{C['panel']};border:1px solid {C['border']}40;
                    border-radius:5px;padding:0.7rem 1rem;margin-top:0.4rem;text-align:center;
                    font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:{C['subtext']};">
          START BACKEND TO ENABLE ANALYSIS  ·  <span style="color:{C['cyan']};">python main.py</span>
        </div>
        """, unsafe_allow_html=True)

    if clicked:
        if not text_input or len(text_input.strip()) < 3:
            st.error("⚠ Input too short — enter at least 3 characters.")
        else:
            with st.spinner("SCANNING · EXTRACTING CLAIMS · VERIFYING EVIDENCE…"):
                try:
                    data = call_analyze(text_input.strip(), max_retries=3)
                    st.session_state["last_result"] = data
                except ConnectionError as e:
                    st.error(f"CONNECTION FAILED: {e}")
                    st.markdown("**Steps:** Open a terminal → `python main.py` → wait for `✅ All components ready` → click **↻ RECHECK** → retry.")
                    st.stop()
                except TimeoutError as e:
                    st.error(f"TIMEOUT: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"ERROR: {e}")
                    st.stop()

    # RESULTS
    data = st.session_state.get("last_result")
    if data:
        render_verdict(data)
        st.markdown("<br>", unsafe_allow_html=True)
        render_stats(data)
        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["◈  DASHBOARD", "◉  CLAIMS", "◎  EVIDENCE", "◍  SOURCES"])

        with tab1:
            c1, c2 = st.columns([1, 1.3])
            with c1: st.pyplot(chart_gauge(data["final_score"], data["label"]))
            with c2: st.pyplot(chart_breakdown(data.get("bert_score",0), data.get("evidence_score",0), data.get("final_score",0)))
            st.markdown("<br>", unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3: st.pyplot(chart_source_pie(data.get("evidence_summary",[])))
            with c4: st.pyplot(chart_credibility(data.get("evidence_summary",[])))

        with tab2:
            claims = data.get("claims_extracted",[])
            st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:{C["subtext"]};margin-bottom:0.9rem;letter-spacing:0.08em;">{len(claims)} CLAIM{"S" if len(claims)!=1 else ""} EXTRACTED</div>', unsafe_allow_html=True)
            if claims:
                for i, c in enumerate(claims, 1): render_claim(i, c)
            else:
                st.markdown(f'<div style="text-align:center;padding:2rem;color:{C["muted"]};font-family:\'Share Tech Mono\',monospace;font-size:0.82rem;">NO CLAIMS EXTRACTED</div>', unsafe_allow_html=True)

        with tab3:
            evs = data.get("evidence_summary",[])
            st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:{C["subtext"]};margin-bottom:0.9rem;letter-spacing:0.08em;">{len(evs)} SOURCE{"S" if len(evs)!=1 else ""} RETRIEVED</div>', unsafe_allow_html=True)
            if evs:
                for i, e in enumerate(evs, 1): render_evidence(i, e)
            else:
                st.markdown(f'<div style="text-align:center;padding:2rem;color:{C["muted"]};font-family:\'Share Tech Mono\',monospace;font-size:0.82rem;">NO EVIDENCE FOUND</div>', unsafe_allow_html=True)

        with tab4:
            evs = data.get("evidence_summary",[])
            seen, uniq = set(), []
            for e in evs:
                if e.get("source","?") not in seen:
                    seen.add(e.get("source","?"))
                    uniq.append(e)
            if uniq:
                for i, e in enumerate(uniq, 1):
                    w  = e.get("source_weight",0.5)
                    bw = int(w*100)
                    bc = C["green"] if w>=0.9 else C["cyan"] if w>=0.7 else C["amber"]
                    stars = "⭐"*max(1,round(w*5))
                    st.markdown(f"""
                    <div style="background:{C['panel']};border:1px solid {C['border']};
                                border-radius:6px;padding:0.95rem 1.15rem;margin-bottom:0.65rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.6rem;">
                        <div>
                          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:{C['subtext']};letter-spacing:0.08em;">SOURCE {i:02d}</div>
                          <div style="font-size:0.92rem;font-weight:700;color:{C['text']};">{e.get('source','?')}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:0.82rem;color:{bc};margin-bottom:3px;">{stars}</div>
                          <a href="{e.get('url','#')}" target="_blank" style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:{C['cyan']};text-decoration:none;">↗ READ ARTICLE</a>
                        </div>
                      </div>
                      <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:{C['subtext']};margin-bottom:4px;">CREDIBILITY · {bw}%</div>
                      <div style="height:5px;background:{C['border']};border-radius:3px;">
                        <div style="width:{bw}%;height:100%;background:{bc};border-radius:3px;box-shadow:0 0 8px {bc}66;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:center;padding:2rem;color:{C["muted"]};font-family:\'Share Tech Mono\',monospace;font-size:0.82rem;">NO SOURCE DATA</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:2.5rem;padding:1.2rem 0;
                border-top:1px solid {C['border']};
                font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                color:{C['muted']};letter-spacing:0.1em;line-height:2;">
      VERIFACT · INTELLIGENCE VERIFICATION SYSTEM · v2.0<br>
      ⚠ FOR EDUCATIONAL PURPOSES ONLY · ALWAYS VERIFY FROM PRIMARY SOURCES
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
