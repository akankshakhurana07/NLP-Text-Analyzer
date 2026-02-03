import streamlit as st
matplotlib.use("Agg")
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from string import punctuation
from heapq import nlargest
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="NLP Text Analyzer", page_icon="🌍", layout="wide")

# ── CSS ──
st.markdown("""<style>
:root { --bg:#0e0f11; --card:#161820; --card2:#1c1e28; --gold:#c8a96e; --txt:#e8e6e0; --mute:#6b6b73; --bdr:#2a2c35; }
.stApp { background:var(--bg) !important; color:var(--txt) !important; font-family:'Segoe UI',sans-serif; }
#MainMenu, footer, .stToolbar { visibility:hidden; }
.block-container { padding-top:1.5rem !important; max-width:1300px !important; }
.stSidebar { background:#111214 !important; border-right:1px solid var(--bdr); }

.card { background:var(--card); border:1px solid var(--bdr); border-radius:12px; padding:1.1rem 1.2rem; margin-bottom:12px; }

.stButton button { background:var(--gold) !important; color:#0e0f11 !important; font-weight:600 !important; border:none !important; border-radius:8px !important; cursor:pointer; transition:background .15s, transform .1s, box-shadow .15s; }
.stButton button:hover { background:#dbbe7e !important; box-shadow:0 0 14px rgba(200,169,110,.5); }
.stButton button:active { transform:scale(0.93); background:#a88a55 !important; }

.stTextarea textarea, .stTextInput input { background:var(--card2) !important; color:var(--txt) !important; border:1px solid var(--bdr) !important; border-radius:8px !important; }
.stTextarea label, .stSlider label, .stSelectbox label { color:var(--mute) !important; font-size:.78rem !important; }

.stTabs button[role="tab"] { color:var(--mute) !important; font-size:.82rem !important; }
.stTabs button[role="tab"][aria-selected="true"] { color:var(--gold) !important; border-bottom-color:var(--gold) !important; }

.token-tag { border-radius:8px; padding:4px 9px; display:inline-flex; flex-direction:column; gap:1px; margin:3px; }
.token-tag .token-word { font-size:.84rem; font-weight:600; }
.token-tag .token-meta { font-size:.64rem; opacity:.7; }
.pos-NOUN  { background:#2e3a52; color:#7eaee0; }
.pos-VERB  { background:#3a2e52; color:#b07ee0; }
.pos-ADJ   { background:#2e4a3a; color:#7ee0a8; }
.pos-ADV   { background:#4a3a2e; color:#e0b07e; }
.pos-OTHER { background:#2a2c35; color:#8a8a96; }

.summary-box { background:linear-gradient(90deg,rgba(200,169,110,.08),transparent); border-left:3px solid var(--gold); padding:9px 13px; border-radius:0 8px 8px 0; margin-bottom:7px; font-size:.87rem; line-height:1.5; }
.summary-box .rank-badge { background:var(--gold); color:#0e0f11; font-size:.62rem; font-weight:700; padding:1px 7px; border-radius:10px; margin-right:5px; }

.stats-row { display:flex; gap:10px; }
.stat-box { flex:1; background:var(--card2); border-radius:10px; padding:10px; text-align:center; }
.stat-box .stat-value { font-size:1.4rem; font-weight:700; color:var(--gold); }
.stat-box .stat-label { font-size:.67rem; color:var(--mute); text-transform:uppercase; letter-spacing:.4px; }

.lang-badge { display:inline-block; background:var(--card2); border:1px solid var(--bdr); border-radius:20px; padding:4px 14px; font-size:.78rem; color:var(--gold); font-weight:600; margin-bottom:10px; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:var(--bdr); border-radius:3px; }
</style>""", unsafe_allow_html=True)


# ── Sample Texts ──
EN_SAMPLE = ("There are broadly two types of extractive summarization tasks depending on what the "
"summarization program focuses on. The first is generic summarization, which focuses on obtaining a generic "
"summary or abstract of the collection (whether documents, or sets of images, or videos, news stories etc.). "
"The second is query relevant summarization, sometimes called query-based summarization, which summarizes "
"objects specific to a query. Summarization systems are able to create both query relevant text summaries and "
"generic machine-generated summaries depending on what the user needs. An example of a summarization problem "
"is document summarization, which attempts to automatically produce an abstract from a given document. "
"Sometimes one might be interested in generating a summary from a single source document, while others can "
"use multiple source documents (for example, a cluster of articles on the same topic). This problem is called "
"multi-document summarization. A related application is summarizing news articles. Imagine a system, which "
"automatically pulls together news articles on a given topic (from the web), and concisely represents the "
"latest news as a summary. Image collection summarization is another application example of automatic "
"summarization. It consists in selecting a representative set of images from a larger set of images. A summary "
"in this context is useful to show the most representative images of results in an image collection exploration "
"system. Video summarization is a related domain, where the system automatically creates a trailer of a long "
"video. This also has applications in consumer or personal videos, where one might want to skip the boring or "
"repetitive actions. Similarly, in surveillance videos, one would want to extract important and suspicious "
"activity, while ignoring all the boring and redundant frames captured.")

FR_SAMPLE = ("Il existe essentiellement deux types de tâches de résumé extractif en fonction de ce sur quoi "
"se concentre le programme de résumé. Le premier est le résumé générique, qui vise à obtenir un résumé "
"générique ou abstrait de la collection (qu'il s'agisse de documents, d'ensembles d'images, de vidéos, "
"d'articles de presse, etc.). Le second est le résumé pertinent pour une requête, parfois appelé résumé basé "
"sur une requête, qui résume des objets spécifiques à une requête. Les systèmes de résumé sont capables de "
"créer à la fois des résumés textuels pertinents pour une requête et des résumés générés automatiquement en "
"fonction des besoins de l'utilisateur. Un exemple de problème de résumé est le résumé de documents, qui tente "
"de produire automatiquement un résumé à partir d'un document donné. Parfois, on peut souhaiter générer un "
"résumé à partir d'un seul document source, tandis que d'autres peuvent utiliser plusieurs documents sources "
"(par exemple, un ensemble d'articles sur le même sujet). Ce problème est appelé résumé multi-documents. Une "
"application connexe est le résumé d'articles d'actualité. Imaginez un système qui rassemble automatiquement "
"des articles d'actualité sur un sujet donné (à partir du Web) et représente de manière concise les dernières "
"informations sous forme de résumé. Le résumé d'une collection d'images est un autre exemple d'application de "
"résumé automatique. Il consiste à sélectionner un ensemble représentatif d'images parmi un ensemble plus large "
"d'images. Dans ce contexte, un résumé est utile pour montrer les images les plus représentatives des résultats "
"dans un système d'exploration de collections d'images. La vidéo résumée est un domaine connexe, où le système "
"crée automatiquement une bande-annonce d'une longue vidéo. Cela trouve également des applications dans les "
"vidéos grand public ou personnelles, où l'on peut vouloir passer les actions ennuyeuses ou répétitives. De "
"même, dans les vidéos de surveillance, on souhaiterait extraire les activités importantes et suspectes, tout "
"en ignorant les images ennuyeuses et redondantes capturées.")


# ── Language Config ──
LANGS = {
    "🇬🇧 English": {"model":"en_core_web_sm", "stopwords":EN_STOP, "flag":"🇬🇧", "name":"English", "placeholder":"Paste your text …", "sample":EN_SAMPLE},
    "🇫🇷 French":  {"model":"fr_core_news_sm","stopwords":FR_STOP, "flag":"🇫🇷", "name":"French",  "placeholder":"Collez votre texte …","sample":FR_SAMPLE},
}

# POS tag → CSS class
POS_CLASS = {"NOUN":"pos-NOUN", "VERB":"pos-VERB", "ADJ":"pos-ADJ", "ADV":"pos-ADV"}

# Legend colors for Token Tags tab
LEGEND_ITEMS = [("NOUN","#7eaee0"),("VERB","#b07ee0"),("ADJ","#7ee0a8"),("ADV","#e0b07e"),("OTHER","#8a8a96")]


# ── Model Loader (auto-downloads if missing) ──
@st.cache_resource(show_spinner="Loading model …")
def load_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)


# ── Helper: Word Frequency ──
def calc_freq(doc, stopwords):
    count = {}
    for token in doc:
        word = token.text.lower()
        if word not in stopwords and word not in punctuation and word.strip():
            count[word] = count.get(word, 0) + 1
    mx = max(count.values()) if count else 1
    return {w: round(c / mx, 4) for w, c in count.items()}


# ── Helper: Extractive Summary ──
def calc_summary(doc, freq, ratio):
    sents, scores = list(doc.sents), {}
    for sent in sents:
        for token in sent:
            if token.text.lower() in freq:
                scores[sent] = scores.get(sent, 0) + freq[token.text.lower()]
    top = nlargest(max(1, int(len(sents) * ratio)), scores, key=scores.get)
    return top, scores


# ── Helper: Frequency Bar Chart ──
def plot_freq(freq, top_n):
    items  = sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    words  = [i[0] for i in items]
    values = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(10, max(3.5, top_n * 0.3)))
    fig.patch.set_facecolor("#161820")
    ax.set_facecolor("#161820")
    ax.barh(words[::-1], values[::-1], color="#c8a96e", edgecolor="none", height=0.6)
    ax.set_xlabel("Normalized Frequency", color="#6b6b73", fontsize=9)
    ax.tick_params(colors="#e8e6e0", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2a2c35")
    ax.spines["left"].set_color("#2a2c35")
    fig.tight_layout()
    return fig


# ── Helper: Word Cloud ──
def plot_wc(text):
    cloud = WordCloud(width=900, height=360, background_color="#161820", colormap="plasma", max_words=120).generate(text)
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#161820")
    ax.set_facecolor("#161820")
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 NLP Text Analyzer")
    st.caption("Powered by spaCy")
    st.divider()
    selected_lang  = st.selectbox("🌐 Select Language", options=list(LANGS.keys()))
    cfg            = LANGS[selected_lang]
    st.divider()
    summary_ratio  = st.slider("📊 Summary Ratio", 0.1, 0.9, 0.4, 0.05)
    top_n_words    = st.slider("🏷️ Top-N Words", 5, 40, 15)
    st.divider()
    st.markdown(f"**Model:** `{cfg['model']}`\n**Language:** {cfg['flag']} {cfg['name']}")


# ── Load Model ──
nlp       = load_model(cfg["model"])
stopwords = set(cfg["stopwords"])


# ── Header ──
st.markdown(f'<div style="background:linear-gradient(135deg,#1a1620,#0e0f11);border:1px solid #2a2c35;padding:1.6rem 2rem;border-radius:12px;margin-bottom:12px;"><h1 style="margin:0;font-size:1.8rem;color:#e8e6e0;">{cfg["flag"]} <span style="color:#c8a96e;">Text Analyzer</span></h1><p style="margin:4px 0 0;color:#6b6b73;font-size:.85rem;">Tokenization · POS Tagging · Lemmatization · Summarization · Word Cloud</p></div>', unsafe_allow_html=True)


# ── Input Section ──
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f'<div class="lang-badge">{cfg["flag"]} {cfg["name"]} Mode</div>', unsafe_allow_html=True)

input_text = st.text_area("Input", value=cfg["sample"], height=150, label_visibility="collapsed", placeholder=cfg["placeholder"], key=selected_lang)

col1, col2 = st.columns([1, 5])
analyze_clicked = col1.button("⚡ Analyze")
if col2.button("↺ Reset"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

if not input_text.strip():
    st.warning("Kuch text dalo please.")
    st.stop()


# ── Run NLP ──
with st.spinner("Analyzing …"):
    doc             = nlp(input_text)
    word_freq       = calc_freq(doc, stopwords)
    summary, scores = calc_summary(doc, word_freq, summary_ratio)
    all_tokens      = list(doc)
    all_sentences   = list(doc.sents)

# Analyze confirmation
if analyze_clicked:
    st.markdown('<div style="background:rgba(126,174,224,.1);border-left:3px solid #7eaee0;padding:8px 13px;border-radius:0 8px 8px 0;color:#7eaee0;font-size:.82rem;margin-bottom:8px;">✅ Analysis complete!</div>', unsafe_allow_html=True)


# ── Stats Row (loop se banao — no repeated HTML blocks) ──
stats = [("Tokens", len(all_tokens)), ("Sentences", len(all_sentences)), ("Unique Words", len(word_freq)), ("Summary Sents", len(summary))]
stats_html = '<div class="card" style="padding:.75rem 1rem;"><div class="stats-row">'
for label, value in stats:
    stats_html += f'<div class="stat-box"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'
stats_html += '</div></div>'
st.markdown(stats_html, unsafe_allow_html=True)


# ── Tabs ──
tab_tokens, tab_table, tab_summary, tab_freq, tab_cloud = st.tabs(["🏷️ Tokens","📋 Table","📝 Summary","📊 Frequency","☁️ Word Cloud"])


# ── TAB 1: Token Tags ──
with tab_tokens:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Legend (loop se banao)
    legend_html = '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">'
    for name, color in LEGEND_ITEMS:
        legend_html += f'<div style="display:flex;align-items:center;gap:6px;font-size:.73rem;color:#6b6b73;"><div style="width:10px;height:10px;border-radius:3px;background:{color};"></div>{name}</div>'
    legend_html += '</div>'

    # Token tags
    tags_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;">'
    for token in all_tokens:
        tags_html += f'<div class="token-tag {POS_CLASS.get(token.pos_,"pos-OTHER")}"><span class="token-word">{token.text}</span><span class="token-meta">{token.pos_} · {token.lemma_}</span></div>'
    tags_html += '</div>'

    st.markdown(legend_html + tags_html + '</div>', unsafe_allow_html=True)


# ── TAB 2: Table ──
with tab_table:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df = pd.DataFrame([{"#":i+1,"Token":t.text,"POS":t.pos_,"Tag":t.tag_,"Lemma":t.lemma_,"Dep":t.dep_,"Stop":t.is_stop,"Punct":t.is_punct} for i,t in enumerate(all_tokens)])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 3: Summary ──
with tab_summary:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Rank by score descending
    ranked   = sorted([(s, scores.get(s, 0)) for s in summary], key=lambda x: -x[1])
    rank_map = {id(s): r + 1 for r, (s, _) in enumerate(ranked)}

    for sent in summary:
        st.markdown(f'<div class="summary-box"><span class="rank-badge">#{rank_map[id(sent)]}</span>{sent.text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Scores table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("All Sentence Scores")
    df_scores = pd.DataFrame([{"#":i+1,"Sentence":(s.text[:80]+"…") if len(s.text)>80 else s.text,"Score":round(scores.get(s,0),3),"In Summary":s in summary} for i,s in enumerate(all_sentences)])
    st.dataframe(df_scores, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 4: Frequency ──
with tab_freq:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.pyplot(plot_freq(word_freq, top_n_words), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 5: Word Cloud ──
with tab_cloud:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.pyplot(plot_wc(input_text), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
