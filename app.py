"""
Voice Customer Support AI — Intent Router
Speech -> Text -> Preprocessing -> TF-IDF -> Logistic Regression -> Category + Confidence

Run with:
    streamlit run app.py
"""

import io
import joblib
import speech_recognition as sr
import streamlit as st

from src.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Load model, vectorizer, label encoder
# ---------------------------------------------------------------------------
model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

CATEGORY_INFO = {
    "ACCOUNT":      ("Account Team",     "Login, password, and profile issues"),
    "ORDER":        ("Order Desk",       "Placing, changing, cancelling, or tracking an order"),
    "REFUND":       ("Refunds Team",     "Refund requests, policy, and status"),
    "INVOICE":      ("Billing Team",     "Invoices and billing documents"),
    "CONTACT":      ("General Support",  "Reaching a human agent"),
    "PAYMENT":      ("Payments Team",    "Payment methods and payment issues"),
    "FEEDBACK":     ("Customer Care",    "Filing a formal complaint or review"),
    "DELIVERY":     ("Delivery Team",    "Delivery options and timing"),
    "SHIPPING":     ("Shipping Team",    "Shipping address setup or changes"),
    "SUBSCRIPTION": ("Subscriptions",    "Newsletter and subscription management"),
    "CANCEL":       ("Cancellations",    "Cancellation fees and termination penalties"),
}
INTENT_EMOJIS = {
    "cancel_order": "❌",
    "change_order": "✏️",
    "change_shipping_address": "📍",
    "check_cancellation_fee": "💰",
    "check_invoice": "🧾",
    "check_payment_methods": "💳",
    "check_refund_policy": "📋",
    "complaint": "😟",
    "contact_customer_service": "🎧",
    "contact_human_agent": "👩‍💼",
    "create_account": "👤",
    "delete_account": "🗑️",
    "delivery_options": "🚚",
    "delivery_period": "⏰",
    "edit_account": "✏️",
    "get_invoice": "📄",
    "get_refund": "💵",
    "newsletter_subscription": "📩",
    "payment_issue": "⚠️",
    "place_order": "🛒",
    "recover_password": "🔑",
    "registration_problems": "⚠️",
    "review": "⭐",
    "set_up_shipping_address": "📍",
    "switch_account": "🔄",
    "track_order": "📦",
    "track_refund": "🔎",
}
INTENT_TO_CATEGORY = {
    # ACCOUNT
    "edit_account": "ACCOUNT",
    "create_account": "ACCOUNT",
    "delete_account": "ACCOUNT",
    "recover_password": "ACCOUNT",
    "registration_problems": "ACCOUNT",
    "switch_account": "ACCOUNT",

    # ORDER
    "cancel_order": "ORDER",
    "change_order": "ORDER",
    "place_order": "ORDER",
    "track_order": "ORDER",

    # REFUND
    "check_refund_policy": "REFUND",
    "get_refund": "REFUND",
    "track_refund": "REFUND",

    # INVOICE
    "check_invoice": "INVOICE",
    "get_invoice": "INVOICE",

    # CONTACT
    "contact_customer_service": "CONTACT",
    "contact_human_agent": "CONTACT",

    # PAYMENT
    "check_payment_methods": "PAYMENT",
    "payment_issue": "PAYMENT",

    # FEEDBACK
    "complaint": "FEEDBACK",
    "review": "FEEDBACK",

    # DELIVERY
    "delivery_options": "DELIVERY",
    "delivery_period": "DELIVERY",

    # SHIPPING
    "set_up_shipping_address": "SHIPPING",
    "change_shipping_address": "SHIPPING",

    # SUBSCRIPTION
    "newsletter_subscription": "SUBSCRIPTION",

    # CANCELLATION
    "check_cancellation_fee": "CANCEL",
}

VOCAB_SIZE = len(vectorizer.vocabulary_)
NUM_CATEGORIES = len(model.classes_)
TRAIN_ROWS = 26872
HELD_OUT_ACCURACY = 99.48


def predict_category(raw_text: str):
    cleaned = preprocess(raw_text)
    vec = vectorizer.transform([cleaned])

    predicted_category = model.predict(vec)[0]

    confidence = None
    top_alternatives = []

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = proba.max() * 100

        top_idx = proba.argsort()[::-1][:3]

        top_alternatives = [
            (model.classes_[i], proba[i] * 100)
            for i in top_idx
        ]

    return predicted_category, confidence, cleaned, top_alternatives


def transcribe_audio_bytes(audio_bytes: bytes, pause_threshold: float) -> str:
    """Transcribe browser-recorded WAV audio without requiring PyAudio."""
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold

    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)

    return recognizer.recognize_google(audio)


# ---------------------------------------------------------------------------
# Page + theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Voice Support Router", page_icon="📞", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.block-container { padding-top: 2rem; max-width: 1200px; }

/* header */
.app-title { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem; }
.app-title h1 { font-size: 2rem; font-weight: 700; color: #F2F5F7; margin: 0; }
.app-tagline { color: #8B98A5; font-size: 1.02rem; max-width: 680px; margin-top: 0.3rem; }

/* info box */
.info-box {
    background: #161B22;
    border: 1px solid #2A313C;
    border-left: 3px solid #2DD4BF;
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin: 1.3rem 0 1.6rem 0;
    color: #C4CDD5;
    font-size: 0.92rem;
    line-height: 1.6;
}
.info-box b { color: #F2F5F7; }

/* panels */
.panel {
    background: #161B22;
    border: 1px solid #2A313C;
    border-radius: 8px;
    padding: 1.5rem;
    height: 100%;
}

.ticket-empty {
    color: #64707C;
    font-size: 0.95rem;
    text-align: center;
    padding: 3rem 1rem;
    border: 1px dashed #2A313C;
    border-radius: 6px;
}

.ticket-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: #64707C;
    margin-bottom: 0.35rem;
}

.category-badge { font-weight: 700; font-size: 1.25rem; color: #F2F5F7; margin-bottom: 0.15rem; }
.category-sub { color: #8B98A5; font-size: 0.9rem; margin-bottom: 1.1rem; }

.route-status {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; font-weight: 700;
    padding: 0.3rem 0.7rem; border-radius: 4px;
    margin-bottom: 1.2rem;
}
.route-auto   { background: rgba(45,212,191,0.14); color: #2DD4BF; }
.route-review { background: rgba(217,142,62,0.16); color: #E4A854; }
.route-low    { background: rgba(193,80,46,0.18); color: #E37A54; }

.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot-auto   { background: #2DD4BF; }
.dot-review { background: #E4A854; }
.dot-low    { background: #E37A54; }

.conf-track { width:100%; height:8px; background:#242B34; border-radius:4px; overflow:hidden; margin-bottom:0.3rem; }
.conf-fill { height:100%; border-radius:4px; }
.conf-value { font-family:'IBM Plex Mono', monospace; font-size:0.85rem; color:#F2F5F7; margin-bottom:1.2rem; }

.alt-row {
    display:flex; justify-content:space-between; font-size:0.85rem;
    color:#8B98A5; padding:0.4rem 0; border-bottom:1px solid #242B34;
}
.alt-row:last-child { border-bottom:none; }
.alt-value { font-family:'IBM Plex Mono', monospace; color:#F2F5F7; }

.transcript-box {
    background: #0E1319;
    border-left: 3px solid #2DD4BF;
    padding: 0.75rem 1rem;
    font-size: 0.95rem;
    color: #F2F5F7;
    border-radius: 0 4px 4px 0;
    margin-bottom: 1rem;
}

/* stat cards */
.stat-label { color:#8B98A5; font-size:0.85rem; margin-bottom:0.2rem; }
.stat-value { font-family:'IBM Plex Mono', monospace; font-weight:700; font-size:2.1rem; color:#2DD4BF; }
.stat-value.alt { color: #F2F5F7; }

/* category grid */
.cat-card {
    background:#161B22; border:1px solid #2A313C; border-radius:8px;
    padding:1rem 1.1rem; margin-bottom:0.7rem;
}
.cat-name { font-weight:700; color:#F2F5F7; font-size:1rem; }
.cat-code { font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:#2DD4BF; letter-spacing:0.05em; }
.cat-desc { color:#8B98A5; font-size:0.87rem; margin-top:0.2rem; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0E1319; border-right: 1px solid #2A313C; }

/* ================= MODERN UI OVERRIDES ================= */
.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(45,212,191,.10), transparent 28%),
        radial-gradient(circle at 92% 10%, rgba(99,102,241,.10), transparent 25%),
        #090D13;
}
.block-container { max-width: 1280px; padding-top: 2rem; }

.hero {
    padding: .3rem 0 1.3rem 0;
}
.hero-kicker {
    display:inline-block;
    padding:.35rem .7rem;
    border-radius:999px;
    background:rgba(45,212,191,.09);
    border:1px solid rgba(45,212,191,.25);
    color:#5EEAD4;
    font-size:.7rem;
    font-weight:800;
    letter-spacing:.1em;
}
.hero h1 {
    margin:.65rem 0 .35rem;
    font-size:2.5rem;
    font-weight:800;
    letter-spacing:-.04em;
    color:#F8FAFC;
}
.hero p {
    margin:0;
    max-width:760px;
    color:#94A3B8;
    line-height:1.65;
}

.panel {
    background:rgba(17,24,34,.78) !important;
    border:1px solid rgba(71,85,105,.42) !important;
    border-radius:18px !important;
    box-shadow:0 18px 45px rgba(0,0,0,.22);
    backdrop-filter:blur(12px);
}

.panel-title {
    color:#F8FAFC;
    font-size:1.02rem;
    font-weight:750;
    margin-bottom:.15rem;
}
.panel-subtitle {
    color:#64748B;
    font-size:.78rem;
    margin-bottom:1rem;
}

div.stButton > button {
    border-radius:11px !important;
    border:1px solid rgba(45,212,191,.30) !important;
    background:linear-gradient(135deg,rgba(45,212,191,.16),rgba(45,212,191,.05)) !important;
    color:#E6FFFB !important;
    font-weight:700 !important;
    transition:.2s ease;
}
div.stButton > button:hover {
    border-color:#2DD4BF !important;
    transform:translateY(-1px);
    box-shadow:0 8px 22px rgba(45,212,191,.12);
}

.transcript-box {
    background:rgba(8,13,19,.82) !important;
    border:1px solid #263241 !important;
    border-left:3px solid #2DD4BF !important;
    border-radius:10px !important;
    line-height:1.55;
}

.route-result {
    display:flex;
    align-items:center;
    gap:.8rem;
    padding:.85rem;
    margin-bottom:.8rem;
    border:1px solid #263241;
    border-radius:13px;
    background:rgba(15,23,42,.55);
}
.route-icon {
    width:44px;
    height:44px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:12px;
    background:rgba(45,212,191,.10);
    font-size:1.35rem;
}

.ticket-empty {
    background:rgba(8,13,19,.35) !important;
    border:1px dashed #2B3645 !important;
    border-radius:14px !important;
    padding:4.5rem 1rem !important;
}

.cat-card {
    background:rgba(17,24,34,.72) !important;
    border:1px solid #263241 !important;
    border-radius:15px !important;
    transition:.2s ease;
}
.cat-card:hover {
    transform:translateY(-2px);
    border-color:rgba(45,212,191,.35) !important;
}

.stat-card {
    background:rgba(15,23,42,.62);
    border:1px solid #263241;
    border-radius:14px;
    padding:.85rem;
}

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0B1118,#080C12) !important;
}
.sidebar-logo {
    display:flex;
    align-items:center;
    justify-content:center;
    width:42px;
    height:42px;
    border-radius:12px;
    background:rgba(45,212,191,.12);
    border:1px solid rgba(45,212,191,.20);
    font-size:1.3rem;
}
.sidebar-title {
    color:#F8FAFC;
    font-weight:800;
    margin-top:.6rem;
}
.sidebar-subtitle {
    color:#64748B;
    font-size:.75rem;
}
#MainMenu, footer { visibility:hidden; }


/* ================= ANIMATION LAYER ================= */

/* Soft animated background glow */
.stApp::before {
    content: "";
    position: fixed;
    inset: -20%;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 15% 20%, rgba(45,212,191,.07), transparent 22%),
        radial-gradient(circle at 85% 30%, rgba(99,102,241,.06), transparent 20%);
    animation: ambientMove 12s ease-in-out infinite alternate;
}

@keyframes ambientMove {
    0%   { transform: translate3d(-1%, -1%, 0) scale(1); }
    100% { transform: translate3d(2%, 1%, 0) scale(1.04); }
}

/* Header entrance */
.hero {
    animation: heroEnter .7s ease-out both;
}

@keyframes heroEnter {
    from {
        opacity: 0;
        transform: translateY(14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Kicker has a subtle glow */
.hero-kicker {
    animation: kickerGlow 3s ease-in-out infinite;
}

@keyframes kickerGlow {
    0%, 100% { box-shadow: 0 0 0 rgba(45,212,191,0); }
    50%      { box-shadow: 0 0 18px rgba(45,212,191,.12); }
}

/* Cards gently rise into view */
.panel {
    animation: cardEnter .6s ease-out both;
}

@keyframes cardEnter {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Recording button gets a soft breathing effect */
div.stButton > button {
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
}

div.stButton > button:focus {
    animation: buttonPulse 1.5s ease-in-out infinite;
}

@keyframes buttonPulse {
    0%, 100% {
        box-shadow: 0 0 0 0 rgba(45,212,191,.18);
    }
    50% {
        box-shadow: 0 0 0 8px rgba(45,212,191,0);
    }
}

/* Routing result slides in when displayed */
.route-result {
    animation: resultEnter .55s cubic-bezier(.2,.8,.2,1) both;
}

@keyframes resultEnter {
    from {
        opacity: 0;
        transform: translateX(12px) scale(.98);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
}

/* Route icon gently floats */
.route-icon {
    animation: iconFloat 3s ease-in-out infinite;
}

@keyframes iconFloat {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-3px); }
}

/* Confidence bar shimmer */
div[data-testid="stProgress"] > div > div {
    position: relative;
    overflow: hidden;
}

div[data-testid="stProgress"] > div > div::after {
    content: "";
    position: absolute;
    inset: 0;
    width: 35%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,.18),
        transparent
    );
    animation: shimmer 2.2s linear infinite;
}

@keyframes shimmer {
    from { transform: translateX(-120%); }
    to   { transform: translateX(350%); }
}

/* Category cards lift smoothly */
.cat-card {
    transition:
        transform .22s ease,
        border-color .22s ease,
        box-shadow .22s ease;
}

.cat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0,0,0,.22);
}

/* Sidebar logo pulse */
.sidebar-logo {
    animation: logoPulse 3s ease-in-out infinite;
}

@keyframes logoPulse {
    0%, 100% { box-shadow: 0 0 0 rgba(45,212,191,0); }
    50%      { box-shadow: 0 0 22px rgba(45,212,191,.14); }
}

/* Respect users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — settings + model stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div>
        <div class="sidebar-logo">📞</div>
        <div class="sidebar-title">Voice Support AI</div>
        <div class="sidebar-subtitle">Real-time intent routing</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Recognition settings**")
    pause_threshold = st.slider(
    "Pause sensitivity (seconds of silence before stopping)",
    min_value=0.5, max_value=4.0, value=2.5, step=0.1,
        help="Higher = waits longer for you to finish talking before cutting off.",
    )
    auto_route_threshold = st.slider(
        "Auto-route confidence threshold (%)",
        min_value=50, max_value=95, value=80, step=5,
        help="Predictions at or above this confidence are auto-routed. Below it, they're flagged for human review.",
    )
    review_threshold = st.slider(
        "Minimum confidence before flagging as low (%)",
        min_value=10, max_value=50, value=50, step=5,
    )

    st.markdown("---")
    st.markdown("**Model**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-label">Held-out accuracy</div><div class="stat-value">{HELD_OUT_ACCURACY}%</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-label">Categories</div><div class="stat-value alt">{NUM_CATEGORIES}</div>', unsafe_allow_html=True)
    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f'<div class="stat-label">Training rows</div><div class="stat-value alt">{TRAIN_ROWS:,}</div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-label">TF-IDF features</div><div class="stat-value alt">{VOCAB_SIZE:,}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-kicker">🤖 AI SUPPORT TRIAGE</div>
    <h1>Voice Customer Support Router</h1>
    <p>Describe your issue by voice or text and let the AI classify it, score its confidence, and route it to the right support team.</p>
</div>
""", unsafe_allow_html=True)

tab_route, tab_categories, tab_how, tab_model = st.tabs(
    ["🎯 Route", "📋 Categories", "⚙️ How it works", "📊 Model"]
)

# ---------------------------------------------------------------------------
# TAB: Route
# ---------------------------------------------------------------------------
with tab_route:
    with st.expander("👋 New here? Read this first", expanded=False):
        st.markdown(
            "**Three steps:**\n\n"
            "1. Pick **Speak** or **Type** below and describe a support issue in one sentence.\n"
            "2. The app transcribes it (if spoken) and runs it through the trained model.\n"
            "3. Read the routed team, the confidence score, and the routing decision on the right.\n\n"
            "If confidence is low, that's expected for casual or ambiguous phrasing — the model "
            "flags it instead of guessing confidently. Adjust thresholds in the sidebar."
        )

    st.write("")
    col_input, col_ticket = st.columns([1, 1], gap="large")

    result = st.session_state.pop("route_result", None)

    with col_input:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">💬 Tell us what you need</div><div class="panel-subtitle">Record your voice or type a support request</div>', unsafe_allow_html=True)
        tab_mic, tab_text = st.tabs(["🎙️ Speak", "⌨️ Type"])

        with tab_mic:
            st.caption("Record your issue using your browser microphone.")

            if "audio_widget_key" not in st.session_state:
                st.session_state["audio_widget_key"] = 0

            audio_value = st.audio_input(
                "🎙️ Record your request",
                key=f"support_audio_{st.session_state['audio_widget_key']}",
            )

            if audio_value is not None:
                try:
                    recognized_text = transcribe_audio_bytes(
                        audio_value.getvalue(),
                        pause_threshold,
                    )
                    st.session_state["voice_transcript"] = recognized_text
                except sr.UnknownValueError:
                    st.error("Couldn't understand the audio. Speak clearly and try again.")
                except sr.RequestError:
                    st.error("Speech recognition service is unavailable. Please try again or use the Type tab.")
                except Exception as exc:
                    st.error(f"Audio processing failed: {exc}")

            voice_transcript = st.session_state.get("voice_transcript", "")

            if voice_transcript:
                st.markdown(
                    f'<div class="transcript-box">🎙️ "{voice_transcript}"</div>',
                    unsafe_allow_html=True,
                )

                record_again, route_voice = st.columns(2)
                with record_again:
                    if st.button("🔄 Record Again", use_container_width=True):
                        st.session_state["voice_transcript"] = ""
                        st.session_state["audio_widget_key"] += 1
                        st.rerun()

                with route_voice:
                    if st.button("🎯 Route Request", use_container_width=True):
                        st.session_state["route_result"] = ("mic", voice_transcript)
                        st.rerun()

        with tab_text:
            typed_text = st.text_area(
                "Describe your issue", placeholder="e.g. I want to cancel my order and get a refund",
                label_visibility="collapsed", height=100,
            )
            if st.button("Route Request", use_container_width=True):
                if typed_text.strip():
                    result = ("text", typed_text)
                else:
                    st.warning("Type something first.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_ticket:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Routing Result</div><div class="panel-subtitle">AI classification and confidence</div>', unsafe_allow_html=True)

        if result:
            source, raw_text = result
            category, confidence, cleaned, alternatives = predict_category(raw_text)
            routing_category = INTENT_TO_CATEGORY.get(category, category)
            team, description = CATEGORY_INFO.get(routing_category,(category, ""))

            st.markdown(f'<div class="transcript-box">"{raw_text}"</div>', unsafe_allow_html=True)

            emoji = INTENT_EMOJIS.get(category, "🤖")
            st.markdown('<div class="ticket-label">ROUTED TO</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="route-result"><div class="route-icon">{emoji}</div>'
                f'<div><div class="category-badge">{team}</div>'
                f'<div class="category-sub">{category} · {description}</div></div></div>',
                unsafe_allow_html=True
            )
            if confidence is not None:
                if confidence >= auto_route_threshold:
                    status_class, dot_class, status_text = "route-auto", "dot-auto", "AUTO-ROUTED"
                elif confidence >= review_threshold:
                    status_class, dot_class, status_text = "route-review", "dot-review", "FLAGGED FOR REVIEW"
                else:
                    status_class, dot_class, status_text = "route-low", "dot-low", "LOW CONFIDENCE"

                st.markdown(
                    f'<div class="route-status {status_class}"><span class="dot {dot_class}"></span>{status_text}</div>',
                    unsafe_allow_html=True,
                )

                fill_color = "#2DD4BF" if confidence >= auto_route_threshold else (
                    "#E4A854" if confidence >= review_threshold else "#E37A54"
                )
                st.markdown(
                    f'<div class="conf-track"><div class="conf-fill" '
                    f'style="width:{min(confidence,100):.0f}%;background:{fill_color};"></div></div>'
                    f'<div class="conf-value">{confidence:.1f}% confidence</div>',
                    unsafe_allow_html=True,
                )

            if alternatives:
                st.markdown('<div class="ticket-label">OTHER CANDIDATES</div>', unsafe_allow_html=True)
                rows = "".join(
                    f'<div class="alt-row"><span>{cat}</span><span class="alt-value">{prob:.1f}%</span></div>'
                    for cat, prob in alternatives[1:]
                )
                st.markdown(rows, unsafe_allow_html=True)

            with st.expander("Preprocessed text sent to the model"):
                st.code(cleaned)
        else:
            st.markdown(
                '<div class="ticket-empty">No request yet.<br>Speak or type an issue to see it routed.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB: Categories
# ---------------------------------------------------------------------------
with tab_categories:
    st.markdown("Every request is routed into one of these 11 categories.")
    st.write("")
    cols = st.columns(2)
    for i, code in enumerate(CATEGORY_INFO.keys()):
        team, desc = CATEGORY_INFO.get(code, (code, ""))
        with cols[i % 2]:
            st.markdown(
                f'<div class="cat-card"><div class="cat-code">{code}</div>'
                f'<div class="cat-name">{team}</div><div class="cat-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# TAB: How it works
# ---------------------------------------------------------------------------
with tab_how:
    st.markdown("""
**Pipeline:** `Speech → Text → Preprocessing → TF-IDF Vectorization → Logistic Regression → Category + Confidence`

1. **Speech-to-text** — browser-recorded audio is transcribed via the Google Web Speech API.
2. **Preprocessing** — the transcript is lowercased, stripped of punctuation and digits,
   filtered of stopwords, and lemmatized (`src/preprocessing.py`) — the same function used
   during training, so live speech is cleaned identically to how the model learned.
3. **Vectorization** — cleaned text is converted into TF-IDF features (unigrams + bigrams,
   5,000-word vocabulary).
4. **Prediction** — a tuned Logistic Regression model predicts a category and a confidence score.
5. **Routing decision** — confidence is compared against the thresholds in the sidebar:
   high confidence auto-routes, mid confidence flags for human review, low confidence
   surfaces the top-3 candidates instead of guessing.

Trained on the **Bitext Customer Support dataset** — 26,872 real, labeled support requests.
    """)

# ---------------------------------------------------------------------------
# TAB: Model
# ---------------------------------------------------------------------------
with tab_model:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Held-out accuracy", f"{HELD_OUT_ACCURACY}%"),
        (c2, "Categories", str(NUM_CATEGORIES)),
        (c3, "Training rows", f"{TRAIN_ROWS:,}"),
        (c4, "TF-IDF features", f"{VOCAB_SIZE:,}"),
    ]:
        with col:
            st.markdown(f'<div class="stat-label">{label}</div><div class="stat-value">{value}</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("""
**Model:** Logistic Regression (`C=10`, tuned via `GridSearchCV`)
**Vectorizer:** TF-IDF, unigrams + bigrams, max 5,000 features
**Dataset:** Bitext Customer Support Training Dataset (Hugging Face)

**Known limitation:** the training data uses fairly formal phrasing for some categories
(e.g. `FEEDBACK` examples say "file a complaint" / "lodge a reclamation" rather than casual
phrasing like "that was terrible"). Casual phrasing outside that vocabulary can produce lower
confidence predictions — which is why the app surfaces a confidence score and flags low-confidence
results for review instead of always auto-routing.
    """)