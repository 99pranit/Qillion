import os
import json
import random
import pandas as pd
import streamlit as st
from openai import OpenAI

from package.classify_query import classifyQuery
from package.identify_purpose import identifyPurpose
from package.learning_path import learningPath

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LearnPath AI · Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS  (matches app_v2 palette)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── typography ── */
.main-header {
    font-size: 2.4rem; font-weight: 800;
    color: #1f77b4; text-align: center; margin-bottom: .4rem;
}
.sub-header {
    font-size: 1.1rem; color: #666;
    text-align: center; margin-bottom: 1.8rem;
}

/* ── step badges ── */
.step-badge {
    display: inline-block;
    background: #1f77b4; color: #fff;
    font-size: .78rem; font-weight: 700;
    padding: 2px 10px; border-radius: 99px;
    margin-right: 6px;
}

/* ── taxonomy badge ── */
.taxonomy-badge {
    padding: .55rem 1.1rem; border-radius: .5rem;
    font-weight: 800; font-size: 1.45rem;
    text-align: center; margin: .8rem 0;
    letter-spacing: .04em;
}
.knowledge    { background:#e3f2fd; color:#1565c0; }
.comprehension{ background:#e8f5e9; color:#2e7d32; }
.application  { background:#fff3e0; color:#ef6c00; }
.analysis     { background:#fce4ec; color:#c2185b; }
.synthesis    { background:#f3e5f5; color:#7b1fa2; }
.evaluation   { background:#ffebee; color:#c62828; }

/* ── purpose box ── */
.purpose-box {
    padding: 1rem; border-radius: .5rem;
    background: #f0f2f6; border-left: 4px solid #1f77b4;
    margin: .8rem 0; font-size: .97rem;
}

/* ── stage card ── */
.stage-card {
    padding: .85rem 1.1rem; border-radius: .55rem;
    background: #f8f9fa; border-left: 5px solid #1f77b4;
    margin-bottom: .9rem;
}

/* ── content card ── */
.content-card {
    background: #f8f9fa; border-radius: .6rem;
    padding: 1.4rem 1.6rem; margin-top: .5rem;
    border: 1px solid #e0e0e0;
}

/* ── regen counter ── */
.regen-pill {
    display: inline-block;
    background: #fff3cd; color: #856404;
    border: 1px solid #ffc107; border-radius: 99px;
    font-size: .8rem; font-weight: 600;
    padding: 2px 12px; margin-left: 8px;
}
.regen-done {
    background: #f8d7da; color: #721c24; border-color: #f5c6cb;
}

/* ── buttons ── */
.stButton>button {
    background: #1f77b4; color: #fff;
    font-weight: 600; border-radius: .45rem;
    border: none; padding: .42rem .9rem;
}
.stButton>button:hover { background: #155a8a; }

/* ── divider ── */
hr { margin: 1.4rem 0; border-color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
_defaults = {
    # shared
    "classification_history": [],
    # chat flow
    "chat_query":                None,
    "chat_label":                None,
    "chat_confidence":           None,
    "chat_votes":                [],
    "chat_purpose":              None,
    "chat_purpose_confirmed":    False,
    "chat_show_editor":          False,
    "chat_topic_data":           None,
    "chat_learning_path":        None,
    "chat_regen_count":          0,
    "chat_show_regen_input":     False,
    "chat_generated_content":    None,
    "chat_show_content":         False,
    "generate_word":             False,
    "generate_pdf":              False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

MAX_REGEN = 3

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_provider = st.selectbox("API Provider", ["groq"],
                                help="Select your API provider")

    if os.environ.get(f"{api_provider.upper()}_API_KEY"):
        api_key_input = os.environ.get(f"{api_provider.upper()}_API_KEY")

    else:
        api_key_input = st.text_input(
            "API Key", type="password",
            placeholder="Enter key (or set GROQ_API_KEY env var)",
            help="Get yours at console.groq.com",
            value=os.environ.get(f"{api_provider.upper()}_API_KEY", ""),
        )
        st.error("⚠️ API Key not found")
        st.info(f"Set {api_provider.upper()}_API_KEY environment variable")

    st.divider()

    model_name = st.selectbox(
        "Primary Model",
        ['openai/gpt-oss-120b', 
         'openai/gpt-oss-20b', 
         'meta-llama/llama-4-scout-17b-16e-instruct'],
        help="Model used for classification, purpose & learning path",
    )

    correction_model = st.selectbox(
        "Correction Model",
        ['openai/gpt-oss-120b', 
         'openai/gpt-oss-20b', 
         'meta-llama/llama-4-scout-17b-16e-instruct'],
        help="Lightweight model for label self-correction",
    )

    content_model = st.selectbox(
        "Content Generation Model",
        ['openai/gpt-oss-120b', 
         'openai/gpt-oss-20b', 
         'meta-llama/llama-4-scout-17b-16e-instruct'],
        help="Model for generating the final learning content",
    )

    st.divider()
    st.header("🎭 Classification Persona")
    persona = st.selectbox(
        "Persona",
        ["multi", "professor", "student", "psychologist", "engineer", "examiner"],
        help="'multi' uses ensemble voting across all personas",
    )
    persona_desc = {
        "multi":       "Ensemble majority-vote — most robust",
        "professor":   "University professor: deep educational expertise",
        "student":     "Student lens on learning objectives",
        "psychologist":"Clinical cognitive-assessment approach",
        "engineer":    "Technical, practical problem-solving focus",
        "examiner":    "Assessment & evaluation specialist",
    }
    st.info(persona_desc[persona])

    st.divider()
    st.header("📖 Bloom's Taxonomy")
    with st.expander("View All Levels"):
        st.markdown("""
        1. **Knowledge** – Recall facts & terms
        2. **Comprehension** – Interpret & explain
        3. **Application** – Use in new situations
        4. **Analysis** – Break down & examine
        5. **Synthesis** – Create / propose solutions
        6. **Evaluation** – Judge based on criteria
        """)

    # ── live session summary ──
    if st.session_state.chat_label:
        st.divider()
        st.header("📊 Current Session")
        st.markdown(f"**Level:** `{st.session_state.chat_label.capitalize()}`")
        if st.session_state.chat_confidence:
            st.progress(st.session_state.chat_confidence)
        if st.session_state.chat_purpose:
            st.markdown(f"**Purpose:**\n> {st.session_state.chat_purpose}")
        regen_left = MAX_REGEN - st.session_state.chat_regen_count
        if st.session_state.chat_learning_path:
            color = "🟡" if regen_left > 0 else "🔴"
            st.markdown(f"**Path Regenerations Left:** {color} {regen_left}")

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎓 LearnPath AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Personalised learning journeys · powered by Bloom\'s Taxonomy</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────
tab_chat, tab_history = st.tabs(["💬 Chatbot", "📊 History"])

# ═════════════════════════════════════════════════════════════════
#  TAB 1 · CHATBOT
# ═════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 AI Educational Assistant")
    st.info("🎯 Ask any question → get a personalised, Bloom-aligned learning journey!")

    # ── query input ──────────────────────────────────────────────
    query_input = st.text_area(
        "Ask your educational question:",
        height=140,
        placeholder="e.g. How does backpropagation work in neural networks?\n"
                    "or: Design a microservices architecture for an e-commerce platform",
        help="Any educational topic or question works.",
        key="chat_query_input",
    )

    start_btn = st.button("🚀 Start Learning Journey", type="primary", key="start_chat")

    # ════════════════════════════════════════════
    #  STEP 1 + 2: classify & identify purpose
    # ════════════════════════════════════════════
    if start_btn:
        if not query_input.strip():
            st.error("⚠️ Please enter a question to begin.")
        else:
            # reset flow
            for k in ["chat_label", "chat_confidence", "chat_votes",
                      "chat_purpose", "chat_purpose_confirmed", "chat_show_editor",
                      "chat_topic_data", "chat_learning_path", "chat_regen_count",
                      "chat_show_regen_input", "chat_generated_content",
                      "chat_show_content", "generate_word", "generate_pdf"]:
                st.session_state[k] = _defaults.get(k)

            st.session_state.chat_query = query_input.strip()

            # ── Step 1: classify ──────────────────────────────
            st.markdown("---")
            st.markdown('<span class="step-badge">STEP 1</span> **Cognitive Level Analysis**',
                        unsafe_allow_html=True)

            with st.spinner("🔄 Classifying your query across expert personas…"):
                try:
                    clf = classifyQuery(
                        api_provider=api_provider,
                        persona=persona,
                        query=st.session_state.chat_query,
                        model_name=model_name,
                        correction_model=correction_model,
                    )
                    if persona == "multi":
                        res = clf.get_ensemble_label()
                        label = res["final_label"]
                        confidence = res["confidence"]
                        votes = res.get("votes", [])
                    else:
                        label = clf.get_label()
                        confidence = 1.0
                        votes = [label]

                    st.session_state.chat_label = label
                    st.session_state.chat_confidence = confidence
                    st.session_state.chat_votes = votes

                except Exception as e:
                    st.error(f"⚠️ Classification error: {e}")
                    st.stop()

            # ── Step 2: purpose ───────────────────────────────
            st.markdown("---")
            st.markdown('<span class="step-badge">STEP 2</span> **Understanding Your Intent**',
                        unsafe_allow_html=True)

            with st.spinner("🔍 Identifying your learning goal…"):
                try:
                    pi = identifyPurpose(
                        api_provider=api_provider,
                        query=st.session_state.chat_query,
                        model_name=model_name,
                        cognitive_level=label,
                    )
                    pi.setup_api()
                    purpose = pi.get_purpose()
                    st.session_state.chat_purpose = purpose
                except Exception as e:
                    st.error(f"⚠️ Purpose identification error: {e}")

            st.rerun()

    # ═══════════════════════════════════════════
    #  DISPLAY: classification result
    # ═══════════════════════════════════════════
    if st.session_state.chat_label:
        label = st.session_state.chat_label
        confidence = st.session_state.chat_confidence or 1.0
        votes = st.session_state.chat_votes or []

        st.markdown("---")
        st.markdown('<span class="step-badge">STEP 1</span> **Cognitive Level Analysis**',
                    unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f'<div class="taxonomy-badge {label}">{label.upper()}</div>',
                unsafe_allow_html=True,
            )
            taxonomy_info = {
                "knowledge":     "Recalling facts, terms, basic concepts, or answers",
                "comprehension": "Demonstrating understanding by interpreting, summarising, or explaining",
                "application":   "Using learned information in new concrete situations",
                "analysis":      "Breaking down information into parts and examining relationships",
                "synthesis":     "Combining elements to form a new whole or proposing solutions",
                "evaluation":    "Making judgements based on criteria and standards",
            }
            st.info(f"**Description:** {taxonomy_info.get(label, '')}")
        with c2:
            st.metric("Confidence", f"{confidence:.0%}")
            st.progress(confidence)

        # persona votes breakdown
        if persona == "multi" and len(votes) == 5:
            with st.expander("📊 Individual Persona Votes"):
                personas_list = ["Professor", "Student", "Psychologist", "Engineer", "Examiner"]
                vcols = st.columns(5)
                for col, pname, vote in zip(vcols, personas_list, votes):
                    with col:
                        st.markdown(
                            f'<div style="text-align:center;border:1px solid #ddd;'
                            f'border-radius:8px;padding:8px 4px">'
                            f'<p style="font-size:.72rem;color:#888;margin:0">{pname}</p>'
                            f'<p style="font-weight:700;margin:4px 0;font-size:.88rem">{vote.capitalize()}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ═══════════════════════════════════════════
    #  DISPLAY: purpose confirmation
    # ═══════════════════════════════════════════
    if st.session_state.chat_purpose and not st.session_state.chat_purpose_confirmed:
        st.markdown("---")
        st.markdown('<span class="step-badge">STEP 2</span> **Your Identified Learning Goal**',
                    unsafe_allow_html=True)

        st.markdown(
            f'<div class="purpose-box"><strong>🎯 Detected Intent:</strong> '
            f'{st.session_state.chat_purpose}</div>',
            unsafe_allow_html=True,
        )
        st.write("**Is this what you want to achieve?**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, that's my goal", key="confirm_purpose", use_container_width=True):
                st.session_state.chat_purpose_confirmed = True
                st.session_state.chat_show_editor = False
                st.rerun()
        with col2:
            if st.button("✏️ No, let me edit it", key="edit_purpose", use_container_width=True):
                st.session_state.chat_show_editor = True
                st.rerun()

        if st.session_state.chat_show_editor:
            st.markdown("**Clarify Your Goal:**")
            edited = st.text_area(
                "What do you actually want to achieve or learn?",
                value=st.session_state.chat_purpose,
                height=90,
                key="purpose_editor_area",
            )
            if st.button("💾 Save My Goal", key="save_purpose", use_container_width=True):
                st.session_state.chat_purpose = edited.strip() or st.session_state.chat_purpose
                st.session_state.chat_purpose_confirmed = True
                st.session_state.chat_show_editor = False
                st.rerun()

    # ═══════════════════════════════════════════
    #  STEP 3: learning path generation
    # ═══════════════════════════════════════════
    if (st.session_state.chat_purpose_confirmed
            and st.session_state.chat_purpose
            and st.session_state.chat_label):

        label = st.session_state.chat_label

        # auto-generate on first confirmation
        if st.session_state.chat_learning_path is None:
            st.markdown("---")
            st.markdown('<span class="step-badge">STEP 3</span> **Building Your Personalised Learning Path…**',
                        unsafe_allow_html=True)
            with st.spinner("🧭 Structuring your learning path…"):
                try:
                    lp_obj = learningPath(
                        api_provider=api_provider,
                        query=st.session_state.chat_query,
                        model_name=model_name,
                        cognitive_level=label,
                        user_purpose=st.session_state.chat_purpose,
                    )
                    lp_obj.setup_api()
                    lp_obj.get_path()            # populates _topic_data
                    lp = lp_obj.get_learning_path()
                    st.session_state.chat_learning_path = lp
                    st.session_state.chat_topic_data = lp_obj._topic_data
                except Exception as e:
                    st.error(f"⚠️ Learning path error: {e}")
            st.rerun()

        # ── display path ─────────────────────────────────────
        if st.session_state.chat_learning_path:
            lp = st.session_state.chat_learning_path
            regen_left = MAX_REGEN - st.session_state.chat_regen_count

            st.markdown("---")
            st.markdown('<span class="step-badge">STEP 3</span> **Your Personalised Learning Path**',
                        unsafe_allow_html=True)

            lp_title = lp.get("learning_path_title", "Your Learning Path")
            st.markdown(f"### 📘 {lp_title}")

            # topic metadata chips
            td = st.session_state.chat_topic_data or {}
            if td:
                m1, m2, m3 = st.columns(3)
                m1.metric("Core Topic", td.get("core_topic", "—"))
                m2.metric("Domain", td.get("domain", "—"))
                diff = td.get("difficulty_hint", "—")
                m3.metric("Difficulty", diff.capitalize())

            st.markdown("<br>", unsafe_allow_html=True)

            # stages
            stages = lp.get("stages", [])
            stage_colors = ["#1f77b4", "#7b1fa2", "#2e7d32",
                            "#ef6c00", "#c2185b", "#c62828"]
            for i, stage in enumerate(stages):
                sc = stage_colors[i % len(stage_colors)]
                st.markdown(
                    f'<div class="stage-card" style="border-left-color:{sc}">'
                    f'<p style="margin:0;font-weight:700;font-size:1rem;color:{sc}">'
                    f'Stage {stage.get("stage_number", i+1)}: {stage.get("title","")}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for obj in stage.get("objectives", []):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;• {obj}")
                st.markdown("")

            # ── regeneration controls ──────────────────────
            st.markdown("---")
            regen_pill_cls = "regen-pill" if regen_left > 0 else "regen-pill regen-done"
            st.markdown(
                f'<p style="margin:0">🔁 Not satisfied?'
                f'<span class="{regen_pill_cls}">{regen_left} regeneration{"s" if regen_left != 1 else ""} left</span>'
                f'</p>',
                unsafe_allow_html=True,
            )

            if regen_left > 0:
                if not st.session_state.chat_show_regen_input:
                    if st.button("🔄 Regenerate Learning Path", key="show_regen", use_container_width=False):
                        st.session_state.chat_show_regen_input = True
                        st.rerun()
                else:
                    regen_feedback = st.text_input(
                        "Tell us what to improve (optional):",
                        placeholder="e.g. 'more practical focus', 'add advanced topics', 'shorter stages'",
                        key="regen_feedback_field",
                    )
                    rc1, rc2 = st.columns([1, 3])
                    with rc1:
                        if st.button("✅ Regenerate", key="do_regen", type="primary"):
                            with st.spinner("🔄 Regenerating your learning path…"):
                                try:
                                    # append feedback to purpose so the LLM sees it
                                    augmented_purpose = st.session_state.chat_purpose
                                    if regen_feedback.strip():
                                        augmented_purpose += f" [User feedback: {regen_feedback.strip()}]"

                                    lp_obj = learningPath(
                                        api_provider=api_provider,
                                        query=st.session_state.chat_query,
                                        model_name=model_name,
                                        cognitive_level=label,
                                        user_purpose=augmented_purpose,
                                    )
                                    lp_obj.setup_api()
                                    lp_obj.get_path()
                                    new_lp = lp_obj.get_learning_path()
                                    st.session_state.chat_learning_path = new_lp
                                    st.session_state.chat_topic_data = lp_obj._topic_data
                                    st.session_state.chat_regen_count += 1
                                    st.session_state.chat_show_regen_input = False
                                    # reset downstream
                                    st.session_state.chat_generated_content = None
                                    st.session_state.chat_show_content = False
                                except Exception as e:
                                    st.error(f"⚠️ Regeneration error: {e}")
                            st.rerun()
                    with rc2:
                        if st.button("Cancel", key="cancel_regen"):
                            st.session_state.chat_show_regen_input = False
                            st.rerun()
            else:
                st.warning("⚠️ All 3 regeneration attempts used. Proceed to generate content below.")

            # ────────────────────────────────────────────────
            #  STEP 4: generate personalised content
            # ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<span class="step-badge">STEP 4</span> **Generate Personalised Learning Content**',
                        unsafe_allow_html=True)

            st.write(f"**Cognitive Level:** {label.capitalize()}")
            st.write(f"**Your Goal:** {st.session_state.chat_purpose}")

            if st.button("🎨 Generate My Learning Content", type="primary",
                         key="gen_content", use_container_width=True):
                st.session_state.chat_show_content = True
                with st.spinner("✨ Your personal tutor is crafting your content…"):
                    try:
                        lp_str = json.dumps(st.session_state.chat_learning_path, indent=2)
                        content_prompt = f"""You are an expert educational content creator.
Generate comprehensive, high-quality educational content for the following query.

Query: {st.session_state.chat_query}
Cognitive Level: {label.upper()} (Bloom's Taxonomy)
User's Learning Goal: {st.session_state.chat_purpose}

Learning Path:
{lp_str}

Instructions:
1. Follow the learning path stages in sequence when structuring your content.
2. Tailor depth and complexity to the {label} level of Bloom's Taxonomy.
3. Ensure the content directly helps the user achieve: {st.session_state.chat_purpose}
4. For each stage, provide:
   - Clear explanation of concepts
   - Worked examples or demonstrations where relevant
   - A short activity or reflection question tied to the stage objective
5. Use a friendly, educational tone throughout.
6. End with a concise summary and suggested next steps beyond this learning path.

Generate the full educational content now:"""

                        client = OpenAI(
                            base_url="https://api.groq.com/openai/v1",
                            api_key=os.environ.get(f"{api_provider.upper()}_API_KEY"),
                        )
                        resp = client.chat.completions.create(
                            model=content_model,
                            temperature=random.uniform(0.5, 0.8),
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an expert personal tutor who tailors content "
                                        "to Bloom's Taxonomy levels, user goals, and structured "
                                        "learning paths. Write engaging, clear, comprehensive material."
                                    ),
                                },
                                {"role": "user", "content": content_prompt},
                            ],
                        )
                        st.session_state.chat_generated_content = resp.choices[0].message.content
                    except Exception as e:
                        st.error(f"⚠️ Content generation error: {e}")
                        st.session_state.chat_generated_content = None
                st.rerun()

            # ── display generated content ─────────────────
            if (st.session_state.chat_show_content
                    and st.session_state.chat_generated_content):

                content = st.session_state.chat_generated_content
                word_count = len(content.split())
                reading_time = max(1, word_count // 200)

                st.markdown("---")
                st.subheader("📚 Your Personalised Learning Content")

                # metrics row
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("📊 Cognitive Level", label.capitalize())
                mc2.metric("📝 Content Length", f"{word_count:,} words")
                mc3.metric("⏱️ Reading Time", f"~{reading_time} min")

                st.markdown(
                    f'<div class="content-card">{content}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                # ── download buttons ───────────────────────
                dc1, dc2, dc3 = st.columns(3)

                with dc1:
                    st.download_button(
                        "📥 Download as TXT",
                        data=content,
                        file_name=f"learning_content_{label}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                with dc2:
                    if st.button("📄 Download as Word", key="dl_word", use_container_width=True):
                        st.session_state.generate_word = True
                        st.rerun()

                with dc3:
                    if st.button("📕 Download as PDF", key="dl_pdf", use_container_width=True):
                        st.session_state.generate_pdf = True
                        st.rerun()

                # ── Word export ────────────────────────────
                if st.session_state.generate_word:
                    with st.spinner("Creating Word document…"):
                        try:
                            import subprocess

                            docx_script = f"""
const {{ Document, Packer, Paragraph, TextRun, HeadingLevel }} = require('docx');
const fs = require('fs');
const content = {json.dumps(content)};
const query   = {json.dumps(st.session_state.chat_query)};
const level   = {json.dumps(label.capitalize())};
const purpose = {json.dumps(st.session_state.chat_purpose)};
const paras   = content.split('\\n').filter(p => p.trim() !== '');
const sections = [
    new Paragraph({{ text: "Educational Content", heading: HeadingLevel.HEADING_1, spacing: {{ after: 240 }} }}),
    new Paragraph({{ children: [new TextRun({{ text: "Query: ", bold: true }}), new TextRun(query)], spacing: {{ after: 120 }} }}),
    new Paragraph({{ children: [new TextRun({{ text: "Cognitive Level: ", bold: true }}), new TextRun(level)], spacing: {{ after: 120 }} }}),
    new Paragraph({{ children: [new TextRun({{ text: "Learning Goal: ", bold: true }}), new TextRun(purpose)], spacing: {{ after: 240 }} }}),
    new Paragraph({{ text: "Content", heading: HeadingLevel.HEADING_2, spacing: {{ before: 240, after: 120 }} }})
];
paras.forEach(p => sections.push(new Paragraph({{ text: p, spacing: {{ after: 120 }} }})));
const doc = new Document({{
    styles: {{ default: {{ document: {{ run: {{ font: "Arial", size: 24 }} }} }} }},
    sections: [{{ properties: {{ page: {{ size: {{ width: 12240, height: 15840 }}, margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }} }} }}, children: sections }}]
}});
Packer.toBuffer(doc).then(buf => {{ fs.writeFileSync('/home/claude/learning_content.docx', buf); }});
"""
                            with open("/home/claude/create_docx.js", "w") as f:
                                f.write(docx_script)
                            subprocess.run(["npm", "install", "-g", "docx"], capture_output=True, check=True)
                            subprocess.run(["node", "/home/claude/create_docx.js"], capture_output=True, check=True)

                            with open("/home/claude/learning_content.docx", "rb") as f:
                                docx_data = f.read()

                            st.download_button(
                                "✅ Click to Download Word Document",
                                data=docx_data,
                                file_name=f"learning_content_{label}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                type="primary",
                            )
                            st.session_state.generate_word = False

                        except Exception as e:
                            st.error(f"⚠️ Word export error: {e}")
                            st.session_state.generate_word = False

                # ── PDF export ─────────────────────────────
                if st.session_state.generate_pdf:
                    with st.spinner("Creating PDF document…"):
                        try:
                            from reportlab.lib.pagesizes import letter
                            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                            from reportlab.lib.units import inch
                            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                            from reportlab.lib.enums import TA_JUSTIFY

                            pdf_path = "/home/claude/learning_content.pdf"
                            doc_pdf = SimpleDocTemplate(pdf_path, pagesize=letter,
                                                        rightMargin=72, leftMargin=72,
                                                        topMargin=72, bottomMargin=18)
                            styles = getSampleStyleSheet()
                            title_s  = ParagraphStyle("CT", parent=styles["Heading1"],  fontSize=18, spaceAfter=12)
                            head_s   = ParagraphStyle("CH", parent=styles["Heading2"],  fontSize=14, spaceAfter=6)
                            body_s   = ParagraphStyle("CB", parent=styles["Normal"],    fontSize=11, alignment=TA_JUSTIFY, spaceAfter=12)
                            elems = [
                                Paragraph("Educational Content", title_s),
                                Spacer(1, .2 * inch),
                                Paragraph(f"<b>Query:</b> {st.session_state.chat_query}", body_s),
                                Paragraph(f"<b>Cognitive Level:</b> {label.capitalize()}", body_s),
                                Paragraph(f"<b>Learning Goal:</b> {st.session_state.chat_purpose}", body_s),
                                Spacer(1, .3 * inch),
                                Paragraph("Content", head_s),
                                Spacer(1, .1 * inch),
                            ]
                            for para in content.split("\n"):
                                if para.strip():
                                    elems.append(Paragraph(para, body_s))
                            doc_pdf.build(elems)

                            with open(pdf_path, "rb") as f:
                                pdf_data = f.read()

                            st.download_button(
                                "✅ Click to Download PDF",
                                data=pdf_data,
                                file_name=f"learning_content_{label}.pdf",
                                mime="application/pdf",
                                type="primary",
                            )
                            st.session_state.generate_pdf = False

                        except Exception as e:
                            st.error(f"⚠️ PDF export error: {e}")
                            st.session_state.generate_pdf = False

                # ── star feedback ──────────────────────────
                st.markdown("---")
                st.subheader("📣 How was this content?")
                fb_cols = st.columns(5)
                for i, stars in enumerate(["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], 1):
                    with fb_cols[i - 1]:
                        st.button(stars, key=f"star_{i}")

                # ── regenerate content ─────────────────────
                st.markdown("---")
                if st.button("🔄 Regenerate Content", key="regen_content", use_container_width=True):
                    st.session_state.chat_generated_content = None
                    st.session_state.chat_show_content = False
                    st.session_state.generate_word = False
                    st.session_state.generate_pdf = False
                    st.rerun()

                # ── save to history ────────────────────────
                entry = {
                    "query":             st.session_state.chat_query,
                    "label":             label,
                    "confidence":        st.session_state.chat_confidence or 1.0,
                    "persona":           persona,
                    "purpose":           st.session_state.chat_purpose,
                    "path_title":        st.session_state.chat_learning_path.get("learning_path_title", ""),
                    "generated_content": content,
                }
                if not any(h["query"] == entry["query"] and h.get("purpose") == entry["purpose"]
                           for h in st.session_state.classification_history):
                    st.session_state.classification_history.insert(0, entry)

    # ── new session button ────────────────────────────────────────
    if st.session_state.chat_label:
        st.markdown("---")
        if st.button("🔄 Start New Learning Session", key="new_session", use_container_width=True):
            for k in ["chat_query", "chat_label", "chat_confidence", "chat_votes",
                      "chat_purpose", "chat_purpose_confirmed", "chat_show_editor",
                      "chat_topic_data", "chat_learning_path", "chat_regen_count",
                      "chat_show_regen_input", "chat_generated_content",
                      "chat_show_content", "generate_word", "generate_pdf"]:
                st.session_state[k] = _defaults.get(k)
            st.rerun()

# ═════════════════════════════════════════════════════════════════
#  TAB 2 · HISTORY
# ═════════════════════════════════════════════════════════════════
with tab_history:
    st.header("📊 Session History")

    if st.session_state.classification_history:
        col_clear, _ = st.columns([1, 5])
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.classification_history = []
                st.rerun()

        history_df = pd.DataFrame(st.session_state.classification_history)
        display_cols = [c for c in ["query", "label", "confidence", "persona", "purpose", "path_title"]
                        if c in history_df.columns]
        st.dataframe(history_df[display_cols], use_container_width=True, hide_index=True)

        csv = history_df.to_csv(index=False)
        st.download_button("📥 Download History as CSV", data=csv,
                           file_name="learnpath_history.csv", mime="text/csv")

        st.subheader("Detailed View")
        for idx, item in enumerate(st.session_state.classification_history):
            with st.expander(f"{idx+1}. [{item['label'].capitalize()}] {item['query'][:60]}…"):
                st.write(f"**Query:** {item['query']}")
                st.write(f"**Bloom's Level:** {item['label'].capitalize()}")
                st.write(f"**Persona:** {item['persona'].capitalize()}")
                if item.get("purpose"):
                    st.write(f"**Purpose:** {item['purpose']}")
                if item.get("path_title"):
                    st.write(f"**Learning Path:** {item['path_title']}")
                if item.get("generated_content"):
                    st.markdown("**Generated Content:**")
                    st.markdown(
                        f'<div class="content-card">{item["generated_content"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "📥 Download Content",
                        data=item["generated_content"],
                        file_name=f"content_{idx+1}_{item['label']}.txt",
                        mime="text/plain",
                        key=f"hist_dl_{idx}",
                    )
    else:
        st.info("No sessions yet. Head to the **Chatbot** tab to begin!")

# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;padding:.8rem">
    <p>🎓 <strong>LearnPath AI</strong> · Built with Streamlit · Powered by Groq API</p>
    <p><small>Bloom's Taxonomy: A framework for categorising educational learning objectives</small></p>
</div>
""", unsafe_allow_html=True)