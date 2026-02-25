import streamlit as st
import pandas as pd
import os
import tempfile
import random

from package.classify_query import classifyQuery
from package.identify_purpose import identifyPurpose
from package.learning_path import learningPath

# Page config
st.set_page_config(page_title="Bloom's Taxonomy Classifier", page_icon="📚", layout="wide")

# Custom CSS (same as original)
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .taxonomy-badge { padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: bold; font-size: 1.5rem; text-align: center; margin: 1rem 0; }
    .purpose-box { padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; border-left: 4px solid #1f77b4; margin: 1rem 0; }
    .knowledge { background-color: #e3f2fd; color: #1565c0; }
    .comprehension { background-color: #e8f5e9; color: #2e7d32; }
    .application { background-color: #fff3e0; color: #ef6c00; }
    .analysis { background-color: #fce4ec; color: #c2185b; }
    .synthesis { background-color: #f3e5f5; color: #7b1fa2; }
    .evaluation { background-color: #ffebee; color: #c62828; }
    .stButton>button { width: 100%; background-color: #1f77b4; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classify_history' not in st.session_state:
    st.session_state.classify_history = []  # for Tab1
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []      # for Tab2
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# Chatbot session state
if 'chat_query' not in st.session_state:
    st.session_state.chat_query = None
if 'chat_current_label' not in st.session_state:
    st.session_state.chat_current_label = None
if 'chat_current_purpose' not in st.session_state:
    st.session_state.chat_current_purpose = None
if 'chat_purpose_confirmed' not in st.session_state:
    st.session_state.chat_purpose_confirmed = False
if 'chat_show_purpose_editor' not in st.session_state:
    st.session_state.chat_show_purpose_editor = False
if 'chat_generated_content' not in st.session_state:
    st.session_state.chat_generated_content = None
if 'chat_show_generated_content' not in st.session_state:
    st.session_state.chat_show_generated_content = False
if 'chat_learning_path' not in st.session_state:
    st.session_state.chat_learning_path = None
if 'chat_purpose_identifier' not in st.session_state:
    st.session_state.chat_purpose_identifier = None
if 'generate_word' not in st.session_state:
    st.session_state.generate_word = False
if 'generate_pdf' not in st.session_state:
    st.session_state.generate_pdf = False

# Sidebar (same as original)
with st.sidebar:
    st.header("⚙️ Configuration")
    api_provider = st.selectbox("API Provider", ["groq"])
    api_key = st.text_input("API Key", type="password", help="Enter your API key or set GROQ_API_KEY environment variable")
    if api_key:
        os.environ[f"{api_provider.upper()}_API_KEY"] = api_key
    if os.environ.get(f"{api_provider.upper()}_API_KEY") is None:
        st.error("⚠️ API Key not found")
        st.info(f"Set {api_provider.upper()}_API_KEY environment variable")

    model_name = st.selectbox("Primary Model", options=['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'meta-llama/llama-4-scout-17b-16e-instruct'])
    correction_model = st.selectbox("Correction Model", options=['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'meta-llama/llama-4-scout-17b-16e-instruct'])
    persona = st.selectbox("Select Persona", ["multi", "professor", "student", "psychologist", "engineer", "examiner"])
    st.info({
        "multi": "Uses all personas and majority voting",
        "professor": "University professor perspective",
        "student": "Student perspective",
        "psychologist": "Clinical approach",
        "engineer": "Technical focus",
        "examiner": "Assessment specialist"
    }[persona])

    st.header("📖 Bloom's Taxonomy Levels")
    with st.expander("View All Levels"):
        st.markdown("""
        1. **Knowledge**: Recalling facts
        2. **Comprehension**: Understanding
        3. **Application**: Using in new situations
        4. **Analysis**: Breaking down
        5. **Synthesis**: Creating new ideas
        6. **Evaluation**: Making judgments
        """)

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Classify Query", "💬 Chatbot", "📊 History"])

# ---------- Tab 1: Classify Query (Single + Batch) ----------
with tab1:
    st.header("🔍 Query Classification")
    mode = st.radio("Choose mode:", ["Single Query", "Batch Processing"], horizontal=True)

    if mode == "Single Query":
        query = st.text_area("Enter your educational query:", height=150, placeholder="Type your question here...")
        if st.button("🚀 Classify", type="primary"):
            if not query.strip():
                st.error("Please enter a query.")
            else:
                with st.spinner("Classifying..."):
                    try:
                        classifier = classifyQuery(
                            api_provider=api_provider,
                            persona=persona,
                            query=query,
                            model_name=model_name,
                            correction_model=correction_model
                        )
                        if persona == 'multi':
                            result = classifier.get_ensemble_label()
                            label = result['final_label']
                            confidence = result['confidence']
                        else:
                            label = classifier.get_label()
                            confidence = 1.0

                        # Store in classify_history
                        st.session_state.classify_history.insert(0, {
                            'query': query,
                            'label': label,
                            'confidence': confidence,
                            'persona': persona
                        })

                        st.success("Classification Complete!")
                        st.markdown(f'<div class="taxonomy-badge {label}">{label.upper()}</div>', unsafe_allow_html=True)
                        taxonomy_info = {
                            'knowledge': 'Recalling facts, terms, basic concepts',
                            'comprehension': 'Demonstrating understanding by explaining',
                            'application': 'Using information in new situations',
                            'analysis': 'Breaking down information',
                            'synthesis': 'Combining elements to form a new whole',
                            'evaluation': 'Making judgments based on criteria'
                        }
                        st.info(f"**Description:** {taxonomy_info.get(label, 'N/A')}")
                        if persona == 'multi':
                            st.metric("Confidence", f"{confidence:.1%}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    else:  # Batch Processing
        st.markdown("Upload a CSV or Excel file with queries.")
        uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"File uploaded! {len(df)} rows.")
                st.dataframe(df.head(10))
                query_column = st.selectbox("Select column containing queries:", df.columns.tolist())
                if st.button("🚀 Process Batch", type="primary"):
                    if not query_column:
                        st.error("Select a column.")
                    else:
                        progress_bar = st.progress(0)
                        status = st.empty()
                        results = []
                        total = len(df)
                        for idx, row in df.iterrows():
                            q = str(row[query_column])
                            status.text(f"Processing {idx+1}/{total}: {q[:50]}...")
                            try:
                                classifier = classifyQuery(
                                    api_provider=api_provider,
                                    persona=persona,
                                    query=q,
                                    model_name=model_name,
                                    correction_model=correction_model
                                )
                                if persona == 'multi':
                                    result = classifier.get_ensemble_label()
                                    label = result['final_label']
                                    conf = result['confidence']
                                else:
                                    label = classifier.get_label()
                                    conf = 1.0
                                results.append({'query': q, 'label': label, 'confidence': conf})
                            except Exception as e:
                                results.append({'query': q, 'label': 'ERROR', 'confidence': 0.0})
                            progress_bar.progress((idx+1)/total)
                        st.session_state.batch_results = pd.DataFrame(results)
                        status.text("Batch processing complete!")

                if st.session_state.batch_results is not None:
                    st.subheader("Batch Results")
                    st.dataframe(st.session_state.batch_results)
                    csv = st.session_state.batch_results.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "batch_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

# ---------- Tab 2: Chatbot (fixed) ----------
with tab2:
    st.header("💬 AI Educational Assistant")
    st.info("Get personalized educational content.")

    # Query input
    query_chat = st.text_area("Ask your educational question:", height=150, placeholder="Type your question...", key="chat_input")
    if st.button("🚀 Start Learning Journey", type="primary", key="start_chat"):
        if not query_chat.strip():
            st.error("Please enter a question.")
        else:
            # Reset relevant states
            st.session_state.chat_query = query_chat
            st.session_state.chat_purpose_confirmed = False
            st.session_state.chat_current_purpose = None
            st.session_state.chat_show_generated_content = False
            st.session_state.chat_generated_content = None
            st.session_state.chat_learning_path = None
            st.session_state.chat_purpose_identifier = None
            st.rerun()

    # If query exists, proceed with classification and purpose
    if st.session_state.chat_query:
        # Step 1: Classify
        if st.session_state.chat_current_label is None:
            with st.spinner("Classifying cognitive level..."):
                classifier = classifyQuery(
                    api_provider=api_provider,
                    persona=persona,
                    query=st.session_state.chat_query,
                    model_name=model_name,
                    correction_model=correction_model
                )
                if persona == 'multi':
                    result = classifier.get_ensemble_label()
                    label = result['final_label']
                else:
                    label = classifier.get_label()
                st.session_state.chat_current_label = label
            st.rerun()

        # Display classification
        st.markdown("---")
        st.subheader("📊 Step 1: Cognitive Level")
        label = st.session_state.chat_current_label
        st.markdown(f'<div class="taxonomy-badge {label}">{label.upper()}</div>', unsafe_allow_html=True)

        # Step 2: Identify purpose (if not yet)
        if st.session_state.chat_current_purpose is None:
            with st.spinner("Identifying your intent..."):
                try:
                    purpose_id = identifyPurpose(
                        api_provider=api_provider,
                        query=st.session_state.chat_query,
                        model_name=model_name,
                        cognitive_level=label
                    )
                    purpose_id.setup_api()
                    purpose = purpose_id.get_purpose()
                    st.session_state.chat_current_purpose = purpose
                    st.session_state.chat_purpose_identifier = purpose_id
                except Exception as e:
                    st.error(f"Error identifying purpose: {str(e)}")
            st.rerun()

        # Purpose confirmation
        if not st.session_state.chat_purpose_confirmed:
            st.markdown("---")
            st.subheader("🎯 Step 2: Your Intent")
            st.markdown(f'<div class="purpose-box"><strong>Identified Goal:</strong> {st.session_state.chat_current_purpose}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's correct", key="confirm_purpose"):
                    st.session_state.chat_purpose_confirmed = True
                    st.rerun()
            with col2:
                if st.button("✏️ No, let me clarify", key="edit_purpose"):
                    st.session_state.chat_show_purpose_editor = True

            if st.session_state.chat_show_purpose_editor:
                edited = st.text_area("Clarify your goal:", value=st.session_state.chat_current_purpose, height=100)
                if st.button("💾 Save", key="save_purpose"):
                    st.session_state.chat_current_purpose = edited
                    st.session_state.chat_purpose_confirmed = True
                    st.session_state.chat_show_purpose_editor = False
                    st.rerun()

        # After confirmation: generate learning path
        if st.session_state.chat_purpose_confirmed and st.session_state.chat_learning_path is None:
            with st.spinner("Building your learning path..."):
                try:
                    lp = learningPath(
                        api_provider=api_provider,
                        query=st.session_state.chat_query,
                        model_name=model_name,
                        cognitive_level=st.session_state.chat_current_label,
                        user_purpose=st.session_state.chat_current_purpose
                    )
                    lp.setup_api()
                    lp.get_path()  # sets _topic_data
                    path = lp.get_learning_path()
                    st.session_state.chat_learning_path = path
                except Exception as e:
                    st.error(f"Error generating learning path: {str(e)}")
            st.rerun()

        # Display learning path
        if st.session_state.chat_learning_path:
            st.markdown("---")
            st.subheader("🗺️ Step 3: Your Learning Path")
            lp = st.session_state.chat_learning_path
            st.markdown(f"### {lp.get('learning_path_title', 'Learning Path')}")
            for stage in lp.get('stages', []):
                with st.expander(f"Stage {stage.get('stage_number')}: {stage.get('title')}"):
                    st.write("**Objectives:**")
                    for obj in stage.get('objectives', []):
                        st.markdown(f"- {obj}")

        # Step 4: Content generation
        st.markdown("---")
        st.subheader("✨ Step 4: Generate Content")
        content_generation_model = st.selectbox("Content Generation Model", [
            'openai/gpt-oss-120b',
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'llama-3.3-70b-versatile',
            'qwen/qwen3-32b'
        ], key="content_model")

        if st.button("🎨 Generate My Learning Content", type="primary",
                     key="chat_generate_content", use_container_width=True):
            st.session_state.chat_show_generated_content = True

            with st.spinner("✨ Creating personalized educational content for you..."):
                try:
                    content_prompt = f"""You are an expert educational content creator. \
Generate comprehensive, high-quality educational content for the following query.

Query: {st.session_state.chat_query}

Cognitive Level: {st.session_state.chat_current_label.upper()} (Bloom's Taxonomy)
User's Learning Goal: {st.session_state.chat_current_purpose}

Learning Path:
{st.session_state.chat_learning_path}

Instructions:
1. Follow the learning path stages above in sequence when structuring your content.
2. Tailor depth and complexity to the {st.session_state.chat_current_label} level of Bloom's Taxonomy.
3. Ensure the content directly helps the user achieve: {st.session_state.chat_current_purpose}
4. For each stage in the learning path, provide a dedicated section with:
   - Clear explanation of concepts
   - Worked examples or demonstrations where relevant
   - A short activity or reflection question tied to the stage milestone
5. Use a friendly, educational tone throughout.
6. End with a summary and suggested next steps beyond the learning path.

Generate comprehensive educational content now:"""

                    from openai import OpenAI

                    client = OpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=os.environ.get(f"{api_provider.upper()}_API_KEY")
                    )

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert educational content creator who tailors content "
                                    "based on Bloom's Taxonomy levels, user learning goals, and structured "
                                    "learning paths. You create engaging, clear, and comprehensive educational materials."
                                )
                            },
                            {
                                "role": "user",
                                "content": content_prompt
                            }
                        ],
                        model=content_generation_model,
                        temperature=random.uniform(5, 8) / 10
                    )

                    generated_content = chat_completion.choices[0].message.content
                    st.session_state.chat_generated_content = generated_content

                except Exception as e:
                    st.error(f"⚠️ Error generating content: {str(e)}")
                    st.session_state.chat_generated_content = None

        # ── Display generated content ─────────────────────────────────────────
        if st.session_state.get('chat_show_generated_content', False) and st.session_state.get('chat_generated_content'):
            st.markdown("---")
            st.subheader("📚 Your Personalized Learning Content")

            with st.container():
                st.markdown(st.session_state.chat_generated_content)

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Cognitive Level", st.session_state.chat_current_label.capitalize())
            with col2:
                word_count = len(st.session_state.chat_generated_content.split())
                st.metric("📝 Content Length", f"{word_count} words")
            with col3:
                reading_time = max(1, word_count // 200)
                st.metric("⏱️ Reading Time", f"~{reading_time} min")

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=st.session_state.chat_generated_content,
                    file_name=f"learning_content_{st.session_state.chat_current_label}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col2:
                if st.button("📄 Download as Word", key="chat_download_word", use_container_width=True):
                    st.session_state.generate_word = True
                    st.rerun()

            with col3:
                if st.button("📕 Download as PDF", key="chat_download_pdf", use_container_width=True):
                    st.session_state.generate_pdf = True
                    st.rerun()

            # ── Word export ───────────────────────────────────────────────────
            if st.session_state.get('generate_word', False):
                with st.spinner("Creating Word document..."):
                    try:
                        import subprocess, json

                        docx_script = f"""
const {{ Document, Packer, Paragraph, TextRun, HeadingLevel }} = require('docx');
const fs = require('fs');

const content = {json.dumps(st.session_state.chat_generated_content)};
const query   = {json.dumps(st.session_state.chat_query)};
const level   = {json.dumps(st.session_state.chat_current_label.capitalize())};
const purpose = {json.dumps(st.session_state.chat_current_purpose)};

const paragraphs = content.split('\\n').filter(p => p.trim() !== '');

const sections = [
    new Paragraph({{ text: "Educational Content", heading: HeadingLevel.HEADING_1, spacing: {{ after: 240 }} }}),
    new Paragraph({{ children: [new TextRun({{ text: "Query: ", bold: true }}), new TextRun(query)], spacing: {{ after: 120 }} }}),
    new Paragraph({{ children: [new TextRun({{ text: "Cognitive Level: ", bold: true }}), new TextRun(level)], spacing: {{ after: 120 }} }}),
    new Paragraph({{ children: [new TextRun({{ text: "Learning Goal: ", bold: true }}), new TextRun(purpose)], spacing: {{ after: 240 }} }}),
    new Paragraph({{ text: "Content", heading: HeadingLevel.HEADING_2, spacing: {{ before: 240, after: 120 }} }})
];

paragraphs.forEach(para => {{
    sections.push(new Paragraph({{ text: para, spacing: {{ after: 120 }} }}));
}});

const doc = new Document({{
    styles: {{ default: {{ document: {{ run: {{ font: "Arial", size: 24 }} }} }} }},
    sections: [{{ properties: {{ page: {{ size: {{ width: 12240, height: 15840 }}, margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }} }} }}, children: sections }}]
}});

Packer.toBuffer(doc).then(buffer => {{
    fs.writeFileSync('/home/claude/learning_content.docx', buffer);
    console.log('Document created successfully');
}});
"""
                        with open('/home/claude/create_docx.js', 'w') as f:
                            f.write(docx_script)

                        subprocess.run(['npm', 'install', '-g', 'docx'], capture_output=True, check=True)
                        subprocess.run(['node', '/home/claude/create_docx.js'], capture_output=True, text=True, check=True)

                        with open('/home/claude/learning_content.docx', 'rb') as f:
                            docx_data = f.read()

                        st.download_button(
                            label="✅ Click to Download Word Document",
                            data=docx_data,
                            file_name=f"learning_content_{st.session_state.chat_current_label}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary"
                        )
                        st.session_state.generate_word = False

                    except Exception as e:
                        st.error(f"Error creating Word document: {str(e)}")
                        st.session_state.generate_word = False

            # ── PDF export ────────────────────────────────────────────────────
            if st.session_state.get('generate_pdf', False):
                with st.spinner("Creating PDF document..."):
                    try:
                        from reportlab.lib.pagesizes import letter
                        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                        from reportlab.lib.units import inch
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                        from reportlab.lib.enums import TA_JUSTIFY

                        pdf_path = '/home/claude/learning_content.pdf'
                        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                                                rightMargin=72, leftMargin=72,
                                                topMargin=72, bottomMargin=18)

                        elements = []
                        styles = getSampleStyleSheet()

                        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                                     fontSize=18, spaceAfter=12)
                        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
                                                       fontSize=14, spaceAfter=6)
                        body_style = ParagraphStyle('CustomBody', parent=styles['Normal'],
                                                    fontSize=11, alignment=TA_JUSTIFY, spaceAfter=12)

                        elements.append(Paragraph("Educational Content", title_style))
                        elements.append(Spacer(1, 0.2 * inch))
                        elements.append(Paragraph(f"<b>Query:</b> {st.session_state.chat_query}", body_style))
                        elements.append(Paragraph(f"<b>Cognitive Level:</b> {st.session_state.chat_current_label.capitalize()}", body_style))
                        elements.append(Paragraph(f"<b>Learning Goal:</b> {st.session_state.chat_current_purpose}", body_style))
                        elements.append(Spacer(1, 0.3 * inch))
                        elements.append(Paragraph("Content", heading_style))
                        elements.append(Spacer(1, 0.1 * inch))

                        for para in st.session_state.chat_generated_content.split('\n'):
                            if para.strip():
                                elements.append(Paragraph(para, body_style))

                        doc.build(elements)

                        with open(pdf_path, 'rb') as f:
                            pdf_data = f.read()

                        st.download_button(
                            label="✅ Click to Download PDF Document",
                            data=pdf_data,
                            file_name=f"learning_content_{st.session_state.chat_current_label}.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                        st.session_state.generate_pdf = False

                    except Exception as e:
                        st.error(f"Error creating PDF: {str(e)}")
                        st.session_state.generate_pdf = False

            # ── Regenerate ────────────────────────────────────────────────────
            st.markdown("---")
            if st.button("🔄 Regenerate Content", key="chat_regenerate", use_container_width=True):
                st.session_state.chat_show_generated_content = False
                st.session_state.chat_generated_content = None
                st.session_state.generate_word = False
                st.session_state.generate_pdf = False
                st.rerun()

            # ── History ───────────────────────────────────────────────────────
            if st.session_state.chat_current_label:
                history_entry = {
                    'query':             st.session_state.chat_query,
                    'label':             st.session_state.chat_current_label,
                    'confidence':        1.0,
                    'persona':           persona,
                    'purpose':           st.session_state.chat_current_purpose,
                    'learning_path':     st.session_state.get('chat_learning_path'),
                    'generated_content': st.session_state.chat_generated_content
                }

                if not any(h['query'] == history_entry['query'] and
                           h.get('purpose') == history_entry['purpose']
                           for h in st.session_state.classification_history):
                    st.session_state.classification_history.insert(0, history_entry)

            # ── Feedback ──────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📣 How was this content?")
            feedback_cols = st.columns(5)
            for i, stars in enumerate(["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], 1):
                with feedback_cols[i - 1]:
                    st.button(stars, key=f"feedback_{i}")

            # ── New session ───────────────────────────────────────────────────
            st.markdown("---")
            if st.button("🔄 Start New Learning Session", key="chat_new_session", use_container_width=True):
                for key in [
                    'chat_query', 'chat_current_label', 'chat_current_purpose',
                    'chat_purpose_confirmed', 'chat_show_purpose_editor',
                    'chat_show_generated_content', 'chat_generated_content',
                    'chat_learning_path', 'chat_purpose_identifier',
                    'generate_word', 'generate_pdf'
                ]:
                    st.session_state[key] = None if 'show' not in key and 'confirmed' not in key else False
                st.rerun()

# ---------- Tab 3: History (separate for classify and chat) ----------
with tab3:
    st.header("📊 History")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Classification History")
        if st.session_state.classify_history:
            df_class = pd.DataFrame(st.session_state.classify_history)
            st.dataframe(df_class[['query', 'label', 'confidence', 'persona']], use_container_width=True)
            csv_class = df_class.to_csv(index=False)
            st.download_button("📥 Download Classify History", csv_class, "classify_history.csv", "text/csv")
        else:
            st.info("No classification history yet.")

    with col2:
        st.subheader("💬 Chatbot History")
        if st.session_state.chat_history:
            df_chat = pd.DataFrame(st.session_state.chat_history)
            # Show only basic info; content can be expanded
            st.dataframe(df_chat[['query', 'label', 'purpose']], use_container_width=True)
            csv_chat = df_chat.to_csv(index=False)
            st.download_button("📥 Download Chat History", csv_chat, "chat_history.csv", "text/csv")

            # Detailed expanders
            st.markdown("**Detailed View**")
            for idx, item in enumerate(st.session_state.chat_history):
                with st.expander(f"{idx+1}. {item['label'].capitalize()} - {item['query'][:50]}..."):
                    st.write(f"**Query:** {item['query']}")
                    st.write(f"**Purpose:** {item['purpose']}")
                    if item.get('generated_content'):
                        st.markdown("**Generated Content:**")
                        st.markdown(item['generated_content'])
        else:
            st.info("No chat history yet.")

    # Clear history buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Classification History"):
            st.session_state.classify_history = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Footer (same)
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with Streamlit | Powered by Groq API</div>", unsafe_allow_html=True)