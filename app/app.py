import streamlit as st
from package.classify_query import classifyQuery
from package.identify_purpose import identifyPurpose
import pandas as pd
import io
import os
import random

# Page configuration
st.set_page_config(
    page_title="Bloom's Taxonomy Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .taxonomy-badge {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .purpose-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .knowledge { background-color: #e3f2fd; color: #1565c0; }
    .comprehension { background-color: #e8f5e9; color: #2e7d32; }
    .application { background-color: #fff3e0; color: #ef6c00; }
    .analysis { background-color: #fce4ec; color: #c2185b; }
    .synthesis { background-color: #f3e5f5; color: #7b1fa2; }
    .evaluation { background-color: #ffebee; color: #c62828; }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'current_purpose' not in st.session_state:
    st.session_state.current_purpose = None
if 'purpose_confirmed' not in st.session_state:
    st.session_state.purpose_confirmed = False
if 'current_label' not in st.session_state:
    st.session_state.current_label = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = None
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = None
if 'show_generated_content' not in st.session_state:
    st.session_state.show_generated_content = False

# Chat tab session state
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
if 'generate_word' not in st.session_state:
    st.session_state.generate_word = False
if 'generate_pdf' not in st.session_state:
    st.session_state.generate_pdf = False

# Header
st.markdown('<div class="main-header">📚 Bloom\'s Taxonomy Query Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify queries using Generative AI-powered cognitive level analysis</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Provider
    api_provider = st.selectbox(
        "API Provider",
        ["groq"],
        help="Select your API provider"
    )
    
    # API Key input
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your API key or set GROQ_API_KEY environment variable",
        placeholder="Enter API key (optional if env var is set)"
    )
    
    if api_key:
        os.environ[f"{api_provider.upper()}_API_KEY"] = api_key
    
    if os.environ.get(f"{api_provider.upper()}_API_KEY") is None:
        st.error("⚠️ API Key not found")
        st.info(f"Set {api_provider.upper()}_API_KEY environment variable")

    # Model selection
    model_name = st.selectbox(
            "Primary Model",
            options=['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'meta-llama/llama-4-scout-17b-16e-instruct'],
            help="Model for classification"
        )
    
    correction_model = st.selectbox(
            "Correction Model",
            options=['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'meta-llama/llama-4-scout-17b-16e-instruct'],
            help="Model for self-correction"
        )
    
    # Persona selection
    st.header("🎭 Classification Persona")
    persona = st.selectbox(
        "Select Persona",
        [
            "multi",
            "professor",
            "student",
            "psychologist",
            "engineer",
            "examiner"
        ],
        help="Choose the persona for classification. 'multi' uses ensemble voting."
    )
    
    persona_descriptions = {
        "multi": "Uses all personas and majority voting for robust classification",
        "professor": "University professor with deep educational expertise",
        "student": "Student perspective on learning objectives",
        "psychologist": "Clinical approach to cognitive assessment",
        "engineer": "Technical and practical problem-solving focus",
        "examiner": "Assessment and evaluation specialist"
    }
    
    st.info(persona_descriptions[persona])
    
    # Taxonomy Information
    st.header("📖 Bloom's Taxonomy Levels")
    with st.expander("View All Levels"):
        st.markdown("""
        1. **Knowledge**: Recalling facts, terms, basic concepts
        2. **Comprehension**: Understanding and explaining information
        3. **Application**: Using information in new situations
        4. **Analysis**: Breaking down and examining information
        5. **Synthesis**: Creating new ideas from existing information
        6. **Evaluation**: Making judgments based on criteria
        """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify", "📦 Batch Processing", "💬 Chat-Bot", "📊 History"])

# Tab 1: Single Query Classification
with tab1:
    st.header("🔍 Query Classification")
    st.info("💡 Classify your educational queries based on Bloom's Taxonomy cognitive levels.")
    
    # Query input
    query_classify = st.text_area(
        "Enter your educational query:",
        height=150,
        placeholder="Type your question here...",
        help="Enter any educational question to classify",
        key="query_classify_tab1"
    )
    
    # Classify button
    classify_button = st.button("🚀 Classify Query", type="primary", key="classify_tab1")
    
    if classify_button:
        if not query_classify.strip():
            st.error("⚠️ Please enter a query to classify")
        else:
            with st.spinner("🔄 Classifying query..."):
                try:
                    # Create classifier instance
                    classifier = classifyQuery(
                        api_provider=api_provider,
                        persona=persona,
                        query=query_classify,
                        model_name=model_name,
                        correction_model=correction_model
                    )
                    
                    # Get classification
                    if persona == 'multi':
                        result = classifier.get_ensemble_label()
                        label = result['final_label']
                        confidence = result['confidence']
                    else:
                        label = classifier.get_label()
                        confidence = 1.0
                    
                    # Store in history
                    st.session_state.classification_history.insert(0, {
                        'query': query_classify,
                        'label': label,
                        'confidence': confidence,
                        'persona': persona
                    })
                    
                    # Display result
                    st.success("✅ Classification Complete!")
                    
                    st.markdown(f'<div class="taxonomy-badge {label}">{label.upper()}</div>', 
                                unsafe_allow_html=True)
                    
                    taxonomy_info = {
                        'knowledge': 'Recalling facts, terms, basic concepts, or answers',
                        'comprehension': 'Demonstrating understanding by interpreting, summarizing, or explaining',
                        'application': 'Using learned information in new concrete situations',
                        'analysis': 'Breaking down information into parts and examining relationships',
                        'synthesis': 'Combining elements to form a new whole or proposing solutions',
                        'evaluation': 'Making judgments based on criteria and standards'
                    }
                    
                    st.info(f"**Description:** {taxonomy_info.get(label, 'N/A')}")
                    
                    if persona == 'multi':
                        st.metric("Confidence Score", f"{confidence:.1%}")
                        st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"⚠️ Error during classification: {str(e)}")
                    st.info("💡 Make sure your API key is set correctly and the model names are valid.")

# Tab 2: Batch Processing
with tab2:
    st.header("📦 Batch Processing")
    st.markdown("Upload a CSV or Excel file with queries to classify multiple items at once.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="File should contain a column with queries to classify"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            # Display preview
            st.subheader("📋 File Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column selection
            query_column = st.selectbox(
                "Select the column containing queries:",
                options=df.columns.tolist(),
                help="Choose which column contains the text to classify"
            )
            
            # Process button
            if st.button("🚀 Process Batch", type="primary"):
                if not query_column:
                    st.error("⚠️ Please select a query column")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    total = len(df)
                    
                    for idx, row in df.iterrows():
                        query_text = str(row[query_column])
                        
                        # Update progress
                        progress = (idx + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {idx + 1}/{total}: {query_text[:50]}...")
                        
                        try:
                            # Create classifier instance
                            classifier = classifyQuery(
                                api_provider=api_provider,
                                persona=persona,
                                query=query_text,
                                model_name=model_name,
                                correction_model=correction_model
                            )
                            
                            # Get classification
                            if persona == 'multi':
                                result = classifier.get_ensemble_label()
                                label = result['final_label']
                                confidence = result['confidence']
                            else:
                                label = classifier.get_label()
                                confidence = 1.0
                            
                            result_entry = {
                                'query': query_text,
                                'label': label,
                                'confidence': confidence
                            }
                            
                            
                        except Exception as e:
                            result_entry = {
                                'query': query_text,
                                'label': 'ERROR',
                                'confidence': 0.0
                            }
                            results.append(result_entry)
                            st.warning(f"Error processing row {idx + 1}: {str(e)}")
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    st.session_state.batch_results = results_df
                    
                    status_text.text("✅ Batch processing complete!")
                    progress_bar.progress(1.0)
        
        except Exception as e:
            st.error(f"⚠️ Error reading file: {str(e)}")
    
    # Display results if available
    if st.session_state.batch_results is not None:
        st.subheader("📊 Batch Results")
        
        # Display results table
        st.dataframe(
            st.session_state.batch_results,
            use_container_width=True,
            hide_index=True
        )
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Processed", len(st.session_state.batch_results))
        with col2:
            avg_confidence = st.session_state.batch_results['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        with col3:
            error_count = len(st.session_state.batch_results[st.session_state.batch_results['label'] == 'ERROR'])
            st.metric("Errors", error_count)
        
        # Label distribution
        st.subheader("📈 Label Distribution")
        label_counts = st.session_state.batch_results['label'].value_counts()
        st.bar_chart(label_counts)
        
        # Show generated content if available
        if 'generated_content' in st.session_state.batch_results.columns:
            st.subheader("📖 View Generated Content")
            
            for idx, row in st.session_state.batch_results.iterrows():
                if row.get('generated_content') and row['generated_content'] not in ['N/A', 'Error']:
                    with st.expander(f"{idx+1}. {row['label'].capitalize()} - {row['query'][:60]}..."):
                        st.write(f"**Query:** {row['query']}")
                        st.write(f"**Level:** {row['label'].capitalize()}")
                        if 'purpose' in row and row['purpose'] != 'N/A':
                            st.write(f"**Purpose:** {row['purpose']}")
                        st.markdown("**Generated Content:**")
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 0.5rem;">
                        {row['generated_content']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.download_button(
                            label="📥 Download This Content",
                            data=row['generated_content'],
                            file_name=f"batch_content_{idx+1}.txt",
                            mime="text/plain",
                            key=f"batch_download_{idx}"
                        )
        
        # Download results
        csv_results = st.session_state.batch_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_results,
            file_name="batch_classification_results.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Clear results button
        if st.button("🗑️ Clear Batch Results"):
            st.session_state.batch_results = None
            st.rerun()

# Tab 3: ChatBot with Content Generation
with tab3:
    st.header("💬 AI Educational Assistant")
    st.info("🎯 Get personalized educational content tailored to your learning needs!")
    
    # Query input
    query_chat = st.text_area(
        "Ask your educational question:",
        height=150,
        placeholder="Type your question here...",
        help="Enter any educational question",
        key="query_chatbot_tab3"
    )
    
    # Start button
    start_button = st.button("🚀 Start Learning Journey", type="primary", key="start_chat")
    
    if start_button:
        if not query_chat.strip():
            st.error("⚠️ Please enter a question to begin")
        else:
            st.session_state.chat_query = query_chat
            st.session_state.chat_purpose_confirmed = False
            st.session_state.chat_current_purpose = None
            st.session_state.chat_show_generated_content = False
            st.session_state.chat_generated_content = None
            
            with st.spinner("🔄 Analyzing your question..."):
                try:
                    # Step 1: Classify cognitive level
                    st.markdown("---")
                    st.subheader("📊 Step 1: Cognitive Level Analysis")
                    
                    classifier = classifyQuery(
                        api_provider=api_provider,
                        persona=persona,
                        query=query_chat,
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
                    
                    st.session_state.chat_current_label = label
                    
                    st.markdown(f'<div class="taxonomy-badge {label}">{label.upper()}</div>', 
                                unsafe_allow_html=True)
                    
                    taxonomy_info = {
                        'knowledge': 'Recalling facts, terms, basic concepts, or answers',
                        'comprehension': 'Demonstrating understanding by interpreting, summarizing, or explaining',
                        'application': 'Using learned information in new concrete situations',
                        'analysis': 'Breaking down information into parts and examining relationships',
                        'synthesis': 'Combining elements to form a new whole or proposing solutions',
                        'evaluation': 'Making judgments based on criteria and standards'
                    }
                    
                    st.info(f"**Level Description:** {taxonomy_info.get(label, 'N/A')}")
                    
                    if persona == 'multi':
                        st.metric("Classification Confidence", f"{confidence:.1%}")
                    
                    # Step 2: Identify purpose
                    st.markdown("---")
                    st.subheader("🎯 Step 2: Understanding Your Intent")
                    
                    with st.spinner("🔍 Analyzing what you want to achieve..."):
                        try:
                            purpose_identifier = identifyPurpose(
                                api_provider=api_provider,
                                query=query_chat,
                                model_name=model_name,
                                cognitive_level=label
                            )
                            purpose_identifier.setup_api()
                            purpose = purpose_identifier.get_purpose()
                            st.session_state.chat_current_purpose = purpose
                            
                        except Exception as e:
                            st.error(f"⚠️ Error identifying purpose: {str(e)}")
                            purpose = None
                    
                except Exception as e:
                    st.error(f"⚠️ Error during analysis: {str(e)}")
                    st.info("💡 Make sure your API key is set correctly and the model names are valid.")
    
    # Display purpose and get feedback
    if st.session_state.get('chat_current_purpose') and not st.session_state.get('chat_purpose_confirmed', False):
        st.markdown("---")
        st.subheader("🎯 Identified Learning Goal")
        
        st.markdown(f'<div class="purpose-box"><strong>🎯 Your Intent:</strong> {st.session_state.chat_current_purpose}</div>', 
                    unsafe_allow_html=True)
        
        st.write("**Is this what you want to achieve?**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, that's correct", key="chat_confirm_purpose", use_container_width=True):
                st.session_state.chat_purpose_confirmed = True
                st.rerun()
        
        with col2:
            if st.button("✏️ No, let me clarify", key="chat_edit_purpose", use_container_width=True):
                st.session_state.chat_purpose_confirmed = False
                st.session_state.chat_show_purpose_editor = True
                st.rerun()
        
        # Show purpose editor if user wants to edit
        if st.session_state.get('chat_show_purpose_editor', False):
            st.markdown("**Clarify Your Learning Goal:**")
            edited_purpose = st.text_area(
                "What do you want to achieve or learn?",
                value=st.session_state.chat_current_purpose,
                height=100,
                help="Describe what you want to do with this information",
                key="chat_purpose_editor"
            )
            
            if st.button("💾 Save My Goal", key="chat_save_purpose"):
                st.session_state.chat_current_purpose = edited_purpose
                st.session_state.chat_purpose_confirmed = True
                st.session_state.chat_show_purpose_editor = False
                st.rerun()
    
    # Generate content after purpose confirmation
    if st.session_state.get('chat_purpose_confirmed', False) and st.session_state.get('chat_current_purpose'):
        st.markdown("---")
        st.subheader("✨ Step 3: Generate Personalized Content")
        
        st.write(f"**Cognitive Level:** {st.session_state.chat_current_label.capitalize()}")
        st.write(f"**Your Goal:** {st.session_state.chat_current_purpose}")


        content_generation_model = st.selectbox(
            "Cotent Generation Model",
            options=['openai/gpt-oss-120b', 'meta-llama/llama-4-scout-17b-16e-instruct', 
                     'llama-3.3-70b-versatile', 'qwen/qwen3-32b', 'moonshotai/kimi-k2-instruct-0905'],
            help="Model to generate content"
            )
        
        # Generate content button
        if st.button("🎨 Generate My Learning Content", type="primary", key="chat_generate_content", use_container_width=True):
            st.session_state.chat_show_generated_content = True
            
            with st.spinner("✨ Creating personalized educational content for you..."):
                try:
                    # Build content generation prompt
                    content_prompt = f"""You are an expert educational content creator. Generate comprehensive, high-quality educational content for the following query.

Query: {st.session_state.chat_query}

Cognitive Level: {st.session_state.chat_current_label.upper()} (Bloom's Taxonomy)
User's Learning Goal: {st.session_state.chat_current_purpose}

Instructions:
1. Tailor the content to the {st.session_state.chat_current_label} level of Bloom's Taxonomy
2. Ensure the content helps achieve the user's specific learning goal: {st.session_state.chat_current_purpose}
3. Make it clear, engaging, and appropriate for this cognitive level
4. Include relevant examples, explanations, step-by-step guidance, or practice as needed
5. Use a friendly, educational tone

Generate comprehensive educational content now:"""

                    # Use the API to generate content
                    from openai import OpenAI
                    
                    client = OpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=os.environ.get(f"{api_provider.upper()}_API_KEY")
                    )
                    
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert educational content creator who tailors content based on Bloom's Taxonomy levels and user learning goals. You create engaging, clear, and comprehensive educational materials."
                            },
                            {
                                "role": "user",
                                "content": content_prompt
                            }
                        ],
                        model=content_generation_model,
                        temperature=random.uniform(5,8)/10
                    )
                    
                    generated_content = chat_completion.choices[0].message.content
                    st.session_state.chat_generated_content = generated_content
                    
                except Exception as e:
                    st.error(f"⚠️ Error generating content: {str(e)}")
                    st.session_state.chat_generated_content = None
        
        # Display generated content
        if st.session_state.get('chat_show_generated_content', False) and st.session_state.get('chat_generated_content'):
            st.markdown("---")
            st.subheader("📚 Your Personalized Learning Content")
            
            # Display content in an expandable box
            with st.container():
                st.markdown(st.session_state.chat_generated_content)
            
            st.markdown("---")
            
            # Content metadata and actions
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
            
            # Generate Word document if requested
            if st.session_state.get('generate_word', False):
                with st.spinner("Creating Word document..."):
                    try:
                        import subprocess
                        import json
                        
                        # Create Node.js script to generate DOCX
                        docx_script = f"""
const {{ Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType }} = require('docx');
const fs = require('fs');

const content = {json.dumps(st.session_state.chat_generated_content)};
const query = {json.dumps(st.session_state.chat_query)};
const level = {json.dumps(st.session_state.chat_current_label.capitalize())};
const purpose = {json.dumps(st.session_state.chat_current_purpose)};

// Split content into paragraphs
const paragraphs = content.split('\\n').filter(p => p.trim() !== '');

// Create document sections
const sections = [
    new Paragraph({{
        text: "Educational Content",
        heading: HeadingLevel.HEADING_1,
        spacing: {{ after: 240 }}
    }}),
    new Paragraph({{
        children: [
            new TextRun({{ text: "Query: ", bold: true }}),
            new TextRun(query)
        ],
        spacing: {{ after: 120 }}
    }}),
    new Paragraph({{
        children: [
            new TextRun({{ text: "Cognitive Level: ", bold: true }}),
            new TextRun(level)
        ],
        spacing: {{ after: 120 }}
    }}),
    new Paragraph({{
        children: [
            new TextRun({{ text: "Learning Goal: ", bold: true }}),
            new TextRun(purpose)
        ],
        spacing: {{ after: 240 }}
    }}),
    new Paragraph({{
        text: "Content",
        heading: HeadingLevel.HEADING_2,
        spacing: {{ before: 240, after: 120 }}
    }})
];

// Add content paragraphs
paragraphs.forEach(para => {{
    sections.push(new Paragraph({{
        text: para,
        spacing: {{ after: 120 }}
    }}));
}});

const doc = new Document({{
    styles: {{
        default: {{
            document: {{
                run: {{ font: "Arial", size: 24 }}
            }}
        }}
    }},
    sections: [{{
        properties: {{
            page: {{
                size: {{
                    width: 12240,
                    height: 15840
                }},
                margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }}
            }}
        }},
        children: sections
    }}]
}});

Packer.toBuffer(doc).then(buffer => {{
    fs.writeFileSync('/home/claude/learning_content.docx', buffer);
    console.log('Document created successfully');
}});
"""
                        
                        # Write script to file
                        with open('/home/claude/create_docx.js', 'w') as f:
                            f.write(docx_script)
                        
                        # Install docx if not already installed
                        subprocess.run(['npm', 'install', '-g', 'docx'], 
                                     capture_output=True, check=True)
                        
                        # Run the script
                        result = subprocess.run(['node', '/home/claude/create_docx.js'], 
                                              capture_output=True, text=True, check=True)
                        
                        # Read the generated file
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
                        st.error(f"Error creating Word document: {{str(e)}}")
                        st.session_state.generate_word = False
            
            # Generate PDF if requested
            if st.session_state.get('generate_pdf', False):
                with st.spinner("Creating PDF document..."):
                    try:
                        from reportlab.lib.pagesizes import letter
                        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                        from reportlab.lib.units import inch
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                        from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
                        
                        # Create PDF
                        pdf_path = '/home/claude/learning_content.pdf'
                        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                                              rightMargin=72, leftMargin=72,
                                              topMargin=72, bottomMargin=18)
                        
                        # Container for elements
                        elements = []
                        styles = getSampleStyleSheet()
                        
                        # Custom styles
                        title_style = ParagraphStyle(
                            'CustomTitle',
                            parent=styles['Heading1'],
                            fontSize=18,
                            textColor='#1f77b4',
                            spaceAfter=12
                        )
                        
                        heading_style = ParagraphStyle(
                            'CustomHeading',
                            parent=styles['Heading2'],
                            fontSize=14,
                            spaceAfter=6
                        )
                        
                        body_style = ParagraphStyle(
                            'CustomBody',
                            parent=styles['Normal'],
                            fontSize=11,
                            alignment=TA_JUSTIFY,
                            spaceAfter=12
                        )
                        
                        # Add content
                        elements.append(Paragraph("Educational Content", title_style))
                        elements.append(Spacer(1, 0.2*inch))
                        
                        elements.append(Paragraph(f"<b>Query:</b> {{st.session_state.chat_query}}", body_style))
                        elements.append(Paragraph(f"<b>Cognitive Level:</b> {{st.session_state.chat_current_label.capitalize()}}", body_style))
                        elements.append(Paragraph(f"<b>Learning Goal:</b> {{st.session_state.chat_current_purpose}}", body_style))
                        elements.append(Spacer(1, 0.3*inch))
                        
                        elements.append(Paragraph("Content", heading_style))
                        elements.append(Spacer(1, 0.1*inch))
                        
                        # Add content paragraphs
                        paragraphs = st.session_state.chat_generated_content.split('\\n')
                        for para in paragraphs:
                            if para.strip():
                                elements.append(Paragraph(para, body_style))
                        
                        # Build PDF
                        doc.build(elements)
                        
                        # Read the generated file
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
                        st.error(f"Error creating PDF: {{str(e)}}")
                        st.session_state.generate_pdf = False
            
            # Regenerate button
            st.markdown("---")
            if st.button("🔄 Regenerate Content", key="chat_regenerate", use_container_width=True):
                st.session_state.chat_show_generated_content = False
                st.session_state.chat_generated_content = None
                st.session_state.generate_word = False
                st.session_state.generate_pdf = False
                st.rerun()
            
            # Store in history
            if st.session_state.chat_current_label:
                history_entry = {
                    'query': st.session_state.chat_query,
                    'label': st.session_state.chat_current_label,
                    'confidence': 1.0,
                    'persona': persona,
                    'purpose': st.session_state.chat_current_purpose,
                    'generated_content': st.session_state.chat_generated_content
                }
                
                # Check if not already in history
                if not any(h['query'] == history_entry['query'] and 
                          h.get('purpose') == history_entry['purpose'] 
                          for h in st.session_state.classification_history):
                    st.session_state.classification_history.insert(0, history_entry)
            
            # Feedback section
            st.markdown("---")
            st.subheader("📣 How was this content?")
            feedback_cols = st.columns(5)
            
            with feedback_cols[0]:
                st.button("⭐", key="feedback_1")
            with feedback_cols[1]:
                st.button("⭐⭐", key="feedback_2")
            with feedback_cols[2]:
                st.button("⭐⭐⭐", key="feedback_3")
            with feedback_cols[3]:
                st.button("⭐⭐⭐⭐", key="feedback_4")
            with feedback_cols[4]:
                st.button("⭐⭐⭐⭐⭐", key="feedback_5")
            
            # Start new session
            st.markdown("---")
            if st.button("🔄 Start New Learning Session", key="chat_new_session", use_container_width=True):
                st.session_state.chat_query = None
                st.session_state.chat_current_label = None
                st.session_state.chat_current_purpose = None
                st.session_state.chat_purpose_confirmed = False
                st.session_state.chat_show_purpose_editor = False
                st.session_state.chat_show_generated_content = False
                st.session_state.chat_generated_content = None
                st.session_state.generate_word = False
                st.session_state.generate_pdf = False
                st.rerun()


# Tab 4: History
with tab4:
    st.header("📊 Classification History")
    
    if st.session_state.classification_history:
        # Clear history button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear History"):
                st.session_state.classification_history = []
                st.rerun()
        
        # Convert history to DataFrame for better display
        history_df = pd.DataFrame(st.session_state.classification_history)
        
        # Display as table
        display_columns = ['query', 'label', 'persona', 'confidence']
        if 'purpose' in history_df.columns:
            display_columns.append('purpose')
        
        st.dataframe(
            history_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Download history as CSV
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="classification_history.csv",
            mime="text/csv"
        )
        
        # Detailed view
        st.subheader("Detailed View")
        for idx, item in enumerate(st.session_state.classification_history):
            with st.expander(f"{idx+1}. {item['label'].capitalize()} - {item['query'][:50]}..."):
                st.write(f"**Full Query:** {item['query']}")
                st.write(f"**Level:** {item['label'].capitalize()}")
                st.write(f"**Persona:** {item['persona'].capitalize()}")
                if item['confidence'] < 1.0:
                    st.write(f"**Confidence:** {item['confidence']:.1%}")
                if item.get('purpose'):
                    st.write(f"**Purpose:** {item['purpose']}")
                if item.get('generated_content'):
                    st.markdown("**Generated Content:**")
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 0.5rem;">
                    {item['generated_content']}
                    </div>
                    """, unsafe_allow_html=True)
                    # Download button for this specific content
                    st.download_button(
                        label="📥 Download This Content",
                        data=item['generated_content'],
                        file_name=f"content_{idx+1}_{item['label']}.txt",
                        mime="text/plain",
                        key=f"download_history_{idx}"
                    )
    else:
        st.info("No classifications yet. Go to the 'Classify' tab to begin!")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with Streamlit | Powered by Groq API</p>
    <p><small>Bloom's Taxonomy: A framework for categorizing educational learning objectives</small></p>
</div>
""", unsafe_allow_html=True)