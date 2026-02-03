import streamlit as st
from package.classify_query import classifyQuery
import pandas as pd
import io
import os
from openai import OpenAI

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
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1565c0;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #666;
    }
    .taxonomy-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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
    
    # ChatBot Model selection
    st.header("💬 ChatBot Settings")
    chat_model = st.selectbox(
        "ChatBot Model",
        options=['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'meta-llama/llama-4-scout-17b-16e-instruct'],
        help="Model for generating chatbot responses"
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

def get_taxonomy_system_prompt(taxonomy_level):
    """Generate a system prompt based on Bloom's taxonomy level."""
    
    prompts = {
        'knowledge': """You are an educational assistant specialized in providing factual, knowledge-level responses.

Your role is to help users recall and recognize information. When responding:
- Provide direct, accurate facts
- Focus on definitions, terms, and basic concepts
- Keep answers concise and to the point
- List key information clearly
- Help users remember and identify information""",

        'comprehension': """You are an educational assistant specialized in helping users understand and explain concepts.

Your role is to facilitate comprehension. When responding:
- Explain concepts in clear, accessible language
- Interpret and translate information
- Provide examples to illustrate ideas
- Summarize key points
- Help users grasp the meaning and significance
- Rephrase complex ideas in simpler terms
""",

        'application': """You are an educational assistant specialized in helping users apply knowledge to new situations.

Your role is to guide practical application. When responding:
- Show how to use concepts in real-world scenarios
- Provide step-by-step procedures
- Give concrete examples of application
- Demonstrate problem-solving approaches
- Help users transfer knowledge to new contexts
- Offer practice opportunities
""",

        'analysis': """You are an educational assistant specialized in helping users analyze and break down information.

Your role is to facilitate analytical thinking. When responding:
- Break down complex information into components
- Examine relationships between parts
- Compare and contrast different elements
- Identify patterns and underlying structures
- Distinguish between facts and inferences
- Analyze cause-and-effect relationships
""",

        'synthesis': """You are an educational assistant specialized in helping users create and synthesize new ideas.

Your role is to foster creative thinking. When responding:
- Help combine ideas in innovative ways
- Guide the creation of new solutions
- Encourage original thinking
- Support the development of plans and proposals
- Facilitate the integration of diverse concepts
- Promote creative problem-solving
""",

        'evaluation': """You are an educational assistant specialized in helping users make informed judgments and evaluations.

Your role is to guide critical evaluation. When responding:
- Help assess based on clear criteria
- Present multiple perspectives for consideration
- Encourage critical thinking
- Support evidence-based judgment
- Guide the evaluation of arguments and solutions
- Promote justified decision-making
"""
    }
    
    return prompts.get(taxonomy_level, prompts['knowledge'])

def generate_llm_response(query, taxonomy_level, chat_model_name, chat_history=None):
    """Generate response using Groq API with taxonomy-aware prompt engineering."""
    
    try:
        # Initialize Groq client
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY not found. Please set your API key in the sidebar."
        
        client = OpenAI(
            base_url = "https://api.groq.com/openai/v1",
            api_key = api_key
        )
        
        # Get taxonomy-specific system prompt
        system_prompt = get_taxonomy_system_prompt(taxonomy_level)
        
        # Build messages array
        messages = [
            {"role": "system", 
             "content": system_prompt}
        ]
        
        # Add chat history if available
        if chat_history:
            for msg in chat_history[-6:]:  # Include last 6 messages for context
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user query
        messages.append({
            "role": "user",
            "content": "query: " + query
        })
        
        # Call Groq API
        response = client.chat.completions.create(
            model=chat_model_name,
            messages=messages
        )
        
        return response.choices[0].message.content.lower()
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify", "📦 Batch Processing", "💬 ChatBot", "📊 History"])

# Tab 1: Single Query Classification
with tab1:
    st.header("🔍 Query Classification")
    
    # Query input
    query = st.text_area(
        "Enter your educational query:",
        height=150,
        placeholder="Type your question here...",
        help="Enter any educational question to classify"
    )
    
    # Example queries
    st.subheader("💡 Example Queries (Click to use)")
    examples = [
        "What is the capital of France?",
        "Explain in your own words what Newton's First Law means.",
        "Use Ohm's law to calculate the current in a circuit.",
        "Examine the causes of World War I.",
        "Propose a plan to reduce plastic pollution in cities.",
        "Critique the author's argument in the article.",
        "Define photosynthesis.",
        "Compare quicksort and mergesort algorithms."
    ]
    
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            if st.button(example, key=f"example_{idx}"):
                query = example
                st.rerun()
    
    # Classify button
    classify_button = st.button("🚀 Classify Query", type="primary")
    
    if classify_button:
        if not query.strip():
            st.error("⚠️ Please enter a query to classify")
        else:
            with st.spinner("🔄 Classifying query..."):
                try:
                    # Create classifier instance
                    classifier = classifyQuery(
                        api_provider=api_provider,
                        persona=persona,
                        query=query,
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
                        'query': query,
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
                            
                            results.append({
                                'query': query_text,
                                'label': label,
                                'confidence': confidence
                            })
                            
                        except Exception as e:
                            results.append({
                                'query': query_text,
                                'label': 'ERROR',
                                'confidence': 0.0
                            })
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

# Tab 3: ChatBot with Bloom's Taxonomy Context
with tab3:
    st.header("💬 Taxonomy-Aware ChatBot")
    st.markdown("Ask questions and receive AI-generated responses tailored to the cognitive level of your query using Groq LLM.")

    # ChatBot settings
    with st.expander("⚙️ ChatBot Settings"):
        auto_classify = st.checkbox("Auto-classify queries", value=True, 
                                    help="Automatically detect Bloom's taxonomy level for each query")
        show_taxonomy_tags = st.checkbox("Show taxonomy tags", value=True,
                                        help="Display taxonomy level with each message")
        include_context = st.checkbox("Include conversation context", value=True,
                                     help="Send previous messages to LLM for better context")
    
    # Display chat history
    st.subheader("💭 Conversation")
    
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                taxonomy_tag = ""
                if show_taxonomy_tags and 'taxonomy' in message and message['taxonomy']:
                    taxonomy_level = message['taxonomy']
                    taxonomy_tag = f'<span class="taxonomy-tag {taxonomy_level}">{taxonomy_level.upper()}</span>'
                
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>You:</strong> {message["content"]}{taxonomy_tag}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f'<strong>Assistant:</strong> {message["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # Chat input
    user_input = st.text_input("Your message:", key="chat_input", placeholder="Type your question here...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("📤 Send", type="primary")
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    if send_button and user_input.strip():
        # Classify the query if auto-classify is enabled
        taxonomy_level = None
        if auto_classify:
            with st.spinner("🔄 Analyzing query..."):
                try:
                    classifier = classifyQuery(
                        api_provider=api_provider,
                        persona=persona,
                        query=user_input,
                        model_name=model_name,
                        correction_model=correction_model
                    )
                    
                    if persona == 'multi':
                        result = classifier.get_ensemble_label()
                        taxonomy_level = result['final_label']
                    else:
                        taxonomy_level = classifier.get_label()
                except Exception as e:
                    st.error(f"Classification error: {str(e)}")
                    taxonomy_level = "knowledge"  # Default fallback
        else:
            taxonomy_level = "knowledge"  # Default when auto-classify is off
        
        # Add user message to chat history
        user_message = {
            'role': 'user',
            'content': user_input,
            'taxonomy': taxonomy_level
        }
        st.session_state.chat_history.append(user_message)
        
        # Generate LLM response with taxonomy-aware prompt
        with st.spinner(f"🤖 Generating {taxonomy_level}-level response..."):
            if include_context:
                response = generate_llm_response(
                    user_input, 
                    taxonomy_level, 
                    chat_model,
                    st.session_state.chat_history[:-1]  # Exclude the just-added user message
                )
            else:
                response = generate_llm_response(
                    user_input, 
                    taxonomy_level, 
                    chat_model,
                    None
                )
        
        # Add assistant response to chat history
        assistant_message = {
            'role': 'assistant',
            'content': response
        }
        st.session_state.chat_history.append(assistant_message)
        
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
        st.dataframe(
            history_df[['query', 'label', 'persona', 'confidence']],
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