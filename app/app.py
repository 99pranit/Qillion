import streamlit as st
import pandas as pd
from package.classify_query import classifyQuery
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Bloom's Taxonomy Classifier",
    page_icon="🎓",
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
    .taxonomy-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# Color mapping for Bloom's levels
LEVEL_COLORS = {
    'knowledge': '#e74c3c',
    'comprehension': '#e67e22',
    'application': '#f39c12',
    'analysis': '#2ecc71',
    'synthesis': '#3498db',
    'evaluation': '#9b59b6'
}

# Level descriptions
LEVEL_DESCRIPTIONS = {
    'knowledge': 'Recalling facts, terms, basic concepts, or answers',
    'comprehension': 'Demonstrating understanding by interpreting or explaining',
    'application': 'Using learned information in new situations',
    'analysis': 'Breaking down information and examining relationships',
    'synthesis': 'Combining elements to form new wholes',
    'evaluation': 'Making judgments based on criteria and standards'
}

def display_taxonomy_pyramid():
    """Display Bloom's Taxonomy pyramid"""
    levels = ['Evaluation', 'Synthesis', 'Analysis', 'Application', 'Comprehension', 'Knowledge']
    colors = ['#9b59b6', '#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    fig = go.Figure()
    
    for i, (level, color) in enumerate(zip(levels, colors)):
        fig.add_trace(go.Bar(
            y=[level],
            x=[len(levels) - i],
            orientation='h',
            marker=dict(color=color),
            name=level,
            showlegend=False,
            text=level,
            textposition='inside',
            hovertemplate=f'<b>{level}</b><br>{LEVEL_DESCRIPTIONS[level.lower()]}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Bloom's Taxonomy Pyramid",
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        barmode='stack'
    )
    
    return fig

def classify_single_query(query, persona, model_name, correction_model, api_provider):
    """Classify a single query"""
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
            return result
        else:
            label = classifier.get_label()
            return {'final_label': label, 'confidence': 1.0}
            
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return None

def display_result(result, query):
    """Display classification result with styling"""
    if result and result.get('final_label'):
        label = result['final_label']
        confidence = result.get('confidence', 1.0)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            color = LEVEL_COLORS.get(label, "#8e9899")
            st.markdown(f"""
            <div class="taxonomy-box" style="background-color: {color}20; border-left: 5px solid {color};">
                <h3 style="color: {color}; margin: 0;">{label.upper()}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #555;">{LEVEL_DESCRIPTIONS.get(label, '')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.metric("Level Rank", f"{list(LEVEL_COLORS.keys()).index(label) + 1}/6")

def main():
    # Header
    st.markdown('<div class="main-header">🎓 Bloom\'s Taxonomy Query Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify educational queries using cognitive learning levels</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_provider = st.selectbox(
            "API Provider",
            options=['groq', 'perplexity', 'gemini'],
            help="Currently supports Groq API"
        )
        
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
        
        persona = st.selectbox(
            "Classification Mode",
            options=['multi', 'professor', 'student', 'psychologist', 'engineer', 'examiner'],
            help="Use 'multi' for ensemble voting with all personas"
        )
        
        st.divider()
        
        # Display taxonomy pyramid
        st.plotly_chart(display_taxonomy_pyramid(), use_container_width=True)
        
        st.divider()
        
        # API Key status
        import os
        api_key = os.environ.get(api_provider.upper() + "_API_KEY")
        if api_key:
            st.success("✅ API Key detected")
        else:
            st.error("❌ API Key not found")
            st.info(f"Set {api_provider.upper()}_API_KEY environment variable")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Query", "📊 Batch Processing", "📈 Analytics", "ℹ️ About"])
    
    # Tab 1: Single Query Classification
    with tab1:
        st.subheader("Classify a Single Query")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query_input = st.text_area(
                "Enter your query:",
                height=100,
                placeholder="e.g., What is the capital of France?"
            )
        
        with col2:
            st.write("")
            st.write("")
            classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if classify_btn:
            if not query_input.strip():
                st.warning("⚠️ Please enter a query to classify")
            else:
                with st.spinner("Classifying query..."):
                    result = classify_single_query(
                        query_input,
                        persona,
                        model_name,
                        correction_model,
                        api_provider
                    )
                    
                    if result:
                        st.success("✅ Classification complete!")
                        display_result(result, query_input)
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query_input,
                            'label': result['final_label'],
                            'confidence': result.get('confidence', 1.0),
                            'persona': persona
                        })
                        
                        # Show detailed info
                        with st.expander("🔍 View Details"):
                            st.json(result)
    
    # Tab 2: Batch Processing
    with tab2:
        st.subheader("Batch Query Classification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload CSV/TXT file with queries",
                type=['csv', 'txt'],
                help="CSV should have a 'query' column. TXT should have one query per line."
            )
        
        with col2:
            st.write("")
            st.write("")
            process_btn = st.button("⚡ Process Batch", type="primary", use_container_width=True)
        
        # Sample data option
        use_sample = st.checkbox("Or use sample queries")
        
        if use_sample:
            sample_queries = [
                "What is photosynthesis?",
                "Explain how Newton's laws apply to everyday life",
                "Calculate the area of a circle with radius 5cm",
                "Compare and contrast democracy and autocracy",
                "Design a solution for reducing traffic congestion",
                "Evaluate the effectiveness of renewable energy policies"
            ]
            st.write("Sample queries loaded:", len(sample_queries))
        
        if process_btn:
            queries = []
            
            if use_sample:
                queries = sample_queries
            elif uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'query' in df.columns:
                        queries = df['query'].tolist()
                    else:
                        st.error("CSV must contain a 'query' column")
                else:
                    queries = uploaded_file.read().decode('utf-8').strip().split('\n')
            else:
                st.warning("Please upload a file or select sample queries")
            
            if queries:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, query in enumerate(queries):
                    status_text.text(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
                    
                    result = classify_single_query(
                        query,
                        persona,
                        model_name,
                        correction_model,
                        api_provider
                    )
                    
                    if result:
                        results.append({
                            'Query': query,
                            'Classification': result['final_label'],
                            'Confidence': result.get('confidence', 1.0)
                        })
                    
                    progress_bar.progress((i + 1) / len(queries))
                
                status_text.text("✅ Batch processing complete!")
                
                # Store results
                st.session_state.batch_results = pd.DataFrame(results)
                
                # Display results
                st.dataframe(st.session_state.batch_results, use_container_width=True)
                
                # Download option
                csv = st.session_state.batch_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Tab 3: Analytics
    with tab3:
        st.subheader("Classification Analytics")
        
        # Choose data source
        data_source = st.radio(
            "Data Source:",
            options=["Session History", "Batch Results"],
            horizontal=True
        )
        
        if data_source == "Session History" and st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution chart
                dist_data = df['label'].value_counts()
                fig = px.pie(
                    values=dist_data.values,
                    names=dist_data.index,
                    title="Classification Distribution",
                    color=dist_data.index,
                    color_discrete_map=LEVEL_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence over time
                fig = px.line(
                    df,
                    x=df.index,
                    y='confidence',
                    title="Confidence Over Time",
                    markers=True
                )
                fig.update_layout(xaxis_title="Query Number", yaxis_title="Confidence")
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", len(df))
            with col2:
                st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
            with col3:
                st.metric("Most Common", df['label'].mode()[0].upper())
            with col4:
                st.metric("Unique Levels", df['label'].nunique())
            
            # Detailed history
            with st.expander("📜 View Full History"):
                st.dataframe(df, use_container_width=True)
                
                # Clear history button
                if st.button("🗑️ Clear History"):
                    st.session_state.history = []
                    st.rerun()
        
        elif data_source == "Batch Results" and st.session_state.batch_results is not None:
            df = st.session_state.batch_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution chart
                dist_data = df['Classification'].value_counts()
                fig = px.bar(
                    x=dist_data.index,
                    y=dist_data.values,
                    title="Classification Distribution",
                    labels={'x': 'Level', 'y': 'Count'},
                    color=dist_data.index,
                    color_discrete_map=LEVEL_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = px.histogram(
                    df,
                    x='Confidence',
                    title="Confidence Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Level breakdown
            st.subheader("Level Breakdown")
            for level in df['Classification'].unique():
                level_df = df[df['Classification'] == level]
                with st.expander(f"{level.upper()} ({len(level_df)} queries)"):
                    st.dataframe(level_df[['Query', 'Confidence']], use_container_width=True)
        
        else:
            st.info("📊 No data available. Classify some queries to see analytics!")
    
    # Tab 4: About
    with tab4:
        st.subheader("About Bloom's Taxonomy")
        
        st.markdown("""
        ### What is Bloom's Taxonomy?
        
        Bloom's Taxonomy is a hierarchical framework for categorizing educational learning objectives 
        into levels of complexity and specificity. It was created in 1956 by Benjamin Bloom.
        
        ### The Six Levels:
        """)
        
        for level, description in LEVEL_DESCRIPTIONS.items():
            color = LEVEL_COLORS[level]
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 1rem; border-radius: 8px; 
                        border-left: 4px solid {color}; margin: 0.5rem 0;">
                <h4 style="color: {color}; margin: 0;">{level.upper()}</h4>
                <p style="margin: 0.5rem 0 0 0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ### How to Use This App:
        
        1. **Single Query Mode**: Enter a question and get instant classification
        2. **Batch Processing**: Upload multiple queries for bulk classification
        3. **Multi-Persona Ensemble**: Use the 'multi' mode for more reliable results through voting
        4. **Analytics**: Track and visualize your classification history
        
        ### Tips for Best Results:
        
        - Use clear, complete questions
        - Be specific about the cognitive task required
        - Use the ensemble mode for ambiguous queries
        - Review the confidence scores
        
        ---
        
        **Need Help?** Check your API key configuration in the sidebar.
        """)

if __name__ == "__main__":
    main()