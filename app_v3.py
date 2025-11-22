"""
app_v3.py - Enhanced Streamlit Document Intelligence Application V3 (FIXED)
Features: Hybrid retrieval, enhanced Q&A, better answer quality, fixed KeyError issues
Run: streamlit run app_v3.py
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any

# Import our V3 utilities (enhanced versions)
from utils_v3 import process_uploaded_files, create_vector_db
from llm_utils_v3 import (
    get_llm_for_langchain,
    extract_structured_data,
    extract_new_column,
    create_enhanced_retrieval_qa_chain,
    enhanced_qa_response
)

# ================================================================== #
# App Configuration                                                  #
# ================================================================== #

st.set_page_config(
    page_title="AI Document Intelligence Platform V3 (Enhanced)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 AI-Powered Document Intelligence Platform V3")
st.markdown("**Enhanced with hybrid retrieval, semantic chunking, and improved answer quality**")
st.markdown("---")

# Version indicator
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.info("📌 **Version 3.0** - Enhanced RAG with BM25 + Semantic Retrieval")

# ================================================================== #
# Session State Management                                           #
# ================================================================== #

def initialize_session_state():
    """Initialize all session state variables with V3 enhancements."""
    defaults = {
        'df': None,
        'cached_texts': {},
        'db': None,
        'ensemble_retriever': None,  # V3: Hybrid retriever
        'qa_chain': None,
        'llm': None,
        'processing_complete': False,
        'last_query': '',
        'last_answer': '',
        'initialization_error': None,
        'processing_stats': {},  # V3: Enhanced stats
        'debug_info': {}         # V3: Debug information
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# ================================================================== #
# LLM Initialization with Enhanced Error Handling                   #
# ================================================================== #

if st.session_state.llm is None and st.session_state.initialization_error is None:
    with st.spinner("🧠 Initializing Enhanced AI Model (V3)..."):
        try:
            st.session_state.llm = get_llm_for_langchain()
            st.sidebar.success("✅ Enhanced AI model (V3) initialized successfully!")
            st.session_state.debug_info['llm_model'] = 'Enhanced V3 LLM'
        except Exception as e:
            st.session_state.initialization_error = str(e)
            st.sidebar.error(f"❌ Failed to initialize enhanced AI model: {str(e)}")
            st.sidebar.info("💡 Please ensure Ollama is running with required models")

# ================================================================== #
# Sidebar - Enhanced File Upload Section                            #
# ================================================================== #

st.sidebar.header("📁 Document Upload (V3 Enhanced)")
st.sidebar.markdown("**Supported formats:** PDF, DOCX, TXT, XLSX, PNG, JPG, JPEG")
st.sidebar.markdown("**V3 Features:** Semantic chunking, hybrid retrieval")

uploaded_files = st.sidebar.file_uploader(
    "Choose files to process:",
    type=["pdf", "docx", "txt", "xlsx", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="V3: Enhanced OCR, semantic chunking, and hybrid BM25+Vector retrieval"
)

if uploaded_files:
    st.sidebar.info(f"📄 {len(uploaded_files)} file(s) selected")
    
    if st.sidebar.button("🔄 Process Documents (V3 Enhanced)", type="primary"):
        if st.session_state.llm is None:
            st.sidebar.error("❌ Cannot process: Enhanced AI model not initialized")
            st.stop()
            
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            # Step 1: Enhanced text extraction
            status_text.text("📄 Extracting text with enhanced OCR...")
            progress_bar.progress(15)
            start_time = time.time()
            
            st.session_state.cached_texts = process_uploaded_files(uploaded_files)
            
            # Step 2: Enhanced structured data extraction
            status_text.text("🔍 Extracting structured data...")
            progress_bar.progress(35)
            
            extracted_data_list = []
            for file_name, text in st.session_state.cached_texts.items():
                if text.strip() and not text.startswith("Error:"):
                    structured_data = extract_structured_data(text)
                    structured_data["File Name"] = file_name
                    structured_data["Text Length"] = len(text)
                    structured_data["Processing Status"] = "✅ Success"
                    extracted_data_list.append(structured_data)
                else:
                    # Handle empty or error files
                    empty_data = {
                        "File Name": file_name,
                        "Applicant Name": "N/A",
                        "Application Date": "N/A",
                        "Income": "N/A",
                        "Family Size": "N/A",
                        "Text Length": 0,
                        "Processing Status": "⚠️ Failed"
                    }
                    extracted_data_list.append(empty_data)
            
            st.session_state.df = pd.DataFrame(extracted_data_list)
            
            # Step 3: Create enhanced vector database with hybrid retrieval
            status_text.text("🔗 Creating hybrid vector database (BM25 + Semantic)...")
            progress_bar.progress(70)
            
            all_texts = [text for text in st.session_state.cached_texts.values() 
                        if text.strip() and not text.startswith("Error:")]
            
            if all_texts:
                # V3: Enhanced vector DB creation
                db_result = create_vector_db(all_texts)
                
                if db_result and hasattr(db_result, 'ensemble_retriever'):
                    # V3: Enhanced vector store with ensemble retriever
                    st.session_state.db = db_result
                    st.session_state.ensemble_retriever = db_result.ensemble_retriever
                    retriever = db_result.ensemble_retriever
                elif db_result:
                    # Fallback to regular retriever
                    st.session_state.db = db_result
                    retriever = db_result.as_retriever(search_kwargs={"k": 3})
                    st.session_state.ensemble_retriever = None
                else:
                    st.sidebar.warning("⚠️ Vector database creation failed")
                    retriever = None
                
                # Step 4: Create enhanced QA chain
                if retriever:
                    status_text.text("🤖 Creating enhanced Q&A chain...")
                    progress_bar.progress(90)
                    st.session_state.qa_chain = create_enhanced_retrieval_qa_chain(retriever, st.session_state.llm)
                    st.sidebar.success("✅ Enhanced Q&A system ready!")
                else:
                    st.sidebar.warning("⚠️ Q&A system creation failed")
            else:
                st.sidebar.warning("⚠️ No text content found in uploaded files")
            
            # Complete with enhanced stats
            processing_time = time.time() - start_time
            progress_bar.progress(100)
            status_text.text(f"✅ V3 Processing Complete! ({processing_time:.1f}s)")
            
            # V3: Store processing statistics
            st.session_state.processing_stats = {
                'processing_time': processing_time,
                'total_files': len(uploaded_files),
                'successful_files': len([t for t in all_texts if t.strip()]),
                'total_characters': sum(len(t) for t in all_texts),
                'chunks_created': len(all_texts) if all_texts else 0,
                'retrieval_type': 'Hybrid (BM25 + Semantic)' if st.session_state.ensemble_retriever else 'Standard'
            }
            
            st.session_state.processing_complete = True
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ V3 Processing failed: {str(e)}")
            st.sidebar.error(f"Error details: {type(e).__name__}")
            progress_bar.empty()
            status_text.empty()
            st.stop()

# ================================================================== #
# Enhanced Processing Summary                                        #
# ================================================================== #

if st.session_state.processing_complete:
    st.sidebar.success("✅ V3 Enhanced processing complete!")
    
    # Enhanced metrics
    if st.session_state.processing_stats:
        stats = st.session_state.processing_stats
        st.sidebar.metric("Files Processed", f"{stats['successful_files']}/{stats['total_files']}")
        st.sidebar.metric("Processing Time", f"{stats['processing_time']:.1f}s")
        st.sidebar.metric("Total Content", f"{stats['total_characters']:,} chars")
        st.sidebar.metric("Retrieval Type", stats['retrieval_type'])

# Enhanced System Status
st.sidebar.markdown("---")
st.sidebar.markdown("**🔧 V3 System Status:**")
status_col1, status_col2 = st.sidebar.columns(2)

with status_col1:
    if st.session_state.db is not None:
        st.success("✅ Vector DB")
    else:
        st.error("❌ Vector DB")
    
    if st.session_state.ensemble_retriever is not None:
        st.success("✅ Hybrid Retrieval")
    elif st.session_state.db is not None:
        st.info("ℹ️ Standard Retrieval")
    else:
        st.error("❌ No Retrieval")

with status_col2:
    if st.session_state.qa_chain is not None:
        st.success("✅ Enhanced Q&A")
    else:
        st.error("❌ Enhanced Q&A")
    
    if st.session_state.llm is not None:
        st.success("✅ Enhanced LLM")
    else:
        st.error("❌ Enhanced LLM")

# ================================================================== #
# Main Content Area with Enhanced Features                          #
# ================================================================== #

# Enhanced tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Extracted Data", 
    "➕ Custom Columns", 
    "💬 Enhanced Q&A", 
    "🔧 V3 Diagnostics"
])

# ================================================================== #
# Tab 1: Enhanced Extracted Data Display (FIXED)                    #
# ================================================================== #

with tab1:
    st.header("📊 Enhanced Structured Data Extraction (V3)")
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        # FIXED: Ensure Processing Status column exists
        if "Processing Status" not in st.session_state.df.columns:
            st.session_state.df["Processing Status"] = "✅ Success"
        
        # Enhanced summary metrics with V3 features
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Files", len(st.session_state.df))
        with col2:
            successful = len(st.session_state.df[st.session_state.df["Processing Status"] == "✅ Success"])
            st.metric("Successful", successful)
        with col3:
            extractions = len(st.session_state.df[st.session_state.df["Applicant Name"] != "N/A"])
            st.metric("Data Extractions", extractions)
        with col4:
            if "Text Length" in st.session_state.df.columns:
                avg_length = st.session_state.df["Text Length"].mean()
                st.metric("Avg. Length", f"{avg_length:.0f} chars")
        with col5:
            if "Text Length" in st.session_state.df.columns:
                total_chars = st.session_state.df["Text Length"].sum()
                st.metric("Total Content", f"{total_chars:,} chars")
        
        st.subheader("📋 Enhanced Data Table")
        
        # Enhanced display options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # Default columns (exclude long text columns)
            default_cols = [col for col in st.session_state.df.columns 
                          if col not in ["Text Length"]]
            
            show_columns = st.multiselect(
                "Select columns to display:",
                options=st.session_state.df.columns.tolist(),
                default=default_cols,
                help="V3: Enhanced column selection"
            )
        with col2:
            if st.button("📥 Download CSV"):
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="Download Enhanced Data",
                    data=csv,
                    file_name="extracted_data_v3.csv",
                    mime="text/csv"
                )
        with col3:
            if st.button("📊 Data Summary"):
                st.info("V3 Enhanced data summary feature - coming soon!")
        
        # Filter and display dataframe
        if show_columns:
            display_df = st.session_state.df[show_columns]
        else:
            display_df = st.session_state.df
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # V3: Enhanced data quality metrics
        st.subheader("📈 V3 Data Quality Metrics")
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            completion_rate = (extractions / len(st.session_state.df)) * 100 if len(st.session_state.df) > 0 else 0
            st.metric("Data Completion", f"{completion_rate:.1f}%")
        
        with quality_col2:
            processing_success = (successful / len(st.session_state.df)) * 100 if len(st.session_state.df) > 0 else 0
            st.metric("Processing Success", f"{processing_success:.1f}%")
        
        with quality_col3:
            if "Text Length" in st.session_state.df.columns:
                non_empty = len(st.session_state.df[st.session_state.df["Text Length"] > 0])
                content_rate = (non_empty / len(st.session_state.df)) * 100 if len(st.session_state.df) > 0 else 0
                st.metric("Content Extraction", f"{content_rate:.1f}%")
    
    else:
        st.info("👆 Upload and process documents to see enhanced V3 data extraction.")
        st.markdown("**V3 Features:**")
        st.markdown("• Enhanced OCR with preprocessing")
        st.markdown("• Semantic chunking for better context")
        st.markdown("• Improved field extraction accuracy")

# ================================================================== #
# Tab 2: Enhanced Custom Column Addition                            #
# ================================================================== #

with tab2:
    st.header("➕ Enhanced Custom Data Extraction (V3)")
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.markdown("**V3 Enhanced:** Better field recognition, smarter extraction, validation")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_column_query = st.text_input(
                "What information do you want to extract?",
                placeholder="e.g., Email Address, Phone Number, Policy ID, Address...",
                help="V3: Enhanced extraction with better pattern recognition and validation"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            add_column_btn = st.button("🔍 Extract with V3", type="primary")
        
        if add_column_btn and new_column_query.strip():
            with st.spinner(f"🔍 V3 Enhanced extraction: '{new_column_query}' from all documents..."):
                try:
                    new_column_data = []
                    progress = st.progress(0)
                    extraction_stats = {'found': 0, 'not_found': 0, 'errors': 0}
                    
                    for idx, file_name in enumerate(st.session_state.df["File Name"]):
                        text = st.session_state.cached_texts.get(file_name, "")
                        try:
                            extracted_value = extract_new_column(text, new_column_query)
                            new_column_data.append(extracted_value)
                            
                            # V3: Track extraction statistics
                            if extracted_value != "N/A":
                                extraction_stats['found'] += 1
                            else:
                                extraction_stats['not_found'] += 1
                                
                        except Exception as e:
                            new_column_data.append("Error")
                            extraction_stats['errors'] += 1
                            
                        progress.progress((idx + 1) / len(st.session_state.df))
                    
                    st.session_state.df[new_column_query] = new_column_data
                    progress.empty()
                    
                    # V3: Enhanced success message with statistics
                    st.success(f"✅ V3 Enhanced extraction complete: '{new_column_query}'!")
                    
                    # Show extraction statistics
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Found", extraction_stats['found'])
                    with stat_col2:
                        st.metric("Not Found", extraction_stats['not_found'])
                    with stat_col3:
                        st.metric("Errors", extraction_stats['errors'])
                    
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ V3 extraction error: {str(e)}")
        
        elif add_column_btn:
            st.warning("⚠️ Please enter a field description for V3 extraction.")
        
        # V3: Enhanced column preview with extraction tips
        if st.session_state.df is not None:
            st.subheader("📋 Current Extracted Columns")
            cols = [col for col in st.session_state.df.columns if col not in ["Text Length"]]
            st.write(", ".join(cols))
            
            # V3: Extraction tips
            with st.expander("💡 V3 Enhanced Extraction Tips"):
                st.markdown("""
                **V3 improvements for better extraction:**
                • More accurate regex patterns for emails, phones, dates
                • Better handling of formatted text (tables, forms)
                • Smart validation and cleaning of extracted values
                • Context-aware field recognition
                
                **Example queries that work well:**
                • "Email Address" or "Email"
                • "Phone Number" or "Mobile"
                • "Social Security Number" or "SSN"
                • "Date of Birth" or "DOB"
                • "Address" or "Home Address"
                """)
    
    else:
        st.info("👆 Process documents first to use V3 enhanced column extraction.")

# ================================================================== #
# Tab 3: Enhanced Q&A Interface                                     #
# ================================================================== #

with tab3:
    st.header("💬 Enhanced Q&A with Hybrid Retrieval (V3)")
    
    # V3: Enhanced system status display
    st.markdown("**🔧 V3 Enhanced System Status:**")
    status_cols = st.columns(4)
    
    with status_cols[0]:
        if st.session_state.db is not None:
            st.success("✅ Vector DB")
        else:
            st.error("❌ Vector DB")
    
    with status_cols[1]:
        if st.session_state.ensemble_retriever is not None:
            st.success("✅ Hybrid Search")
        elif st.session_state.db is not None:
            st.info("ℹ️ Standard Search")
        else:
            st.error("❌ No Search")
    
    with status_cols[2]:
        if st.session_state.qa_chain is not None:
            st.success("✅ Enhanced Q&A")
        else:
            st.error("❌ Enhanced Q&A")
    
    with status_cols[3]:
        if st.session_state.llm is not None:
            st.success("✅ Enhanced LLM")
        else:
            st.error("❌ Enhanced LLM")
    
    if st.session_state.qa_chain is not None:
        st.markdown("**V3 Features:** Multi-query expansion, BM25+Semantic hybrid search, enhanced answer validation")
        
        # Enhanced query input with examples
        user_query = st.text_input(
            "Ask anything about your documents:",
            placeholder="e.g., Summarize the main points, What are the income levels?, Who applied in 2024?",
            help="V3: Enhanced with query expansion and hybrid retrieval for better answers"
        )
        
        # Enhanced ask button
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🤖 Ask Enhanced V3", type="primary")
        
        if ask_button and user_query.strip():
            with st.spinner("🔍 V3 Enhanced processing: Multi-query expansion + Hybrid search..."):
                try:
                    # Enhanced QA processing
                    start_time = time.time()
                    retriever = st.session_state.ensemble_retriever or st.session_state.db.as_retriever()
                    
                    answer = enhanced_qa_response(
                        user_query,
                        retriever,
                        st.session_state.llm
                    )
                    processing_time = time.time() - start_time
                    
                    st.session_state.last_query = user_query
                    st.session_state.last_answer = answer
                    
                    # V3: Store query metadata
                    st.session_state.debug_info['last_query_time'] = processing_time
                    
                    st.success(f"✅ V3 processing complete in {processing_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ V3 processing error: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")
        
        # Enhanced answer display
        if st.session_state.last_query and st.session_state.last_answer:
            st.subheader("💡 V3 Enhanced Answer")
            
            # V3: Query metadata
            if 'last_query_time' in st.session_state.debug_info:
                query_time = st.session_state.debug_info['last_query_time']
                st.caption(f"⚡ Processed in {query_time:.2f}s with V3 hybrid retrieval")
            
            st.markdown(f"**Question:** {st.session_state.last_query}")
            
            # V3: Enhanced answer formatting
            st.markdown("**Enhanced Answer:**")
            st.markdown(st.session_state.last_answer)
            
            # Enhanced feedback with V3 features
            st.subheader("📝 V3 Enhanced Feedback")
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                helpful = st.radio("Answer Quality:", ["Excellent", "Good", "Poor"], horizontal=True)
            
            with feedback_col2:
                if st.button("📤 Submit V3 Feedback"):
                    st.success("Thank you for V3 feedback!")
            
            if helpful == "Poor":
                feedback_text = st.text_area("How can V3 improve this answer?")
        
        # V3: Query suggestions
        st.subheader("💡 V3 Enhanced Query Suggestions")
        suggestion_cols = st.columns(2)
        
        with suggestion_cols[0]:
            st.markdown("**📊 Data Analysis:**")
            if st.button("🔢 Count all applications"):
                user_query = "How many applications are there in total?"
                st.rerun()
            if st.button("💰 Average income analysis"):
                user_query = "What is the average income of all applicants?"
                st.rerun()
        
        with suggestion_cols[1]:
            st.markdown("**📝 Document Summary:**")
            if st.button("📋 Summarize all documents"):
                user_query = "Provide a comprehensive summary of all documents"
                st.rerun()
            if st.button("🔍 Key information overview"):
                user_query = "What are the key pieces of information across all documents?"
                st.rerun()
    
    else:
        st.error("❌ V3 Enhanced Q&A system not initialized")
        st.info("Please ensure all documents are processed and the system is ready.")
        
        if st.session_state.db is None:
            st.warning("⚠️ Vector database not created. Please process documents first.")
        if st.session_state.llm is None:
            st.warning("⚠️ LLM not initialized. Check Ollama configuration.")

# ================================================================== #
# Tab 4: V3 Diagnostics and Debug Information                       #
# ================================================================== #

with tab4:
    st.header("🔧 V3 Enhanced Diagnostics")
    
    # System information
    st.subheader("📊 System Status")
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        st.markdown("**Core Components:**")
        st.write(f"• Vector DB: {'✅ Active' if st.session_state.db else '❌ Inactive'}")
        st.write(f"• Hybrid Retriever: {'✅ Active' if st.session_state.ensemble_retriever else '❌ Inactive'}")
        st.write(f"• Enhanced Q&A: {'✅ Active' if st.session_state.qa_chain else '❌ Inactive'}")
        st.write(f"• Enhanced LLM: {'✅ Active' if st.session_state.llm else '❌ Inactive'}")
    
    with diag_col2:
        st.markdown("**V3 Features:**")
        st.write("• ✅ Semantic chunking")
        st.write("• ✅ BM25 + Vector hybrid search")
        st.write("• ✅ Multi-query expansion")
        st.write("• ✅ Enhanced answer validation")
    
    # Processing statistics
    if st.session_state.processing_stats:
        st.subheader("📈 Processing Statistics")
        stats = st.session_state.processing_stats
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Files Processed", f"{stats['successful_files']}/{stats['total_files']}")
        with stat_cols[1]:
            st.metric("Processing Time", f"{stats['processing_time']:.2f}s")
        with stat_cols[2]:
            st.metric("Total Characters", f"{stats['total_characters']:,}")
        with stat_cols[3]:
            st.metric("Retrieval Method", stats['retrieval_type'])
    
    # Debug information
    st.subheader("🐛 Debug Information")
    if st.session_state.debug_info:
        st.json(st.session_state.debug_info)
    else:
        st.info("No debug information available. Process documents to generate debug data.")
    
    # V3 Performance metrics
    st.subheader("⚡ V3 Performance Metrics")
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("**Retrieval Performance:**")
        if st.session_state.ensemble_retriever:
            st.success("🚀 Hybrid BM25 + Semantic retrieval active")
            st.write("• Keyword matching: BM25 (40% weight)")
            st.write("• Semantic matching: Vector search (60% weight)")
        elif st.session_state.db:
            st.info("ℹ️ Standard semantic retrieval only")
        else:
            st.warning("⚠️ No retrieval system active")
    
    with perf_col2:
        st.markdown("**Answer Quality:**")
        st.write("• ✅ Multi-query expansion")
        st.write("• ✅ Answer validation")
        st.write("• ✅ Hallucination detection")
        st.write("• ✅ Response length optimization")
    
    # System logs (placeholder)
    if st.button("📋 Show System Logs"):
        st.info("V3 system logs feature - coming soon!")
    
    # Reset system
    if st.button("🔄 Reset V3 System", type="secondary"):
        # Clear session state
        keys_to_reset = ['df', 'cached_texts', 'db', 'ensemble_retriever', 'qa_chain', 
                        'processing_complete', 'processing_stats', 'debug_info']
        for key in keys_to_reset:
            if key in st.session_state:
                if key == 'processing_complete':
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None
        st.success("V3 system reset complete!")
        st.rerun()

# ================================================================== #
# Enhanced Footer                                                    #
# ================================================================== #

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
        **🚀 V3 Enhanced Features:**
        Hybrid BM25+Semantic Retrieval • Multi-Query Expansion • Enhanced Answer Validation • Semantic Chunking
        
        **💡 Tips:** 
        • Use specific questions for best results
        • Try summary queries for document overviews  
        • Check V3 diagnostics for system status
        """
    )

# Version info
st.markdown("---")
st.caption("AI Document Intelligence Platform V3 - Enhanced with hybrid retrieval and improved answer quality")