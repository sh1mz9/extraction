"""
llm_utils_v3_fixed.py - Enhanced Q&A System with Fixed MultiQueryRetriever
Fixes: MultiQueryRetriever num_queries parameter issue, improved prompting, answer validation
"""

import re
import functools
from typing import Dict, Any, Optional, List
import json

import ollama
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.retrievers.multi_query import MultiQueryRetriever

# ================================================================== #
# 1. Enhanced LLM Initialization with Better Error Handling         #
# ================================================================== #

def get_llm_for_langchain() -> OllamaLLM:
    """Initialize LangChain-compatible Ollama LLM with fallback options."""
    models_to_try = ["smollm", "llama3.2", "llama2", "qwen2:0.5b", "mistral"]
    
    for model in models_to_try:
        try:
            print(f"Attempting to initialize {model}...")
            llm = OllamaLLM(
                model=model, 
                temperature=0,  # Deterministic responses
                request_timeout=60,
                num_predict=512,  # Longer responses for better quality
                top_k=1,         # Most likely tokens only
                top_p=0.1        # Low randomness
            )
            # Test the model
            test_response = llm.invoke("Respond with 'OK' if you can understand this.")
            if test_response and "OK" in test_response.upper():
                print(f"Successfully initialized {model}")
                return llm
        except Exception as e:
            print(f"Failed to initialize {model}: {e}")
            continue
    
    # Final fallback
    print("Using fallback LLM configuration")
    return OllamaLLM(model="smollm", temperature=0, request_timeout=30)

# ================================================================== #
# 2. Enhanced Regex Patterns and Hybrid Extraction                  #
# ================================================================== #

_COMPILED_REGEXES = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
    "phone": re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
    "date": re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4})\b'),
    "income": re.compile(r'\$\s?[\d,]+(?:\.\d{2})?\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "social": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "zip": re.compile(r'\b\d{5}(?:-\d{4})?\b'),
    "policy": re.compile(r'\b[A-Z]{2,4}\d{6,12}\b'),
    "account": re.compile(r'\b\d{8,16}\b'),
    "name": re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b'),
    "address": re.compile(r'\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)', re.IGNORECASE),
}

@functools.lru_cache(maxsize=1024)
def hybrid_extraction(text: str, field_name: str) -> str:
    """Enhanced extraction with better regex patterns and LLM fallback."""
    if not text or not text.strip():
        return "N/A"
    
    # Normalize field name for pattern matching
    field_lower = field_name.lower().replace(" ", "").replace("_", "")
    
    # Try regex first for speed and accuracy
    for pattern_name, compiled_regex in _COMPILED_REGEXES.items():
        if pattern_name in field_lower:
            matches = compiled_regex.findall(text)
            if matches:
                return matches[0] if len(matches) == 1 else "; ".join(matches[:3])
    
    # Enhanced LLM fallback with better prompt
    return _llm_extract_with_validation(text, field_name)

@functools.lru_cache(maxsize=512)
def _llm_extract_with_validation(text: str, field_name: str) -> str:
    """LLM extraction with validation and quality control."""
    truncated_text = text[:2000] if len(text) > 2000 else text
    
    # More specific prompt for better extraction
    prompt = f"""Extract ONLY the "{field_name}" from this text.

TEXT:
{truncated_text}

INSTRUCTIONS:
- Return ONLY the specific value requested
- If not found, return exactly "N/A"
- Do not add explanations or extra text
- For names: return full name only
- For dates: use MM/DD/YYYY format
- For phone: include area code
- For addresses: return complete address

{field_name.upper()}:"""
    
    try:
        response = ollama.chat(
            model='smollm',
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0,
                'num_predict': 50,  # Short responses
                'top_k': 1,
                'top_p': 0.1
            }
        )
        
        content = response['message']['content'].strip()
        
        # Validate and clean response
        content = _validate_extraction_response(content, field_name)
        
        return content if content and content.lower() != "n/a" else "N/A"
    
    except Exception as e:
        print(f"LLM extraction error for '{field_name}': {e}")
        return "N/A"

def _validate_extraction_response(content: str, field_name: str) -> str:
    """Validate and clean extraction responses."""
    if not content:
        return "N/A"
    
    # Remove common LLM artifacts
    content = re.sub(r'^(The\s+|Answer:\s*|Result:\s*)', '', content, flags=re.IGNORECASE)
    content = content.replace("**", "").replace("*", "").strip()
    
    # Check for explanation text (indicates low confidence)
    explanation_indicators = [
        "based on", "according to", "it appears", "seems to be", 
        "might be", "could be", "not clearly", "unclear"
    ]
    
    if any(indicator in content.lower() for indicator in explanation_indicators):
        return "N/A"
    
    # Length validation
    if len(content) > 200:  # Too long for a simple field
        return "N/A"
    
    return content

# ================================================================== #
# 3. Enhanced Structured Data Extraction                            #
# ================================================================== #

def extract_structured_data(text: str) -> Dict[str, str]:
    """Extract predefined structured fields with enhanced accuracy."""
    if not text or not text.strip():
        return {
            "Applicant Name": "N/A",
            "Application Date": "N/A",
            "Income": "N/A",
            "Family Size": "N/A"
        }
    
    fields = ["Applicant Name", "Application Date", "Income", "Family Size"]
    result = {}
    
    for field in fields:
        result[field] = hybrid_extraction(text, field)
    
    return result

def extract_new_column(text: str, column_query: str) -> str:
    """Extract user-defined field using enhanced hybrid approach."""
    return hybrid_extraction(text, column_query)

# ================================================================== #
# 4. FIXED Advanced Q&A Chain with MultiQueryRetriever             #
# ================================================================== #

def create_enhanced_retrieval_qa_chain(retriever, llm):
    """Create advanced Q&A chain with multi-query retrieval and better prompting."""
    
    # Enhanced prompt with strict document adherence
    qa_prompt = ChatPromptTemplate.from_template("""
You are a precise document analysis assistant. Your job is to answer questions based STRICTLY on the provided document context.

DOCUMENT CONTEXT:
{context}

QUESTION: {input}

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information explicitly stated in the document context above
2. If the answer is not in the documents, respond: "This information is not available in the provided documents."
3. Do NOT make assumptions or add information not in the context
4. Do NOT use your general knowledge - stick to the documents only
5. Be specific and cite relevant details from the context
6. If multiple documents contain related information, synthesize clearly
7. Keep responses concise but complete

ANSWER:""")
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # FIXED: Create multi-query retriever without num_queries parameter
    try:
        # Create custom prompt for multi-query generation
        multi_query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 3 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help 
the user overcome some of the limitations of distance-based similarity search. 

Provide these alternative questions separated by newlines.

Original question: {question}"""
        )
        
        # Create multi-query retriever with custom prompt (FIXED)
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=multi_query_prompt
        )
        
        # Create the retrieval chain with multi-query
        retrieval_chain = create_retrieval_chain(multi_query_retriever, document_chain)
        
    except Exception as e:
        print(f"Failed to create MultiQueryRetriever: {e}")
        print("Falling back to standard retriever")
        
        # Fallback to standard retriever if MultiQueryRetriever fails
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# ================================================================== #
# 5. Query Analysis and Optimization                                #
# ================================================================== #

@functools.lru_cache(maxsize=256)
def analyze_and_optimize_query(query: str) -> Dict[str, Any]:
    """Analyze query and return optimization strategies."""
    query_lower = query.lower()
    
    analysis = {
        'original_query': query,
        'query_type': 'general',
        'is_summarization': False,
        'is_factual': False,
        'requires_aggregation': False,
        'optimized_queries': [query]
    }
    
    # Detect query types
    if any(word in query_lower for word in ['summarize', 'summary', 'overview', 'describe']):
        analysis['query_type'] = 'summarization'
        analysis['is_summarization'] = True
        # Add specific summarization queries
        analysis['optimized_queries'] = [
            query,
            "What are the main points in this document?",
            "What is the key information contained in these documents?"
        ]
    
    elif any(word in query_lower for word in ['how many', 'count', 'total', 'sum']):
        analysis['query_type'] = 'aggregation'
        analysis['requires_aggregation'] = True
    
    elif any(word in query_lower for word in ['what is', 'who is', 'when', 'where']):
        analysis['query_type'] = 'factual'
        analysis['is_factual'] = True
    
    return analysis

# ================================================================== #
# 6. Enhanced Q&A Response with Quality Control                     #
# ================================================================== #

def enhanced_qa_response(query: str, retriever, llm) -> str:
    """Enhanced Q&A with query optimization and answer validation."""
    if not query or not query.strip():
        return "Please provide a specific question about your documents."
    
    try:
        # Analyze query for optimization
        query_analysis = analyze_and_optimize_query(query)
        
        # Handle summarization queries specially
        if query_analysis['is_summarization']:
            return _handle_summarization_query(query, retriever, llm)
        
        # For factual queries, try direct extraction first
        if query_analysis['query_type'] in ['factual', 'aggregation']:
            try:
                direct_result = _try_direct_extraction(query, retriever)
                if direct_result and direct_result != "N/A":
                    return f"Based on the documents: {direct_result}"
            except Exception:
                pass
        
        # Use enhanced Q&A chain
        qa_chain = create_enhanced_retrieval_qa_chain(retriever, llm)
        
        # Try the original query first
        response = qa_chain.invoke({"input": query})
        answer = response.get('answer', "I couldn't process your question properly.")
        
        # Validate and improve answer quality
        answer = _validate_and_improve_answer(answer, query, query_analysis)
        
        return answer
        
    except Exception as e:
        error_msg = f"I encountered an error while processing your question: {str(e)}"
        print(f"Q&A Error: {e}")
        return error_msg

def _handle_summarization_query(query: str, retriever, llm) -> str:
    """Special handling for document summarization queries."""
    try:
        # Get more documents for comprehensive summary
        docs = retriever.get_relevant_documents(query, k=5)
        
        if not docs:
            return "I don't have any documents to summarize."
        
        # Combine document content
        combined_content = "\n\n".join([doc.page_content for doc in docs])
        
        # Summarization-specific prompt
        summary_prompt = f"""Provide a comprehensive summary of the following document content:

DOCUMENT CONTENT:
{combined_content[:3000]}  # Limit to avoid token limits

INSTRUCTIONS:
- Create a clear, structured summary
- Include the main points and key information
- Be comprehensive but concise
- Organize information logically
- Only include information from the documents provided

SUMMARY:"""
        
        try:
            response = ollama.chat(
                model='smollm',
                messages=[{'role': 'user', 'content': summary_prompt}],
                options={
                    'temperature': 0.1,  # Slightly higher for better flow
                    'num_predict': 500,  # Longer for summaries
                    'top_p': 0.3
                }
            )
            
            summary = response['message']['content'].strip()
            
            if summary and len(summary) > 50:  # Ensure substantial content
                return summary
            else:
                return "I was unable to generate a comprehensive summary from the available documents."
                
        except Exception as e:
            print(f"Summarization error: {e}")
            return "I encountered an error while trying to summarize the documents."
            
    except Exception as e:
        return f"I couldn't access the documents for summarization: {str(e)}"

def _try_direct_extraction(query: str, retriever) -> str:
    """Try direct extraction for specific data types."""
    try:
        docs = retriever.get_relevant_documents(query)
        if docs:
            combined_text = "\n".join([doc.page_content for doc in docs[:3]])
            return hybrid_extraction(combined_text, query)
    except Exception:
        pass
    return "N/A"

def _validate_and_improve_answer(answer: str, query: str, query_analysis: Dict) -> str:
    """Validate and improve answer quality."""
    if not answer or answer.strip() == "":
        return "I couldn't find relevant information in your documents to answer this question."
    
    # Remove common LLM artifacts
    answer = answer.replace("Based on the provided context,", "")
    answer = answer.replace("According to the documents,", "")
    answer = answer.strip()
    
    # Check for non-substantive answers
    non_substantive_phrases = [
        "I don't have enough information",
        "The context doesn't provide",
        "I cannot determine",
        "It's not clear from"
    ]
    
    if any(phrase in answer for phrase in non_substantive_phrases):
        return "This information is not available in the provided documents."
    
    # Ensure minimum quality for substantive answers
    if len(answer) < 20:
        return "I found limited information. Please try asking a more specific question about your documents."
    
    # For summarization, ensure comprehensive response
    if query_analysis['is_summarization'] and len(answer) < 100:
        return "I was unable to generate a comprehensive summary. The documents may not contain sufficient content for summarization."
    
    return answer

# ================================================================== #
# 7. Query Classification for Better Performance                    #
# ================================================================== #

@functools.lru_cache(maxsize=256)
def classify_query_type(query: str) -> str:
    """Enhanced query classification for performance optimization."""
    query_lower = query.lower()
    
    # Detailed classification
    if any(word in query_lower for word in ['summarize', 'summary', 'overview', 'describe', 'tell me about']):
        return 'summarization'
    elif any(word in query_lower for word in ['email', 'mail', '@', 'e-mail']):
        return 'email'
    elif any(word in query_lower for word in ['phone', 'number', 'call', 'contact', 'mobile']):
        return 'phone'
    elif any(word in query_lower for word in ['date', 'when', 'time', 'day']):
        return 'date'
    elif any(word in query_lower for word in ['income', 'salary', 'money', '$', 'earn']):
        return 'income'
    elif any(word in query_lower for word in ['name', 'applicant', 'who', 'person']):
        return 'name'
    elif any(word in query_lower for word in ['how many', 'count', 'total', 'number of']):
        return 'count'
    else:
        return 'general'
