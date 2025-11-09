"""
LangChain Agent with DuckDuckGo Search
"""
import streamlit as st
from agent import create_agent, execute_query

st.set_page_config(page_title="LangChain Agent", page_icon="🔍")

# Session state
if "agent" not in st.session_state:
    st.session_state.agent = None

st.title("🔍 LangChain Agent with Web Search")

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    
    api_key = st.text_input(
        "Hugging Face API Key",
        type="password",
        value=st.session_state.get("api_key", "")
    )
    
    model = st.selectbox(
        "Model",
        ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"],
        index=0
    )
    
    if st.button("Initialize", type="primary"):
        if api_key:
            with st.spinner("Initializing..."):
                try:
                    st.session_state.agent = create_agent(api_key, model)
                    st.session_state.api_key = api_key
                    st.success("Ready! ✅")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Please enter API key")

# Main area
if st.session_state.agent:
    st.info("💡 The agent can search the web for current information (2025) using DuckDuckGo!")
    query = st.text_area("Ask a question:", height=100, placeholder="Ask about current events, recent news, or anything from 2025...")
    
    if st.button("🔍 Search & Answer"):
        if query:
            with st.spinner("Searching the web and thinking..."):
                try:
                    response = execute_query(st.session_state.agent, query)
                    st.markdown("### Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Enter your API key and click Initialize to get started")
    st.markdown("""
    ### Features:
    - 🤖 Uses Llama 3.1 models for reasoning
    - 🔍 **Searches DuckDuckGo for current information (2025)**
    - 📰 Can answer questions about recent events and news
    - 🌐 Accesses real-time web data, not just training data (up to 2021)
    """)
