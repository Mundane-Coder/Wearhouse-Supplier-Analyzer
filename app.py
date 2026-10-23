import streamlit as st  # type: ignore[import-not-found]
from rag_chain import index_catalog, ask_catalog, is_catalog_indexed
import tempfile
import os
import time


st.set_page_config(page_title="Supplier Catalog Analyzer", page_icon="📦", layout="centered")
st.title("📦 Supplier Catalog Analyzer")
st.caption("Upload a supplier PDF catalog and ask questions about products, prices, and SKUs.")

# if st.session_state.get("query_count", 0) >= 3:
#     st.warning("Demo limit reached. Contact me for full access.")
#     st.stop()

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("Upload Catalog")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File too large. Maximum size is 10MB.")
            st.stop()

        # Basic check for PDF file signature
        header = uploaded_file.read(4)
        uploaded_file.seek(0)  # Reset file pointer after reading
        if header != b"%PDF":
            st.error("Invalid file. Please upload a valid PDF.")
            st.stop()

        if st.button("Index Catalog", type="primary"):
            with st.spinner("Indexing catalog..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    count = index_catalog(tmp_path)
                    st.session_state["indexed"] = True
                    st.session_state["catalog_name"] = uploaded_file.name
                    st.success(f"Indexed {count} chunks")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp_path)

    # For demo purposes, auto-index the sample catalog if no file is uploaded
    if not uploaded_file and not st.session_state.get("indexed"):
        count = index_catalog("sample_catalog.pdf")
        st.session_state["indexed"] = True
        st.session_state["catalog_name"] = "sample_catalog.pdf (demo)"

    if st.session_state.get("indexed"):
        st.info(f"Active: {st.session_state.get('catalog_name', 'catalog')}")

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "What USB-C cables are available?",
        "Show products under $5",
        "What wireless chargers do you have?",
        "Compare all power banks",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state["example_q"] = q

# --- Main: Chat ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(s)

# Handle example question click
default_input = st.session_state.pop("example_q", "")

if question := st.chat_input("Ask about the catalog...", ):
    default_input = question

if default_input:
    # Basic rate limiting to prevent spamming the LLM
    if "last_query_time" in st.session_state:
        elapsed = time.time() - st.session_state["last_query_time"]
        if elapsed < 3:  # 3 seconds between queries
            st.warning("Please wait before sending another message.")
            st.stop()
    st.session_state["last_query_time"] = time.time()

    if not st.session_state.get("indexed") and not is_catalog_indexed():
        st.warning("Please upload and index a catalog first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": default_input})
        with st.chat_message("user"):
            st.markdown(default_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching catalog..."):
                result = ask_catalog(default_input)
            st.markdown(result["answer"])
            if result.get("sources"):
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.caption(s)

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", [])
        })

        st.session_state["query_count"] = st.session_state.get("query_count", 0) + 1
        st.rerun()