import streamlit as st
import requests

# URL của FastAPI Backend (đã cấu hình ở api/main.py)
API_URL = "http://localhost:8000/verify"

st.set_page_config(page_title="Research AI Agent", layout="wide")

st.title("🔬 Scientific Claim Verifier Agent")
st.markdown("Enter a scientific statement to let the AI Agent search papers, extract knowledge, and verify facts.")

claim_input = st.text_area("Enter your claim:", value="Graph neural networks achieve better performance than CNN on molecular property prediction")
top_k = st.slider("Number of papers to fetch", min_value=1, max_value=10, value=5)

if st.button("Verify Claim"):
    with st.spinner("Agent is searching, reading, and reasoning..."):
        try:
            response = requests.post(API_URL, json={"claim": claim_input, "top_k": top_k})
            response.raise_for_status()
            data = response.json()
            
            # Hiển thị kết quả
            st.subheader(f"Verdict: {data['verdict']}")
            
            if data['verdict'] == "SUPPORTED":
                st.success(data['reasoning'])
            elif data['verdict'] == "REFUTED":
                st.error(data['reasoning'])
            else:
                st.warning(data['reasoning'])
                
            # Hiển thị trích dẫn
            st.markdown("### 📚 Citations")
            for i, p in enumerate(data['citations'], 1):
                st.markdown(f"**[{i}]** {p['title']} ({p['year']})")
                
        except Exception as e:
            st.error(f"Error connecting to backend API: {e}")