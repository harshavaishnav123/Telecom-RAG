# app.py
import streamlit as st
import requests

# setting the streamlit file page configuration
st.set_page_config(page_title="RAG Analytics Dashboard", page_icon="📊", layout="centered")

# setting the title
st.title("📊 RAG Analytics & Query Interface")
st.markdown("Select an input type to submit your request to the backend pipeline.")

input_type = st.radio("Choose Input Type:",options=["Plain Text Query", "KPI File Upload (Note that the KPI file must be in the standard 46 col format)"],horizontal=True)

# initializing the backend url
BACKEND_URL = "http://localhost:5000/api/process"

if input_type == "Plain Text Query":
    st.subheader("Submit a Query")
    # writing into the user query
    user_query = st.text_area("Enter your prompt or analytical question:", placeholder="e.g., High BSR scheduling bottlenecks")
    
    # writing the button for submitting the query
    if st.button("Submit Query", type="primary"):
        if not user_query.strip():
            st.error("Please enter a valid query before submitting.")
        else:
            with st.spinner("Processing your query through RAG pipeline..."):
                try:
                    # api call into the backend for the response of the query
                    payload = {"type": "query", "data": user_query}
                    response = requests.post(BACKEND_URL, json=payload)
                    
                    if response.status_code == 200:
                        st.success("Response Received!")
                        st.markdown("### 🤖 Backend Expert Analysis:")
                        st.write(response.json().get("result", "No context returned."))
                    else:
                        st.error(f"Backend Error ({response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Is it running?")

else:
    # subheader for uploading kpi file
    st.subheader("📁 Upload KPI File")
    uploaded_file = st.file_uploader("Upload your KPI document in the start 46 col format (in CSV):", type=["csv"])
    
    # button for processing the file
    if st.button("Process File", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a file before clicking process.")
        else:
            with st.spinner("Uploading and analyzing KPI file..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"type": "kpi_file"}
                    # api call into thr backend for the response for a kpi file uploaded
                    response = requests.post(BACKEND_URL, files=files, data=data)
                    
                    if response.status_code == 200:
                        st.success("File Processed Successfully!")
                        st.markdown("### 📈 Extracted Performance Analytics:")
                        st.write(response.json().get("result", "No analysis returned."))
                    else:
                        st.error(f"Backend Error ({response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Is it running?")