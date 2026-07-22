# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from query_gen import get_query
from query_retrival import make_retrival

# importing llm router function from your core logic script
app = Flask(__name__)
CORS(app)

# route for processing the query or the kpi file
@app.route('/api/process', methods=['POST'])
def process_request():
    # ---- SCENARIO A: KPI File Upload ----
    if 'file' in request.files:
        uploaded_file = request.files['file']
        file_name = uploaded_file.filename
        file_bytes = uploaded_file.read()
        
        # initializing default response string to prevent unbound variable crashes
        backend_result = "File uploaded, but no retrieval query was generated."
        
        result, b = get_query(file_name, file_bytes)
        if b:
            # type 0 indicates it came from a kpi text block extraction
            backend_result = make_retrival(result, 0) 
            
        return jsonify({
            "status": "success",
            "result": backend_result # standardized key to match streamlit query display
        })
        
    # ---- SCENARIO B: Plain Text Query ----
    elif request.is_json:
        json_data = request.get_json()
        user_query = json_data.get('data')
        # making the retrieval for llm output
        llm_output = make_retrival(user_query, 1)
        
        return jsonify({
            "status": "success",
            "result": llm_output
        })

    else:
        return jsonify({"status": "error", "message": "Invalid request format"}), 400

if __name__ == '__main__':
    # set to 0.0.0.0 so your streamlit instance can access it over local network ports
    app.run(host='0.0.0.0', port=5000, debug=True)