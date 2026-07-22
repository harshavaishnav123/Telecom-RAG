import pandas as pd
import joblib
import torch
import shap
import io
from pypdf import PdfReader
from unsloth import FastLanguageModel

## creates dynamic prompt by shap ,as we can take static features which cause anaomaly by model which has highest weight or priority based how it had trained ,
## it creates dynamic prompt based on exact metric causing the anomaly for specific row/test case .
def build_dynamic_kpi_description(row, prediction, top_features_dict):
    status = "anomalous" if prediction == 1 else "normal"
    # dynamic list of anomaly features
    observed_kpis = ""
    for feature_name, shap_val in top_features_dict.items():
        observed_kpis += f"- {feature_name}: {row[feature_name]} (impact: {shap_val:.4f})\n    "
    ## it the description which includes the anomaly status and features which are causing anomaly(by xgboost features)
    return f"""
        The anomaly detector has identified a {status} LTE KPI sample.

        The primary metrics triggering this network degradation are:
        {observed_kpis}
    """

## Query genaration by Qwen2.5-7B model , which can be sent for RAG retrival.
def rewrite_query(description, model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": """
            You are a senior LTE/5G Radio Access Network (RAN) optimization engineer.

            Your task is NOT to diagnose the network problem.
            Your task is ONLY to convert the observed KPI behaviour into a semantic retrieval query for a telecom Retrieval-Augmented Generation (RAG) system.

            The retrieval query will be used to search 3GPP and O-RAN specifications for relevant procedures, protocol behaviour, and possible root causes.

            Instructions:

            • Interpret the KPI behaviour instead of copying KPI names.
            • Translate KPI observations into telecom concepts.
            • Do NOT mention feature names such as:
            - mac_ul_brate
            - mac_dl_brate
            - mac_ul_bsr
            - mac_dl_cqi_offset
            - mac_ul_snr_offset
            - mac_dl_buffer
            - mac_dl_ok
            - mac_dl_nok
            - mac_ul_ok
            - mac_ul_nok
            - mac_rnti
            • Do NOT mention exact numerical values.
            • Do NOT mention UE IDs, RNTI values, timestamps or identifiers.
            • Do NOT provide a diagnosis or solution.

            Instead describe the network behaviour using terminology commonly found in LTE/5G standards.

            Use concepts such as:

            - uplink throughput
            - downlink throughput
            - throughput degradation
            - channel quality indication (CQI)
            - degraded channel quality
            - CSI reporting
            - link adaptation
            - modulation and coding scheme (MCS)
            - uplink radio quality
            - HARQ retransmissions
            - MAC scheduling
            - scheduler inefficiency
            - radio resource allocation
            - resource block utilization
            - buffer occupancy
            - Buffer Status Reporting (BSR)
            - radio resource congestion
            - interference
            - mobility
            - handover
            - RRC procedures
            - MAC procedures
            - PHY procedures
            - radio link performance
            - cell edge conditions

            Generate ONE semantic retrieval query between 40 and 80 words.

            The query should maximize semantic similarity with 3GPP and O-RAN specifications.

            Return ONLY the retrieval query.
            """
        },
        {
            "role": "user",
            "content": description
        }
    ]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

    inputs = tokenizer(text,return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs,max_new_tokens=120,do_sample=False)

    generated = outputs[0][inputs.input_ids.shape[1]:]

    query = tokenizer.decode(generated,skip_special_tokens=True).strip()

    return query
def get_query(filename,file_bytes):
    ## data loading and cleaning
    if filename.endswith('.csv'):
        # io.BytesIO wraps the raw bytes so pandas can read it like a real file
        df = pd.read_csv(io.BytesIO(file_bytes),sep=';')
    # loading the model and the tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=8192,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    ## cols which are in Full KPI file which are not usefull.
    drop_cols = [
        "ue_ident", "id_ue", "timestamp", 'mac_pci', 'mac_cc_idx', 'mac_dl_ri', 
        'mac_dl_pmi', 'mac_ul_rssi', 'mac_fec_iters', 'mac_dl_mcs_samples', 
        'mac_ul_mcs', 'mac_ul_mcs_samples', 'phy_ul_n', 'phy_ul_pusch_tpc', 
        'phy_dl_pucch_tpc', 'rf_o', 'rf_u', 'rf_l', 'phy_ul_pucch_ni', 'phy_dl_mcs'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # feature preprocessing:
    ## load label_encoder which is used during training
    le = joblib.load('..../label_encoder.pkl')
    df["mob_pattern"] = le.transform(df["mob_pattern"]) 
    X=df
    ## load standardscaler from training file
    scaler = joblib.load('..../scaler.pkl')
    X_scaled = scaler.transform(X) 
    
    # inference model loading
    xgb_model = joblib.load('..../model.pkl')
    predictions = xgb_model.predict(X_scaled)
    
    # initialize shap explainer ,which targets our xgboost model
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)
    
    row_idx = 0
    pred = predictions[0]
    
    if pred == 1:
        print(f"Anomaly Detected for filename {filename}.")
        
        # map raw array indices into input pandas feature labels
        row_shap = pd.Series(shap_values[row_idx], index=X.columns)
        
        # select the top 4 absolute contributors towards the anomaly ,so that by this we can get the query for our retrival
        top_4 = row_shap.abs().nlargest(4).index
        top_4_values = row_shap[top_4].to_dict()
    
        # generate dynamic prompt
        description = build_dynamic_kpi_description(X.iloc[row_idx], pred, top_4_values)        
        
        # rca query for our input KPI data 
        rca_query= rewrite_query(description, model, tokenizer)
        return rca_query,True
    else:
        return f"uploaded KPI file {filename} is classified as Normal. Skipping RCA Retrieval pipeline as there is no Anomaly detected ,Everything is fine from your KPI data.",False