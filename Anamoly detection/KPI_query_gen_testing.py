import pandas as pd
import joblib
import shap  
from unsloth import FastLanguageModel
import torch

llm, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(llm)
## It generates dynamic description
def build_dynamic_kpi_description(row, prediction, top_features_dict):
    
    status = "anomalous" if prediction == 1 else "normal"
    
    # dynamically builds the breakdown of the most impacted features
    observed_kpis = ""
    for feature in top_features_dict.keys():
        observed_kpis += f"- {feature}: {row[feature]}\n    "

    return f"""
    The anomaly detector has identified a {status} LTE KPI sample.
    
    The top network metrics contributing to this detection are:
    {observed_kpis}
    """
## It generates description based on mac_dl_brate,mac_ul_brate,mac_dl_cqi_offset,mac_ul_snr_offset
def build_kpi_description(row, prediction):

    status = "anomalous" if prediction == 1 else "normal"

    return f"""
    The anomaly detector has identified a {status} LTE KPI sample.

    Observed KPI values:
    - Downlink Bitrate (mac_dl_brate): {row['mac_dl_brate']}
    - Uplink Bitrate (mac_ul_brate): {row['mac_ul_brate']}
    - Downlink CQI Offset (mac_dl_cqi_offset): {row['mac_dl_cqi_offset']}
    - Uplink SNR Offset (mac_ul_snr_offset): {row['mac_ul_snr_offset']}
    """
## It genarates dynamic query based on description provided
def rewrite_query(description):
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
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(llm.device)

    outputs = llm.generate(**inputs,max_new_tokens=120,do_sample=False)

    generated = outputs[0][inputs.input_ids.shape[1]:]

    query = tokenizer.decode( generated, skip_special_tokens=True).strip()

    return query

df = pd.read_csv("kpi_stream.csv",sep=';')

drop_cols = [
    "ue_ident",
    "id_ue",
    "timestamp",
    'mac_pci',
    'mac_cc_idx',
    'mac_dl_ri',
    'mac_dl_pmi',
    'mac_ul_rssi',
    'mac_fec_iters',
    'mac_dl_mcs_samples',
    'mac_ul_mcs',
    'mac_ul_mcs_samples',
    'phy_ul_n',
    'phy_ul_pusch_tpc',
    'phy_dl_pucch_tpc',
    'rf_o',
    'rf_u',
    'rf_l',
    'phy_ul_pucch_ni',
    'phy_dl_mcs'
    ]

df = df.drop(columns=drop_cols)

le=joblib.load('/home/shreya/Nihal/label_encoder.pkl')

df["mob_pattern"] = le.transform(df["mob_pattern"])

# attack_labels = {
#     "dos-hulk-C",
#     "ddos-ripper-C",
#     "slowloris-C",
#     "portscan"
# }

# df["anomaly"] = (
#     df["label"]
#     .isin(attack_labels)
#     .astype(int)
# )

# X=df.drop(columns=['label','anomaly'])

X=df

scaler=joblib.load('/home/shreya/Nihal/scaler.pkl')

X_scaled = scaler.transform(X)

model=joblib.load('/home/shreya/Nihal/model.pkl')

Y=model.predict(X_scaled)

explainer = shap.TreeExplainer(model)

rows=[]

for i in range(len(Y)):
    description = build_kpi_description(X.iloc[i],Y[i])

    query = rewrite_query(description)

    print(query)

    # Only generate explanations if an anomaly is detected (saves compute)
    if Y[i] == 1:
        # Initialize SHAP explainer (we mostly use TreeExplainer for XGBoost/RF, LinearExplainer for Linear, or Explainer for black-box)
        # Calculate SHAP values for our specific row
        shap_values = explainer(X_scaled[i : i + 1])
        # Map back to original feature names
        feature_importance = pd.Series(shap_values.values[0], index=X.columns)
        # Take the absolute value to get magnitude of impact, then sort
        top_4_contributors = feature_importance.abs().nlargest(4).to_dict()
        # Build dynamic prompt using ONLY the metrics that caused the alarm for anomaly
        description = build_dynamic_kpi_description(X.iloc[i], Y[i], top_4_contributors)
    else:
       # it is normal without anomaly
        description = f"The anomaly detector found a normal LTE KPI sample."

    # genearte retrival query
    query = rewrite_query(description)
    print(query)
    rows.append(query)
out=pd.DataFrame(rows)
out.to_csv('queries.csv')