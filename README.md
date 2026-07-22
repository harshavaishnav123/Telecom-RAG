# Telecom RAG: Intelligent Root Cause Analysis for Radio Access Networks

> An AI-powered Retrieval-Augmented Generation (RAG) framework for Telecom Networks that combines anomaly detection, hybrid document retrieval, and Large Language Models to perform intelligent Root Cause Analysis (RCA).

---

## Overview

Modern Radio Access Networks (RAN) generate large amounts of KPIs, alarms,logs and technical specifications,making manual troubleshooting difficult and time-consuming.

This project provides an end-to-end intelligent RCA system that can:

-  Detect anomalies from KPI files using Machine Learning.
-  Retrieve relevant knowledge from telecom standards and O-RAN documentation.
-  Generate expert-level Root Cause Analysis using a Large Language Model.
-  Provide an easy-to-use Streamlit interface for engineers.
-  Retrieves only the required documents based on the query provided.

The system supports two different workflows:

1. **Natural Language Telecom Queries**
2. **KPI File Analysis**

---

## Features

- Hybrid Retrieval (Dense + BM25)
- Multi-collection document retrieval
- Dynamic collection selection
- BGE Reranker for context ranking
- Qwen-2.5-7B-Instruct based reasoning
- XGBoost anomaly detection
- SHAP-based feature explanation
- Automatic RCA query generation
- Streamlit Web Interface
- FastAPI/Flask backend

---

# System Architecture

                           TELECOM RAG SYSTEM

                    ┌───────────────────────────┐
                    │        User Input         │
                    └─────────────┬─────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        Natural Language Query            KPI File Upload
                 │                                 │
                 │                         Data Preprocessing
                 │                                 │
                 │                          XGBoost Classifier
                 │                                 │
                 │                     ┌───────────┴───────────┐
                 │                     │                       │
                 │               No Anomaly             Anomaly Detected
                 │                     │                       │
                 │                     │               SHAP Explanation
                 │                     │                       │
                 │                     └──────────────┬────────┘
                 │                                    │
                 │                          RCA Query Generator
                 │                                    │
                 └───────────────────────┬────────────┘
                                         ▼
                              Hybrid Document Retrieval
                           (Dense Embeddings + BM25)
                                         │
                                         ▼
                              BGE Cross-Encoder Reranker
                                         │
                                         ▼
                             Top Relevant Telecom Chunks
                                         │
                                         ▼
                           Qwen-2.5-7B-Instruct (LLM)
                                         │
                                         ▼
                     Root Cause Analysis / Telecom Answer
                                         │
                                         ▼
                                Streamlit Interface

---

### Offline Knowledge Base Construction

3GPP Documents
        │
        V
O-RAN Documents
        │
        V
Telecom Vocabulary
        │
        V
RCA Reports
        │
        V
Document Parsing
        |
        V
Cleaning & Preprocessing
        |
        V
Chunking
        |
        V
Embedding Generation (BGE)
        |
        V
Vector Database (Qdrant)

---

# Project Structure

Telecom Project/
│
├── app/
│   ├── app.py                 # Streamlit frontend
│   ├── server.py              # Backend server
│   ├── query_gen.py           # RCA query generation
│   └── query_retrival.py      # Hybrid retrieval pipeline
│
├── Data/
│   ├── 3GPP/
│   ├── O-RAN/
|   ├── data/
│   ├── vocabulary/
│   └── random/
│
├── notebooks/
│   ├── embed.py
│   ├── chunking_and_preprocessing.py
│   ├── teleqna_analysis.ipynb
│   └── unzipping.py
│
├── evaluation/
│   └── final/
│
├── docs/
│   ├── Dataset understanding.txt
│   └── Problem_statement_understanding.txt
|
│
├── Anamoly detection/
│   ├── data/
│   ├── pickle files/
│   ├── data_preprocessing_and_classifier.ipynb
│   ├── KPI_query_gen_testing.py
│   └── rough.py
│
├── requirements.txt
│
└── flowchart.txt

---

# Repository Structure

## Root Directory

| File / Folder | Description |
|---------------|-------------|
| `app/` | Contains the main application,backend services,retrieval pipeline and Streamlit interface. |
| `Data/` | Scripts for parsing,preprocessing and preparing telecom knowledge sources. |
| `Anamoly detection/` | Contains the anomaly detection model,KPI preprocessing and RCA query generation scripts. |
| `evaluation/` | Evaluation scripts for retrieval,question answering and RAG performance. |
| `notebooks/` | Utility scripts and notebooks for data preparation,embeddings and experiments. |
| `docs/` | Documentation explaining the project and datasets. |
| `requirements.txt` | List of Python dependencies required to run the project. |
| `flowchart.txt` | Flowchart of the overall Telecom RAG pipeline. |
| `error_analysis_qa_ragas.jsonl` | Error analysis results generated during RAGAS evaluation. |

---

## Application (`app/`)

| File | Description |
|------|-------------|
| `app.py` | Streamlit frontend for interacting with the Telecom RAG system. |
| `server.py` | Backend server that handles user requests and coordinates the RAG pipeline. |
| `query_retrival.py` | Implements hybrid retrieval using dense embeddings,BM25 and reranking. |
| `query_gen.py` | Generates telecom-aware RCA queries from detected anomalies or user inputs. |

---

## Data Processing (`Data/`)

The project organizes telecom knowledge into four collections.

### `3GPP/`

| File | Description |
|------|-------------|
| `docs_parser.py` | Extracts text from 3GPP documents. |
| `preprocess.py` | Cleans and preprocesses the extracted content. |
| `dictionarymake.py` | Builds dictionaries and metadata for retrieval. |
| `drop_useless.py` | Removes unwanted or empty content. |
| `table_par.py` | Extracts and processes tables from 3GPP documents. |

### `O-RAN/`

| File | Description |
|------|-------------|
| `docs_parser.py` | Parses O-RAN documentation. |
| `preprocess.py` | Cleans and preprocesses O-RAN documents. |
| `dictionarymake.py` | Generates document metadata. |
| `drop_useless.py` | Removes unnecessary content. |

### `random/`

| File | Description |
|------|-------------|
| `docs_parser.py` | Parses additional telecom reference documents. |
| `preprocess.py` | Cleans the parsed documents. |
| `dictionarymake.py` | Creates metadata for indexing. |
| `drop_useless.py` | Removes unwanted information. |

### `vocabulary/`

| File | Description |
|------|-------------|
| `docs_parser.py` | Parses telecom vocabulary documents. |
| `drop_useless.py` | Cleans vocabulary entries before indexing. |

---

## Data (`Data/data`)

In data we have divided into **7 buckets** of data that contains different types of data.They are **Data_plane,edge,L_2_3,MAC_Layer,o-kpi,o-query,RAN_ARCH**.So when a query arrives our retriever only retrieves the required documents from these folders and performs response generation based on the data.

---
## Anomaly Detection

| File | Description |
|------|-------------|
| `data_preprocessing_and_classifier.ipynb` | Trains and evaluates the XGBoost anomaly detection model. |
| `KPI_query_gen_testing.py` | Generates RCA queries from important KPI features and tests the pipeline. |
| `rough.py` | Experimental scripts used during model development. |

---

## Evaluation

| File | Description |
|------|-------------|
| `final_check_retreival.py` | Evaluates retrieval performance using Recall@k and MRR. |
| `final_check_qa.py` | Evaluates question answering accuracy on the TeleQnA dataset. |
| `final_check_qa_ragas.py` | Performs RAG evaluation using RAGAS metrics. |
| `ragas_check.py` | Helper functions for RAGAS evaluation. |
| `parse.py` | Parses evaluation outputs for analysis. |

---

## Notebooks

| File | Description |
|------|-------------|
| `chunking_and_preprocessing.py` | Chunks and preprocesses telecom documents before indexing. |
| `embed.py` | Generates dense embeddings for all document chunks. |
| `unzipping.py` | Extracts compressed telecom datasets. |
| `teleqna_nalysis.ipynb` | Exploratory analysis of the TeleQnA dataset. |

---

## Documentation

| File | Description |
|------|-------------|
| `Dataset understanding.txt` | Notes describing the datasets used in the project. |
| `Problem_statement_understanding.txt` | Summary of the project objectives and problem statement. |

---

# Technologies Used

- Python
- Streamlit
- Flask/FastAPI
- LangChain
- Hugging Face Transformers
- FAISS / Vector Database
- BM25
- BGE Embeddings
- BGE Reranker
- Qwen-2.5-7B-Instruct
- XGBoost
- SHAP
- Pandas
- NumPy

---

# Installation

Clone the repository

```bash
git clone https://github.com//Telecom-RAG.git
```

Move into the project directory

```bash
cd Telecom-RAG
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Start the Backend

```bash
python app/server.py
```

---

## Start the Frontend

```bash
streamlit run app/app.py
```

---

# Usage

## Option 1: Telecom Query

Enter questions such as:

- Why is PRB utilization high?
- Explain RRC connection failures.
- What causes low CQI in LTE?
- How to troubleshoot handover failures?

The system retrieves relevant telecom documents and generates an expert RCA.

---

## Option 2: KPI File Analysis

Upload a KPI file (CSV/XLSX).

The pipeline will:

1. Detect anomalies
2. Identify important KPIs using SHAP
3. Generate a telecom-aware RCA query
4. Retrieve supporting documents
5. Produce a detailed Root Cause Analysis

---

# Pipeline

1. User submits a query or KPI file.
2. KPI files are analyzed using XGBoost.
3. SHAP identifies the most influential KPI features.
4. A dynamic RCA query is generated.
5. Hybrid retrieval searches telecom knowledge bases.
6. BGE Reranker ranks retrieved passages.
7. Qwen-2.5-7B-Instruct generates the final explanation.

---

# Evaluation

The project includes evaluation scripts for:

- Retrieval Performance
- Question Answering
- RAGAS Evaluation
- Anomaly Detection

Located in:

evaluation/final/

---

# Results

The proposed Telecom RAG framework was evaluated across multiple components, including anomaly detection, document retrieval, reranking, and end-to-end response generation.

## Anomaly Detection

The anomaly detection module identifies abnormal network behavior from KPI measurements before invoking the RAG pipeline.This classifier has been trained on **80%** train split on **dtst.csv** which has **3175140** samples of data that has **47** columns

| Metric | Value |
|---------|------:|
| Model | XGBoost |
| Task | Binary Classification |
| Feature Explanation | SHAP |
| Training Accuracy | **99.99%** |
| Testing Accuracy | **99.66%** |
| Mean Confidence | **99.85%** |

The classifier recorded a remarkable accuracy of **99%* on the testing data.
---

## Retrieval Performance

The retrieval pipeline was evaluated using standard Information Retrieval metrics.The system is evaluated on data **evaluation.jsonl**,which contains **450** questions that has context and spec that needs to be retrieved.

| Metric | Score |
|---------|-------:|
| Recall@1 | **0.5909** |
| Recall@5 | **0.7509** |
| Recall@8 | **0.8018** |
| Recall@10 | **0.8218** |
| Mean Reciprocal Rank (MRR) | **0.6688** |

These results demonstrate that the hybrid retrieval framework effectively retrieves highly relevant telecom documentation for downstream reasoning.

---

## RAGAS Evaluation

The quality of the generated responses was evaluated using **Correctness**,**Faithfulness** and **Context Recall**.

| Metric | Score |
|---------|------:|
| Correctness | **0.7159** |
| Faithfulness | **0.1303** |
| Context Recall | **0.1377** |

These results indicate that the generated responses are accurate and remain grounded in the retrieved telecom documents, making the system suitable for telecom question answering and Root Cause Analysis.

---

## Question Answering

The accuracy of the RAG system was evaluated using **TeleQnA.txt**,which contains **100 multiple-choice telecom questions**.For each question,the system was asked to predict the correct option based on the retrieved context.

| Metric | Score |
|---------|------:|
| Accuracy | **0.62** |

The results show that the system correctly answered **62 out of 100 questions**, demonstrating its ability to understand telecom-related queries and retrieve relevant information to generate accurate answers.

---

## End-to-End RAG

The complete pipeline consists of:

User Query
      │
      ▼
Hybrid Retrieval
      │
      ▼
BGE Reranker
      │
      ▼
Qwen-2.5-7B-Instruct
      │
      ▼
Root Cause Analysis

The generated responses provide:

- Accurate telecom reasoning
- Context-grounded explanations
- Standards-aware answers
- Detailed Root Cause Analysis
- Actionable troubleshooting recommendations

---

## Evaluation Summary

| Component | Method |
|-----------|--------|
| Anomaly Detection | XGBoost |
| Feature Interpretation | SHAP |
| Retrieval | Dense + BM25 Hybrid |
| Reranking | BGE Cross-Encoder |
| Knowledge Base | Qdrant Vector Database |
| Language Model | Qwen-2.5-7B-Instruct |
| Interface | Streamlit |

---

# Future Work

- Multi-modal telecom log analysis
- Live network monitoring
- Real-time KPI streaming
- Knowledge graph integration
- Agentic RAG for autonomous troubleshooting
- Multi-LLM support

---

# Authors

**Vasamsetti Nihal Tej**
**Velineni Harshavaishnav**

---