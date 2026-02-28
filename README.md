# REAL-TIME TRANSACTION ANOMALY DETECTION WITH CONCEPT DRIFT HANDLING

##  Overview

This project implements a real-time transaction anomaly detection system using **online machine learning**.  
It continuously processes streaming transaction data, detects anomalies using an unsupervised model, and adapts to changing patterns using concept drift detection.

Unlike traditional batch ML systems, this model learns incrementally and updates itself with every new transaction.

---

##  Problem Statement

Fraud detection systems deployed in real-world financial environments face two major challenges:

1. **Evolving Fraud Patterns (Concept Drift)**  
   Fraud behavior changes over time, making static models unreliable.

2. **Real-Time Decision Requirements**  
   Financial transactions require immediate risk scoring.

This project addresses both challenges using online machine learning and drift detection.

---

##  Solution Approach

The system combines:

- **Half-Space Trees (River)** → Online anomaly detection  
- **ADWIN Drift Detector** → Detects distribution changes in streaming data  
- **Feature Engineering** → Rolling statistics (global + per-user)  
- **Streamlit Dashboard** → Live monitoring UI  

---

##  System Architecture

1. Simulated transaction stream.  
2. Online Features generation (rolling means, user stats)  
3. Online anomaly scoring.  
4. Drift detection monitoring.  
5. Real-time dashboard visualization.  

---

##  Dashboard Features

-  Live anomaly score tracking.  
-  Red alerts for detected anomalies.  
-  Green indicator for normal transactions.  
-  Anomaly scores over time.
-  Live alerts panel.
-  Rolling global vs user behavior comparison.   
- KPI metrics:
   - Total transactions
   - Total alerts
   - Maximum anomaly score 
   - Drift detections
-  Per-user risk summary table.  
-  Concept drift detection indicator.

---

##  Tech Stack

- Python 3.10  
- River (online machine learning)  
- Streamlit (monitoring dashboard)  
- Plotly (interactive visualization)  
- Pandas / NumPy  

---

##  How to Run 

### 1-  Create virtual environment
```bash
  py -3.10 -m venv fraud_env
  fraud_env\Scripts\activate
```

### 2- Install dependencies
```bash
  pip install streamlit pandas numpy plotly river
```

### 3- Run the dashboard
```bash
  streamlit run streamlit_app.py
```

---

##  Example Outputs

- High anomaly scores during fraud spikes.  
- Automatic alert generation.  
- Drift detection triggered when transaction behavior shifts.  
- Adaptive model response after drift.  


---

##  Why Online Machine Learning?

Traditional batch ML models:
- Require retraining.
- Cannot adapt in real-time.
- Fail under concept drift.

This system:
- Learns incrementally.
- Detects distribution shifts.
- Adapts without retraining from scratch.

This design is closer to real-world fintech monitoring systems.

---

##  Model Behavior

- Mean anomaly score remains stable during normal periods.  
- Spikes during abnormal activity.  
- ADWIN detects statistical shifts and signals concept drift.  
- Model adapts post-drift.  

---

##  Limitations

- Uses simulated transaction data.  
- Unsupervised anomaly detection (no labeled fraud feedback loop).  
- No explainability module (feature contribution analysis).

---

##  Future Improvements

- Add supervised feedback mechanism.  
- Integrate SHAP-style explainability.  
- Deploy with real financial datasets.  
- Implement alert prioritization logic.  
- Add automated reporting export.  

---

##  Portfolio Impact

This project demonstrates:

- Online machine learning implementation.  
- Concept drift handling.  
- Real-time anomaly detection.  
- Production-style monitoring dashboard.  
- End-to-end ML system design.  

---

##  Author  
Syed Muhib Ali Jaffri
- 
Junior Machine Learning Engineer

Focus: Applied ML & Real-world Problems