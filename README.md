
# ⚙️ Marine Gas Turbine Maintenance System

A real-time monitoring and predictive maintenance system for marine gas turbine plants using **WebSocket streaming**, **machine learning**, and an interactive **Streamlit dashboard**.

---

## 🚀 Project Overview

### 🔌 WebSocket Data Streaming
- Real-time data is streamed from marine gas turbine plant sensors via a WebSocket server.
- Data includes live readings and predicted decay values from the ML model.
- Updates every **2 seconds**.

### 🧠 Machine Learning (Random Forest)
- A **RandomForestRegressor** model predicts:
  - Compressor decay (`gt_c_decay`)
  - Turbine decay (`gt_t_decay`)
- Model is trained on historical sensor data and uses **StandardScaler** for feature normalization.

### 📊 Streamlit Dashboard
- Live graphs for temperature, pressure, decay rates.
- Instant alerts for threshold violations.
- Maintenance history management and actionable suggestions.

---

## 📸 Dashboard Preview

### 🔧 Real-Time Monitoring Dashboard
![Live Monitoring](![image](https://github.com/user-attachments/assets/9b2174b6-a363-496d-bd0b-c16e2df2c865))

(![image](https://github.com/user-attachments/assets/c831720d-8154-43de-ab23-167a43f69ddf))

### 📈 Decay Estimation & Warnings
![Decay Predictions](![image](https://github.com/user-attachments/assets/05e4f82f-1eb0-461b-b2d4-08c68b817a01))
(![image](https://github.com/user-attachments/assets/9b657d27-72c2-4e86-93f6-b48b5ab66478))

### 🛠 Maintenance Logs
![Maintenance Logs](![image](https://github.com/user-attachments/assets/0c31d976-54c1-42d3-8e37-80f39cab0145))



---

## ⚙️ Setup Instructions

### 📁 1. Clone the Repository

```bash
git clone https://github.com/yourusername/marine_maintenance.git
cd marine_maintenance
```

### 🐍 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts ctivate
pip install -r requirements.txt
```

> Example `requirements.txt`:
```
streamlit
pandas
numpy
scikit-learn
websockets
matplotlib
plotly
```

---

### 🔁 3. Run the WebSocket Server

```bash
python backend/websocket_server.py
```

This simulates real-time sensor streaming and prediction. It should output:
```
WebSocket server started on ws://localhost:8000
```

---

### 📊 4. Launch the Streamlit Dashboard

In a new terminal, run:

```bash
streamlit run dashboard/app.py
```

> The dashboard will open in your browser at:  
> `http://localhost:8501`

---

## 🧠 Machine Learning Details

### 📈 Estimation Formulas

#### Compressor Decay (y1 = gt_c_decay)
```
y1 = (1/N) * Σ Ti^(c)(X)
```

#### Turbine Decay (y2 = gt_t_decay)
```
y2 = (1/N) * Σ Ti^(t)(X)
```

- `Ti^(c)(X)` and `Ti^(t)(X)` are outputs from each decision tree.
- `X` are standardized sensor inputs.
- `N` is the number of trees in the Random Forest.

### 🔃 Standardization Formula

```
X_scaled = (X - μ) / σ
```

- Ensures all features have zero mean and unit variance.

### 🔍 Fault Detection Rule

```
IF y1 > threshold_c OR y2 > threshold_t:
    Trigger Fault Alert
```

---

## 📁 Local Files Used

| File Name                  | Purpose                                  |
|---------------------------|------------------------------------------|
| `navalplantmaintenance.csv` | ML training dataset                     |
| `fault_condition_data.csv`  | Predefined fault conditions              |
| `sensor_logs.csv`           | Logs of real-time sensor and prediction |
| `maintenance_records.csv`   | Maintenance history                     |

---

## 💡 Novel Features

- ✅ Real-time prediction with **WebSockets**.
- ✅ Random Forest-based fault prediction.
- ✅ Streamlit-powered visual interface.
- ✅ Complete maintenance tracking and logging.

---

## 🖥️ Technologies Used

- `Python`
- `Streamlit`
- `scikit-learn`
- `websockets`
- `pandas`, `numpy`
- `matplotlib`, `plotly`

---

## 🛠 Future Enhancements

- Add user authentication for maintenance logs.
- Historical analytics and fault trend predictions.
- Integration with external alert systems (email/SMS).

---

## 📬 Contact

For questions or collaboration:  
📧 **kavya.student@saveetha.ac.in** 

---
