import asyncio
import websockets
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from collections import deque
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import plotly.figure_factory as ff

# Streamlit page configuration
st.set_page_config(page_title="Live Fault Detection & Maintenance", layout="wide")

# Page Title
st.markdown("# 🚀 Live Fault Detection & Maintenance Dashboard")
st.write("📡 Streaming real-time responses from WebSocket server every **5 seconds**...")

# Sidebar Metrics
st.sidebar.header("📊 Live Data Metrics")
processed_rows = st.sidebar.empty()  # Dynamic update of data sent count
fault_status = st.sidebar.empty()  # Fault detection status

# Sidebar Maintenance Record Section
st.sidebar.markdown("---")
st.sidebar.markdown("## 🛠 Maintenance Records")

# Maintenance Record File
maintenance_file = "maintenance_records.csv"

# Check if maintenance file exists, if not, create it
try:
    maintenance_data = pd.read_csv(maintenance_file)
except FileNotFoundError:
    maintenance_data = pd.DataFrame(columns=["Date", "Description", "Action Taken"])
    maintenance_data.to_csv(maintenance_file, index=False)

# Function to save maintenance record
def add_maintenance_record(description, action):
    """Adds a maintenance record to the CSV file."""
    new_record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Description": description,
        "Action Taken": action
    }])

    # Append new record safely
    maintenance_data = pd.read_csv(maintenance_file)
    maintenance_data = pd.concat([maintenance_data, new_record], ignore_index=True)
    maintenance_data.to_csv(maintenance_file, index=False)

    st.success("✅ Maintenance record added successfully!")
    st.rerun()  # Refresh page and clear form

# Form to add new maintenance records in sidebar
with st.sidebar.form("add_maintenance"):
    st.write("### 📝 Add New Maintenance Record")
    description = st.text_input("🔹 Description (e.g., 'Replaced turbine filter')")
    action_taken = st.text_input("🔹 Action Taken (e.g., 'Installed new filter, tested performance')")

    submitted = st.form_submit_button("✅ Add Maintenance Record")
    if submitted:
        if description and action_taken:
            add_maintenance_record(description, action_taken)
        else:
            st.sidebar.warning("⚠️ Please fill in all fields before submitting.")

# Display Past Maintenance Records in sidebar
st.sidebar.markdown("### 📋 View Maintenance History")
if maintenance_data.empty:
    st.sidebar.info("No maintenance records found. Add a new maintenance entry above.")
else:
    st.sidebar.dataframe(maintenance_data, use_container_width=True)

# Placeholder for dynamic content
placeholder = st.empty()

# Load dataset
data_path = "fault_condition_data.csv"
data = pd.read_csv(data_path, sep='\s+', header=None)

# Define column names
data.columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate', 'gg_rate',
                'sp_torque', 'pp_torque', 'hpt_temp', 'gt_c_i_temp', 'gt_c_o_temp',
                'hpt_pressure', 'gt_c_i_pressure', 'gt_c_o_pressure', 'gt_exhaust_pressure',
                'turbine_inj_control', 'fuel_flow']

# WebSocket connection URI
uri = "ws://localhost:8765"

# **🔹 REAL-TIME DATA STORAGE**
# Use deque for live updating (store last 100 records)
max_points = 100
data_buffer = {col: deque(maxlen=max_points) for col in data.columns}
time_buffer = deque(maxlen=max_points)

# Function to connect and stream data
async def send_data():
    try:
        async with websockets.connect(uri) as websocket:
            for index, row in data.iterrows():
                # Convert row to dictionary
                live_data = row.to_dict()

                # Send data to the WebSocket server
                await websocket.send(json.dumps(live_data))

                # Receive and process the response
                try:
                    response = await websocket.recv()
                    result = json.loads(response)

                    # Determine machine risk status
                    warnings = result.get("Warnings", [])
                    suggestions = result.get("Suggestions", [])
                    
                    if warnings or suggestions:
                        machine_status = "⚠️ **Abnormal (At Risk of Failure)**"
                        status_color = "red"
                    else:
                        machine_status = "🟢 **Normal (Operating Safely)**"
                        status_color = "green"

                    # Update Sidebar Metrics
                    processed_rows.metric("📊 Total Data Points Sent", index + 1)
                    fault_status.markdown(f"## 🚨 Machine Status: <span style='color:{status_color}; font-size:20px;'>{machine_status}</span>", unsafe_allow_html=True)

                    # Update real-time buffer (store last 100 records)
                    time_buffer.append(datetime.now().strftime("%H:%M:%S"))
                    for col in live_data:
                        data_buffer[col].append(live_data[col])

                    # Update UI with the latest response
                    with placeholder.container():
                        st.subheader("🔍 Latest Response")
                        st.json(result)

                        st.markdown("### 📊 Data Overview")
                        st.dataframe(pd.DataFrame([live_data]), use_container_width=True)

                        # **🔹 REAL-TIME PLOTTING**
                        st.markdown("### 📈 Real-Time Parameter Trends")
                        col1, col2 = st.columns(2)

                        with col1:
                            fig1 = px.line(
                                x=list(time_buffer),
                                y=list(data_buffer['gt_c_i_temp']),
                                labels={'x': 'Time', 'y': 'GT Compressor Inlet Temp'},
                                title="GT Compressor Inlet Temperature"
                            )
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            fig2 = px.line(
                                x=list(time_buffer),
                                y=list(data_buffer['hpt_temp']),
                                labels={'x': 'Time', 'y': 'HPT Temperature'},
                                title="High-Pressure Turbine Temperature"
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        # **🔹 ALERTS & WARNINGS**
                        st.markdown("### 📢 Alerts & Suggestions")
                        for warning in warnings:
                            st.warning(warning)
                        for suggestion in suggestions:
                            st.info(suggestion)

                except json.JSONDecodeError as e:
                    st.error(f"❌ Error decoding JSON: {e}")

                # Delay before sending next data point (Now 2 seconds)
                await asyncio.sleep(5)
    except websockets.exceptions.ConnectionClosed as e:
        st.error(f"❌ WebSocket connection closed: {e}")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")

# Start the WebSocket connection
st.write("⏳ Connecting to WebSocket server...")

# Run the WebSocket function asynchronously
asyncio.run(send_data())
