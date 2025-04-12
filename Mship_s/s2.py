import asyncio
import websockets
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque  # ✅ For real-time data storage

# Load dataset
data_path = 'navalplantmaintenance.csv'
data = pd.read_csv(data_path, sep='\s+', header=None)
data.columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate', 'gg_rate',
                'sp_torque', 'pp_torque', 'hpt_temp', 'gt_c_i_temp', 'gt_c_o_temp',
                'hpt_pressure', 'gt_c_i_pressure', 'gt_c_o_pressure', 'gt_exhaust_pressure',
                'turbine_inj_control', 'fuel_flow', 'gt_c_decay', 'gt_t_decay']

X = data.iloc[:, :-2]
Y1 = data['gt_c_decay']  # Compressor decay
Y2 = data['gt_t_decay']  # Turbine decay

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def train_random_forest(X_train, Y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    return model

rfr_y1 = train_random_forest(X_scaled, Y1)
rfr_y2 = train_random_forest(X_scaled, Y2)

# Store last 100 sensor readings for visualization
sensor_history = deque(maxlen=100)

# Save logs to CSV
log_file = "sensor_logs.csv"

# Threshold conditions
threshold_conditions = {
    'gt_c_i_temp': (lambda x: x < 280 or x > 400, 280, 400),
    'gt_c_o_temp': (lambda x: x > 590, 590),
    'gt_c_i_pressure': (lambda x: x < 0.90, 0.90),
    'gt_c_o_pressure': (lambda x: x < 5.00, 5.00),
    'fuel_flow': (lambda x: x < 0.06 or x > 1.5, 0.06, 1.5),
    'turbine_inj_control': (lambda x: x < 4 or x > 7.5, 4, 7.5),
    'hpt_temp': (lambda x: x > 650, 650),
    'gt_shaft_torque': (lambda x: x < 280 or x > 300, 280, 300),
    'gg_rate': (lambda x: x < 5000 or x > 7200, 5000, 7200)
}

def estimate_time_before_failure(predicted_decay):
    hours = max(0, (1 - predicted_decay) * 100)  # Adjust factor as necessary
    return f"{int(hours * 60)} minutes"  # Convert hours to minutes

def detect_fault_and_identify_critical_sensor(live_data):
    feature_columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate',
                       'gg_rate', 'sp_torque', 'pp_torque', 'hpt_temp',
                       'gt_c_i_temp', 'gt_c_o_temp', 'hpt_pressure',
                       'gt_c_i_pressure', 'gt_c_o_pressure', 'gt_exhaust_pressure',
                       'turbine_inj_control', 'fuel_flow']

    live_data_df = pd.DataFrame([live_data], columns=feature_columns)
    live_data_scaled = scaler.transform(live_data_df)

    y1_pred = rfr_y1.predict(live_data_scaled)
    y2_pred = rfr_y2.predict(live_data_scaled)

    result = {
        "Predicted_Compressor_Decay": y1_pred[0],
        "Predicted_Turbine_Decay": y2_pred[0],
        "Time_Before_Failure": {
            "Compressor": estimate_time_before_failure(y1_pred[0]),
            "Turbine": estimate_time_before_failure(y2_pred[0])
        },
        "Compressor_Fault_Detected": "No Fault",
        "Turbine_Fault": "No Fault",
        "Warnings": [],
        "Suggestions": []
    }

    for sensor, (condition, *thresholds) in threshold_conditions.items():
        if sensor in live_data_df.columns:
            sensor_value = live_data_df[sensor].values[0]
            if condition(sensor_value):
                result["Warnings"].append(f"⚠️ {sensor}: {sensor_value} (Threshold: {', '.join(map(str, thresholds))})")
                result["Compressor_Fault_Detected"] = "⚠️ Fault Detected"
                result["Suggestions"].append(f"Adjust {sensor} within {', '.join(map(str, thresholds))}.")

    # Save to history for visualization
    sensor_history.append({**live_data, **result})

    # Append to CSV log
    pd.DataFrame([result]).to_csv(log_file, mode='a', header=not pd.io.common.file_exists(log_file), index=False)

    return result

async def handle_client(websocket, path):
    print(f"Client connected: {path}")
    try:
        async for message in websocket:
            live_data = json.loads(message)

            if message == "GET_HISTORY":
                # Send last 100 readings for visualization
                await websocket.send(json.dumps(list(sensor_history)))
            else:
                result = detect_fault_and_identify_critical_sensor(live_data)
                await websocket.send(json.dumps(result))
    except websockets.exceptions.ConnectionClosedOK:
        print("Client disconnected")

async def start_server():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("✅ WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(start_server())
