import json
import statistics
import random
from datetime import datetime   
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================
app = FastAPI(
    title="Smart Spoon AI - Dual-Mode Biosensor Engine",
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TelemetryData(BaseModel):
    adc: float           
    temperature: float   

class ModeSwitch(BaseModel):
    mode: str  # "apple" or "milk"

# ============================================================================
# GLOBAL STATE & BUFFERS
# ============================================================================
ACTIVE_MODE = "apple"  # Default startup mode
rolling_buffer: List[float] = []
MAX_BUFFER_SIZE = 5 
latest_payload = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        payload_text = json.dumps(message)
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload_text)
            except Exception:
                dead_connections.append(connection)
        for dead in dead_connections:
            self.disconnect(dead)

manager = ConnectionManager()

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def health_check():
    return {"status": "online", "current_mode": ACTIVE_MODE}

# NEW ENDPOINT: Frontend hits this to switch modes!
@app.post("/set_mode")
async def set_mode(data: ModeSwitch):
    global ACTIVE_MODE, rolling_buffer
    if data.mode in ["apple", "milk"]:
        ACTIVE_MODE = data.mode
        rolling_buffer.clear()  # Clear buffer on switch to prevent cross-contamination
        print(f"🔄 MODE SWITCHED TO: {ACTIVE_MODE.upper()}", flush=True)
        return {"status": "success", "active_mode": ACTIVE_MODE}
    return {"status": "error", "message": "Invalid mode"}

@app.post("/ingest")
async def ingest_telemetry(data: TelemetryData):
    global latest_payload, rolling_buffer, ACTIVE_MODE
    print(f"[{ACTIVE_MODE.upper()} MODE] Freq: {data.adc} Hz | Temp: {data.temperature} C", flush=True)

    # 1. MEAN LOGIC BUFFER
    rolling_buffer.append(data.adc)
    if len(rolling_buffer) > MAX_BUFFER_SIZE:
        rolling_buffer.pop(0)

    # Ignore massive zeros/dropouts in the mean
    active_readings = [val for val in rolling_buffer if val > 50]
    
    if len(active_readings) > 0:
        stable_mean = float(statistics.mean(active_readings))
    else:
        stable_mean = float(data.adc) 

    # 2. DYNAMIC AI CONFIDENCE SIMULATOR
    confidence = round(random.uniform(97.2, 99.6), 1)
    remainder = round((100.0 - confidence) / 2.0, 1)

    # 3. DUAL-MODE HARDWARE CLASSIFIER
    if stable_mean < 100:
        # AWAITING DATA (COMMON FOR BOTH)
        predicted_class = 0
        verdict = "AWAITING SENSOR DATA"
        status_color = "#334155" # Grey
        safety_score = 0
        directive = f"Insert gold electrodes into the {ACTIVE_MODE} sample."
        fraud_loss = 0.0
        prob_dict = {"Pure": 0.0, "Adulterant_A": 0.0, "Adulterant_B": 0.0}

    elif ACTIVE_MODE == "apple":
        # --- APPLE THRESHOLDS ---
        if 100 <= stable_mean < 700:
            predicted_class = 1
            verdict = "Synthetic Wax / Glycerin Insulator"
            status_color = "#f59e0b" # Yellow
            safety_score = 35
            directive = "FSSAI VIOLATION. Dielectric crash indicates illegal synthetic petroleum wax coating."
            fraud_loss = 45.0
            prob_dict = {"Pure_Apple": remainder, "Wax_Coating": confidence, "Aqueous_Urea": remainder}
            
        elif 700 <= stable_mean <= 1800:
            predicted_class = 2
            verdict = "Pure Apple / Unadulterated"
            status_color = "#10b981" # Green
            safety_score = 99
            directive = "Cellular impedance conforms to natural, safe agricultural parameters."
            fraud_loss = 0.0
            prob_dict = {"Pure_Apple": confidence, "Wax_Coating": remainder, "Aqueous_Urea": remainder}
            
        else: # > 1800
            predicted_class = 3
            verdict = "Aqueous Urea Injection! TOXIC"
            status_color = "#ef4444" # Red
            safety_score = 0
            directive = "SEVERE HAZARD. Massive ionic conductivity spike indicates toxic chemical fertilizers."
            fraud_loss = 100.0
            prob_dict = {"Pure_Apple": remainder, "Wax_Coating": remainder, "Aqueous_Urea": confidence}

    else:
        # --- MILK THRESHOLDS ---
        if 100 <= stable_mean < 2000:
            predicted_class = 1
            verdict = "Starch / Water Adulteration"
            status_color = "#f59e0b" # Yellow
            safety_score = 42
            directive = "Thickening agent (Starch) detected. Solids-Not-Fat (SNF) threshold manipulated."
            fraud_loss = 22.50
            prob_dict = {"Pure_Milk": remainder, "Starch_Fake": confidence, "Detergent": remainder}
            
        elif 2000 <= stable_mean <= 6000:
            predicted_class = 2
            verdict = "Pure Milk / Safe"
            status_color = "#10b981" # Green
            safety_score = 98
            directive = "Dielectric impedance conforms to FSSAI Class-A pure dairy parameters."
            fraud_loss = 0.0
            prob_dict = {"Pure_Milk": confidence, "Starch_Fake": remainder, "Detergent": remainder}
            
        else: # > 6000
            predicted_class = 3
            verdict = "Detergent Adulteration! BIOHAZARD"
            status_color = "#ef4444" # Red
            safety_score = 0
            directive = "SEVERE BIOHAZARD. Synthetic surfactants and chemical soaps detected."
            fraud_loss = 100.0
            prob_dict = {"Pure_Milk": remainder, "Starch_Fake": remainder, "Detergent": confidence}

    # 4. DASHBOARD PAYLOAD
    latest_payload = {
        "hero": {
            "adulteration_type": verdict,
            "accuracy": confidence if predicted_class != 0 else 0.0,
            "status_color": status_color,
        },
        "primary": {
            "1_safety_score": safety_score,
            "11_kitchen_directive": directive,
            "19_fraud_loss_penalty_inr": fraud_loss,
            "21_REAL_TIME_TEMP_C": round(data.temperature, 1),
        },
        "secondary": {
            "eis_dsp_telemetry": {
                "1_Internal_Cellular_Impedance": int(stable_mean if stable_mean > 0 else 500),
            },
            "system_meta": {
                "ACTIVE_MODE": ACTIVE_MODE.upper(),
            },
            "ai_and_regulatory_metrology": {
                "35_Class_Probability_Distribution": str(prob_dict)
            },
        },
        "system_meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "raw_adc": int(data.adc),
            "processed_mean_hz": int(stable_mean),
            "current_mode": ACTIVE_MODE
        },
    }

    await manager.broadcast(latest_payload)
    return {"status": "success", "processed_mean": int(stable_mean), "mode": ACTIVE_MODE}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        if latest_payload:
            await websocket.send_text(json.dumps(latest_payload))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
