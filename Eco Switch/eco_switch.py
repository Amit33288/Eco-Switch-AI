import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import os
from datetime import datetime
from pathlib import Path
from io import BytesIO

# ==========================================
# 1. APP CONFIGURATION & USERS
# ==========================================

st.set_page_config(page_title="Eco-Switch Pro", layout="wide", initial_sidebar_state="expanded")

# --- 🟢 USER DATABASE (SIMPLE TEXT PASSWORDS) ---
ADMIN_USERS = {
    "admin": {
        "password": "admin123",
        "role": "Super Admin"
    },
    "editor": {
        "password": "edit456",
        "role": "Editor"
    },
    "analyst": {
        "password": "view789",
        "role": "Viewer"
    },
    "amit.k": {
        "password": "amit328",
        "role": "Developer🧑‍💻"
    }
}

# ==========================================
# 2. STYLING
# ==========================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { background-color: #0E1117; font-family: 'Poppins', sans-serif; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .kpi-container { display: flex; gap: 20px; margin-bottom: 30px; }
    .kpi-card { background: rgba(30, 34, 45, 0.6); backdrop-filter: blur(10px); border-radius: 16px; padding: 20px; flex: 1; border: 1px solid rgba(255, 255, 255, 0.08); }
    .kpi-label { color: #8b949e; font-size: 13px; font-weight: 500; text-transform: uppercase; }
    .kpi-value { background: linear-gradient(90deg, #00E676, #69F0AE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-size: 32px; margin-top: 5px; }
    .stButton>button { background: linear-gradient(90deg, #00C853, #00E676); border: none; border-radius: 12px; color: #000; font-weight: 700; padding: 0.6rem 1.2rem; }
    .success-box { background: rgba(0, 200, 83, 0.1); border-left: 4px solid #00E676; padding: 20px; border-radius: 8px; margin-top: 20px; }
    .warning-box { background: rgba(255, 171, 0, 0.1); border-left: 4px solid #FFAB00; padding: 20px; border-radius: 8px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True
)

# ==========================================
# 3. DATA & HELPERS
# ==========================================

COUNTRY_DATA = {
    "Algeria": {"currency": "DZD", "elec_price": 5.5}, "Argentina": {"currency": "$", "elec_price": 55.0},
    "Australia": {"currency": "A$", "elec_price": 0.35}, "Austria": {"currency": "€", "elec_price": 0.38},
    "Belgium": {"currency": "€", "elec_price": 0.36}, "Brazil": {"currency": "R$", "elec_price": 0.85},
    "Canada": {"currency": "C$", "elec_price": 0.16}, "Chile": {"currency": "$", "elec_price": 140.0},
    "China": {"currency": "¥", "elec_price": 0.60}, "Colombia": {"currency": "$", "elec_price": 750.0},
    "Czech Republic": {"currency": "Kč", "elec_price": 8.5}, "Denmark": {"currency": "kr", "elec_price": 2.8},
    "Egypt": {"currency": "E£", "elec_price": 1.4}, "Finland": {"currency": "€", "elec_price": 0.25},
    "France": {"currency": "€", "elec_price": 0.23}, "Germany": {"currency": "€", "elec_price": 0.40},
    "Greece": {"currency": "€", "elec_price": 0.22}, "Hungary": {"currency": "Ft", "elec_price": 36.0},
    "India": {"currency": "₹", "elec_price": 8.0}, "Indonesia": {"currency": "Rp", "elec_price": 1450.0},
    "Ireland": {"currency": "€", "elec_price": 0.38}, "Israel": {"currency": "₪", "elec_price": 0.60},
    "Italy": {"currency": "€", "elec_price": 0.35}, "Japan": {"currency": "¥", "elec_price": 31.0},
    "Malaysia": {"currency": "RM", "elec_price": 0.50}, "Mexico": {"currency": "$", "elec_price": 1.9},
    "Netherlands": {"currency": "€", "elec_price": 0.35}, "New Zealand": {"currency": "NZ$", "elec_price": 0.30},
    "Norway": {"currency": "kr", "elec_price": 1.5}, "Pakistan": {"currency": "₨", "elec_price": 45.0},
    "Philippines": {"currency": "₱", "elec_price": 11.0}, "Poland": {"currency": "zł", "elec_price": 0.95},
    "Portugal": {"currency": "€", "elec_price": 0.24}, "Russia": {"currency": "₽", "elec_price": 5.5},
    "Saudi Arabia": {"currency": "﷼", "elec_price": 0.18}, "Singapore": {"currency": "S$", "elec_price": 0.30},
    "South Africa": {"currency": "R", "elec_price": 2.80}, "South Korea": {"currency": "₩", "elec_price": 140.0},
    "Spain": {"currency": "€", "elec_price": 0.24}, "Sweden": {"currency": "kr", "elec_price": 2.5},
    "Switzerland": {"currency": "CHF", "elec_price": 0.32}, "Taiwan": {"currency": "NT$", "elec_price": 3.0},
    "Thailand": {"currency": "฿", "elec_price": 4.7}, "Turkey": {"currency": "₺", "elec_price": 2.6},
    "UAE": {"currency": "AED", "elec_price": 0.30}, "UK": {"currency": "£", "elec_price": 0.34},
    "USA": {"currency": "$", "elec_price": 0.16}, "United Kingdom": {"currency": "£", "elec_price": 0.34},
    "United States": {"currency": "$", "elec_price": 0.16}, "Vietnam": {"currency": "₫", "elec_price": 2000.0}
}

DEFAULT_CURRENCY = "$"
DEFAULT_ELEC_PRICE = 0.14
SEGMENT_MILEAGE = {
    "Small Hatchback (e.g., Alto, Swift)": 18,
    "Sedan (e.g., City, Civic)": 14,
    "Compact SUV (e.g., Creta, Nexon)": 12,
    "Large SUV / Truck (e.g., Fortuner, F-150)": 9,
    "Sports Car": 7
}
EV_SEGMENT = {"Budget City EV":12, "Standard Sedan EV":15, "Performance SUV EV":22}

# File Paths
FILE_MILEAGE = "Fuel Car Mileage.csv"
FILE_PREDS = "fuel_price_predictions_2025_2030.csv"
FILE_HIST = "clean_fuel_prices.csv"
FILE_MASTER = "Master_Fuel_Prices_2015_2024.csv"
FILE_EV = "clean_ev_specs.csv"
FILE_GEF = "clean_gef.csv"

def get_country_info(country_name):
    if country_name in COUNTRY_DATA: return COUNTRY_DATA[country_name]
    for k, v in COUNTRY_DATA.items():
        if k.lower() in country_name.lower(): return v
    return {"currency": DEFAULT_CURRENCY, "elec_price": DEFAULT_ELEC_PRICE}

@st.cache_data(show_spinner=False)
def load_data(path):
    if not Path(path).exists(): return pd.DataFrame()
    try: df = pd.read_csv(path, low_memory=False)
    except: df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def save_data(df, path):
    try:
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving: {e}")
        return False

def parse_float(x):
    try: return float(x)
    except:
        m = re.search(r"\d+(\.\d+)?", str(x))
        return float(m.group(0)) if m else None

def prepare_mileage(raw):
    if raw.empty: return pd.DataFrame(columns=["Make","Model","Fuel","kmpl"])
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    make_col = next((c for c in df.columns if re.search("identification\\.make|\\bmake\\b|\\bbrand\\b", c, re.I)), None)
    class_col = next((c for c in df.columns if re.search("identification\\.classification|classification|model\\b", c, re.I)), None)
    year_col = next((c for c in df.columns if re.search("\\byear\\b|identification\\.year", c, re.I)), None)
    fuel_col = next((c for c in df.columns if re.search("fuel.*type|fuel type", c, re.I)), None)
    city_col = next((c for c in df.columns if re.search("city mpg|city_mpg", c, re.I)), None)
    hwy_col = next((c for c in df.columns if re.search("highway mpg|highway_mpg", c, re.I)), None)

    if make_col: df["Make"] = df[make_col].astype(str).str.strip()
    else: df["Make"] = "Unknown"

    if class_col:
        if year_col and year_col in df.columns:
            df["Model"] = df[class_col].astype(str).str.strip() + " (" + df[year_col].astype(str).str.strip() + ")"
        else:
            df["Model"] = df[class_col].astype(str).str.strip()
    elif year_col:
        df["Model"] = df["Make"].astype(str).str.strip() + " " + df[year_col].astype(str).str.strip()
    else:
        df["Model"] = df["Make"].astype(str).str.strip() + " " + df.index.astype(str)

    if fuel_col and fuel_col in df.columns:
        df["Fuel"] = df[fuel_col].astype(str).str.strip()
    else:
        merged = df.astype(str).agg(" ".join, axis=1).str.lower()
        df["Fuel"] = merged.apply(lambda t: "Diesel" if "diesel" in t else ("Petrol" if ("petrol" in t or "gasoline" in t or " gas " in t) else ""))

    conv = 0.425143707
    city_vals = df[city_col].apply(parse_float) if (city_col and city_col in df.columns) else pd.Series([None]*len(df))
    hwy_vals = df[hwy_col].apply(parse_float) if (hwy_col and hwy_col in df.columns) else pd.Series([None]*len(df))
    mpg_avg = pd.concat([city_vals, hwy_vals], axis=1).mean(axis=1)
    df["kmpl"] = mpg_avg.apply(lambda v: v*conv if v and not pd.isna(v) else None)

    out = df[["Make","Model","Fuel","kmpl"]].drop_duplicates().reset_index(drop=True)
    out["Make"] = out["Make"].fillna("").astype(str)
    out["Model"] = out["Model"].fillna("").astype(str)
    out["Fuel"] = out["Fuel"].fillna("").astype(str)
    return out

# ==========================================
# 4. INITIALIZATION & STATE
# ==========================================

# Load Data
if "mileage_db" not in st.session_state:
    raw = load_data(FILE_MILEAGE)
    st.session_state.mileage_db = prepare_mileage(raw)

for key, file in zip(
    ["preds_df", "hist_prices_df", "master_prices_df", "ev_df", "gef_df"],
    [FILE_PREDS, FILE_HIST, FILE_MASTER, FILE_EV, FILE_GEF]
):
    if key not in st.session_state:
        st.session_state[key] = load_data(file)

# Session State
if "params" not in st.session_state:
    st.session_state.params = {
        "country": "India", "fuel": "Petrol", "car_mode": "By Segment (Easy)",
        "segment": list(SEGMENT_MILEAGE.keys())[0], "brand": "Other", "model": "My Car",
        "mileage": 15.0, "ev_mode": "Generic EV", "ev_choice": list(EV_SEGMENT.keys())[0],
        "daily_km": 40, "use_avg_elec": True, "elec_price": 8.0, 
        "currency_symbol": "₹", "manual_fuel_price": 0.0
    }
if "admin_user" not in st.session_state: st.session_state.admin_user = None
if "admin_role" not in st.session_state: st.session_state.admin_role = None
if "view_mode" not in st.session_state: st.session_state.view_mode = "User"
if "audit_log" not in st.session_state: st.session_state.audit_log = []
if "last_country" not in st.session_state: st.session_state.last_country = "India"
if "last_fuel" not in st.session_state: st.session_state.last_fuel = "Petrol"

# Aliases
preds_df = st.session_state.preds_df
hist_prices_df = st.session_state.hist_prices_df
master_prices_df = st.session_state.master_prices_df
ev_df = st.session_state.ev_df
gef_df = st.session_state.gef_df

# ==========================================
# 5. ADMIN UI
# ==========================================

def render_admin_panel():
    user = st.session_state.admin_user
    role = st.session_state.admin_role
    
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🛠️ Advanced Admin Panel")
        st.caption(f"Logged in as: **{user}**")
    with c2:
        if role == "Super Admin": st.markdown("#### 🔴 Super Admin")
        elif role == "Editor": st.markdown("#### 🟠 Editor")
        elif role == "Developer🧑‍💻": st.markdown("#### 🟣 Developer🧑‍💻")
        else: st.markdown("#### 🟢 Viewer")
    
    can_edit = role in ["Super Admin", "Editor", "Developer🧑‍💻"]
    if not can_edit: st.warning("🔒 View-Only mode.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚡ EV Specs", "⛽ Fuel Prices", "🌍 GEF Data", "🔮 Predictions", "📜 Logs"])
    
    with tab1:
        st.subheader("Manage EV Specifications")
        edited_ev = st.data_editor(st.session_state.ev_df, num_rows="dynamic", key="editor_ev", disabled=not can_edit)
        if can_edit and st.button("💾 Save EV Data"):
            if save_data(edited_ev, FILE_EV):
                st.session_state.ev_df = edited_ev
                st.cache_data.clear()
                st.session_state.audit_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {user} updated EV Data")
                st.success("Saved!")
    
    with tab2:
        st.subheader("Manage Historical Fuel Prices")
        edited_master = st.data_editor(st.session_state.master_prices_df, num_rows="dynamic", key="editor_master", disabled=not can_edit)
        if can_edit and st.button("💾 Save Fuel Prices"):
            if save_data(edited_master, FILE_MASTER):
                st.session_state.master_prices_df = edited_master
                st.cache_data.clear()
                st.session_state.audit_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {user} updated Fuel Prices")
                st.success("Saved!")

    with tab3:
        st.subheader("Manage Grid Emission Factors")
        edited_gef = st.data_editor(st.session_state.gef_df, num_rows="dynamic", key="editor_gef", disabled=not can_edit)
        if can_edit and st.button("💾 Save GEF Data"):
            if save_data(edited_gef, FILE_GEF):
                st.session_state.gef_df = edited_gef
                st.cache_data.clear()
                st.session_state.audit_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {user} updated GEF Data")
                st.success("Saved!")

    with tab4:
        st.subheader("Manage Predictions")
        edited_preds = st.data_editor(st.session_state.preds_df, num_rows="dynamic", key="editor_preds", disabled=not can_edit)
        if can_edit and st.button("💾 Save Predictions"):
            if save_data(edited_preds, FILE_PREDS):
                st.session_state.preds_df = edited_preds
                st.cache_data.clear()
                st.session_state.audit_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {user} updated Predictions")
                st.success("Saved!")

    with tab5:
        st.subheader("Session Activity Log")
        if st.session_state.audit_log:
            for log in reversed(st.session_state.audit_log): st.text(f"• {log}")
        else: st.text("No activity.")

# ==========================================
# 6. SIMULATION LOGIC
# ==========================================

def compute_simulation(params):
    daily_km = params["daily_km"]
    current_mileage = float(params["mileage"])
    ev_eff = float(params.get("ev_eff", 15.0))
    elec_price = float(params["elec_price"])
    curr = params.get("currency_symbol", "$")
    base_price = float(params["manual_fuel_price"])
    
    years = list(range(2025, 2031))
    prices = [base_price * (1.04 ** (y - 2025)) for y in years] # 4% Inflation

    # Get GEF from State Data
    gef = 0.7
    current_gef_df = st.session_state.gef_df
    if not current_gef_df.empty and 'country' in current_gef_df.columns:
        row = current_gef_df[current_gef_df['country'].str.lower() == params["country"].lower()]
        if not row.empty: gef = float(row.iloc[0]['gef'])

    co2_factor = 2.3 if params["fuel"] == "Petrol" else 2.7
    records = []
    cumulative = 0.0

    for yr, price in zip(years, prices):
        liters = (daily_km * 365) / current_mileage
        cost_fuel = liters * price
        co2_fuel = liters * co2_factor

        kwh_needed = (daily_km * 365 / 100.0) * ev_eff
        cost_ev = kwh_needed * elec_price
        co2_ev = kwh_needed * gef

        saving = cost_fuel - cost_ev
        cumulative += saving
        
        records.append({
            "Year": int(yr), "Currency": curr, "Fuel Price": round(price,2),
            "Cost Fuel": round(cost_fuel,2), "Cost EV": round(cost_ev,2),
            "Savings": round(saving,2), "Cumulative": round(cumulative,2),
            "CO2 Avoided (kg)": round(co2_fuel - co2_ev,2)
        })

    return pd.DataFrame(records), gef

# ==========================================
# 7. MAIN APP STRUCTURE
# ==========================================

# --- Sidebar Login ---
with st.sidebar:
    st.markdown("---")
    if st.session_state.admin_user is None:
        with st.expander("🔐 Admin Access", expanded=False):
            u_input = st.text_input("Username")
            p_input = st.text_input("Password", type="password")
            if st.button("Login"):
                # --- SIMPLE PASSWORD CHECK ---
                if u_input in ADMIN_USERS and p_input == ADMIN_USERS[u_input]["password"]:
                    st.session_state.admin_user = u_input
                    st.session_state.admin_role = ADMIN_USERS[u_input]["role"]
                    st.session_state.view_mode = "Admin Panel"
                    st.session_state.audit_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {u_input} logged in")
                    st.rerun()
                else: st.error("Invalid credentials")
    else:
        st.success(f"Hi, {st.session_state.admin_user}!")
        mode = st.radio("Mode", ["User Simulation", "Admin Panel"])
        if mode != st.session_state.view_mode:
            st.session_state.view_mode = mode
            st.rerun()
        if st.button("Logout"):
            st.session_state.audit_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {st.session_state.admin_user} logged out")
            st.session_state.admin_user = None
            st.session_state.view_mode = "User"
            st.rerun()
    st.markdown("---")

# --- Render View ---
if st.session_state.view_mode == "Admin Panel" and st.session_state.admin_user:
    render_admin_panel()
else:
    # --- USER SIMULATION UI ---
    p = st.session_state.params
    with st.sidebar:
        st.title("🚘 Simulation Parameters")
        
        # 1. Location
        st.subheader("1. Location")
        
        # Merge countries logic
        pred_countries = set(preds_df["country"].dropna().unique()) if not preds_df.empty else set()
        hist_countries = set(hist_prices_df["country"].dropna().unique()) if not hist_prices_df.empty else set()
        master_countries = set(master_prices_df["Country"].dropna().unique()) if not master_prices_df.empty else set()
        all_countries = sorted(list(pred_countries.union(hist_countries).union(master_countries)))
        
        if not all_countries: all_countries = ["India", "USA"]
        
        sel_country = st.selectbox("Select Country", all_countries, index=all_countries.index(p["country"]) if p["country"] in all_countries else 0)
        
        if sel_country != st.session_state.last_country:
            info = get_country_info(sel_country)
            p["country"] = sel_country
            p["elec_price"] = info["elec_price"]
            p["currency_symbol"] = info["currency"]
            p["manual_fuel_price"] = 0.0 
            st.session_state.last_country = sel_country
            st.rerun()
            
        st.markdown("---")
        
        # 2. Car Details
        st.subheader("2. Your Car")
        sel_fuel = st.radio("Fuel Type", ["Petrol","Diesel"], index=0 if p["fuel"]=="Petrol" else 1, horizontal=True)
        if sel_fuel != st.session_state.last_fuel:
            p["fuel"] = sel_fuel
            p["manual_fuel_price"] = 0.0
            st.session_state.last_fuel = sel_fuel
            st.rerun()
        p["fuel"] = sel_fuel

        # Price Detection
        det_price = None
        if p["manual_fuel_price"] == 0.0:
            # 1. Master
            try:
                m_row = master_prices_df[master_prices_df["Country"].str.lower() == p["country"].lower()]
                if not m_row.empty:
                    match = m_row[m_row["Fuel Type"].str.contains(p["fuel"], case=False, na=False)]
                    if not match.empty:
                        for y in ["2024","2023", "2022"]: 
                            val = match.iloc[0].get(y)
                            if pd.notna(val): 
                                det_price = float(val)
                                break
            except: pass
            
            # 2. Prediction
            if not det_price and not preds_df.empty:
                try:
                    cp = preds_df[preds_df['country'].str.lower() == p["country"].lower()]
                    fp = cp[cp['fuel'].str.lower().str.contains(p["fuel"].lower(), na=False)].sort_values('year')
                    if not fp.empty: det_price = fp.iloc[0]['predicted_price']
                except: pass

            # 3. Hist
            if not det_price and not hist_prices_df.empty:
                try:
                    h_rows = hist_prices_df[hist_prices_df['country'].str.lower() == p["country"].lower()]
                    col = "petrol_price" if "petrol" in p["fuel"].lower() else "diesel_price"
                    if not h_rows.empty and col in h_rows.columns:
                        det_price = float(h_rows.sort_values('year', ascending=False).iloc[0][col])
                except: pass
            
            p["manual_fuel_price"] = det_price if det_price else 100.0

        c1, c2 = st.columns([1, 2])
        with c1: st.markdown(f"**{p['currency_symbol']}**")
        with c2: p["manual_fuel_price"] = st.number_input("Fuel Price", value=float(p["manual_fuel_price"]), format="%.2f")

        p["car_mode"] = st.radio("Car Selection", ["By Segment (Easy)", "By Exact Model (Advanced)"])
        if p["car_mode"] == "By Segment (Easy)":
            p["segment"] = st.selectbox("Segment", list(SEGMENT_MILEAGE.keys()))
            p["mileage"] = SEGMENT_MILEAGE[p["segment"]]
        else:
            # Custom Model Logic
            filtered = st.session_state.mileage_db.copy()
            if "Fuel" in filtered.columns:
                filtered = filtered[filtered["Fuel"].str.lower().str.contains(p["fuel"].lower(), na=False)]
            brands = ["Other"] + sorted(filtered["Make"].dropna().unique().tolist())
            p["brand"] = st.selectbox("Brand", brands)
            
            models = ["My Car"]
            if p["brand"] != "Other":
                models += sorted(filtered[filtered["Make"]==p["brand"]]["Model"].dropna().unique().tolist())
            p["model"] = st.selectbox("Model", models)
            
            if p["brand"]!="Other" and p["model"]!="My Car":
                match = filtered[(filtered["Make"]==p["brand"]) & (filtered["Model"]==p["model"])]
                if not match.empty: p["mileage"] = float(match.iloc[0]["kmpl"])
            
            p["mileage"] = st.number_input("Mileage (km/L)", value=float(p["mileage"]))

        st.markdown("---")
        
        # 3. EV Logic
        st.subheader("3. Target EV")
        p["ev_mode"] = st.radio("EV Mode", ["Generic EV", "Specific EV"])
        if p["ev_mode"] == "Generic EV":
            p["ev_choice"] = st.selectbox("Type", list(EV_SEGMENT.keys()))
            p["ev_eff"] = EV_SEGMENT[p["ev_choice"]]
        else:
            # Load EV DB
            ev_list = st.session_state.ev_df
            if not ev_list.empty:
                b_col = next((c for c in ev_list.columns if "brand" in c.lower() or "make" in c.lower()), None)
                m_col = next((c for c in ev_list.columns if "model" in c.lower() or "name" in c.lower()), None)
                eff_col = next((c for c in ev_list.columns if "kwh" in c.lower()), None)
                
                if b_col:
                    brands_ev = sorted(ev_list[b_col].dropna().unique())
                    ev_brand = st.selectbox("EV Brand", brands_ev)
                    ev_models = sorted(ev_list[ev_list[b_col]==ev_brand][m_col].dropna().unique())
                    ev_model = st.selectbox("EV Model", ev_models)
                    
                    row = ev_list[(ev_list[b_col]==ev_brand) & (ev_list[m_col]==ev_model)]
                    if not row.empty and eff_col: 
                        p["ev_eff"] = float(row.iloc[0].get(eff_col, 15.0))
            else:
                p["ev_eff"] = st.number_input("Efficiency", value=15.0)

        st.markdown("---")
        
        # 4. Driving
        st.subheader("4. Driving")
        p["daily_km"] = st.slider("Daily km", 10, 200, value=int(p.get("daily_km", 40)))
        
        # --- FIXED AVERAGE ELEC PRICE DISPLAY ---
        p["use_avg_elec"] = st.checkbox(f"Use Avg Elec Price", value=True)
        if p["use_avg_elec"]:
            st.caption(f"Using average: **{p['currency_symbol']}{p['elec_price']} / kWh**")
        else:
            p["elec_price"] = st.number_input("Custom Elec Price", value=float(p["elec_price"]))

        st.markdown("---")
        if st.button("Run Simulation 🚀"):
            st.session_state.run_triggered = True

    # --- MAIN DISPLAY ---
    col_main_left, col_main_right = st.columns([1,3])
    with col_main_left:
        st.title("🌱 Eco-Switch")
        st.markdown(f"**{p['fuel']}** vs **{p.get('ev_choice', 'EV')}**")
        st.caption(f"Location: {p['country']}")

    with col_main_right:
        if st.session_state.get("run_triggered"):
            res, gef_val = compute_simulation(p)
            total_saved = res["Cumulative"].iloc[-1]
            total_co2 = res["CO2 Avoided (kg)"].sum()
            curr = p["currency_symbol"]

            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card"><div class="kpi-label">💰 5-Year Savings</div><div class="kpi-value">{curr}{total_saved:,.0f}</div></div>
                <div class="kpi-card"><div class="kpi-label">🌍 CO₂ Reduced (kg)</div><div class="kpi-value">{total_co2:,.0f}</div></div>
                <div class="kpi-card"><div class="kpi-label">⚡ Grid Factor</div><div class="kpi-value" style="-webkit-text-fill-color: #FAFAFA;">{gef_val:.3f}</div></div>
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["💸 Financial", "🌿 Emissions"])
            with tab1:
                fig = px.area(res, x="Year", y="Cumulative", title=f"Cumulative Savings ({curr})", color_discrete_sequence=["#00E676"])
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                fig2 = px.bar(res, x="Year", y="CO2 Avoided (kg)", title="CO₂ Avoided", color_discrete_sequence=["#00E676"])
                fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig2, use_container_width=True)
            
            if total_saved > 0:
                st.markdown(f"""<div class="success-box"><h3>✅ Switch Recommended</h3><p>You save <b>{curr}{total_saved:,.0f}</b></p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="warning-box"><h3>⚠️ Keep Current Car</h3><p>Cost increases by <b>{curr}{abs(total_saved):,.0f}</b></p></div>""", unsafe_allow_html=True)

            st.download_button("📥 Download Report", res.to_csv(index=False).encode("utf-8"), "report.csv", "text/csv")
            
            # --- RESTORED DETAIL TABLE ---
            with st.expander("🔍 View Detailed Data"):
                st.dataframe(res.style.format({
                    "Fuel Price": "{:.2f}", 
                    "Cost Fuel": "{:.2f}", 
                    "Cost EV": "{:.2f}", 
                    "Savings": "{:.2f}",
                    "Cumulative": "{:.2f}"
                }), use_container_width=True)

        else:
            st.markdown("<div style='text-align: center; padding: 50px; opacity: 0.6;'><h2>Ready?</h2><p>Click Run Simulation in sidebar.</p></div>", unsafe_allow_html=True)