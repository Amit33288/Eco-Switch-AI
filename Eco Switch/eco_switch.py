import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from pathlib import Path
from io import BytesIO

# Set page config
st.set_page_config(page_title="Eco-Switch Pro • Polished", layout="wide", initial_sidebar_state="expanded")

# ---------- ENHANCED VISUAL STYLE (CSS) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global settings */
    html, body, [class*="css"]  {
      background-color: #0E1117;
      font-family: 'Poppins', sans-serif;
      color: #FAFAFA;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
      background-color: #161B22;
      border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Headers */
    h1, h2, h3 {
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    
    /* Modern Glassmorphism KPI Cards */
    .kpi-container {
      display: flex;
      gap: 20px;
      margin-bottom: 30px;
    }
    .kpi-card {
      background: rgba(30, 34, 45, 0.6);
      backdrop-filter: blur(10px);
      border-radius: 16px;
      padding: 20px;
      flex: 1;
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 4px 20px rgba(0,0,0,0.2);
      transition: transform 0.2s ease;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      border-color: rgba(0, 230, 118, 0.3);
    }
    .kpi-label { color: #8b949e; font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { 
      background: linear-gradient(90deg, #00E676, #69F0AE);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 700; 
      font-size: 32px; 
      margin-top: 5px; 
    }
    
    /* Glowing Action Button */
    .stButton>button {
      background: linear-gradient(90deg, #00C853, #00E676);
      border: none;
      border-radius: 12px;
      color: #000;
      font-weight: 700;
      padding: 0.6rem 1.2rem;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(0, 230, 118, 0.3);
    }
    .stButton>button:hover { 
      box-shadow: 0 6px 20px rgba(0, 230, 118, 0.5);
      transform: scale(1.02);
    }

    /* Result Containers */
    .success-box {
      background: rgba(0, 200, 83, 0.1);
      border-left: 4px solid #00E676;
      padding: 20px;
      border-radius: 8px;
      margin-top: 20px;
    }
    .warning-box {
      background: rgba(255, 171, 0, 0.1);
      border-left: 4px solid #FFAB00;
      padding: 20px;
      border-radius: 8px;
      margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- COUNTRY DATA (Prices & Currencies) ----------
COUNTRY_DATA = {
    "Algeria": {"currency": "DZD", "elec_price": 5.5},
    "Argentina": {"currency": "$", "elec_price": 55.0},
    "Australia": {"currency": "A$", "elec_price": 0.35},
    "Austria": {"currency": "€", "elec_price": 0.38},
    "Belgium": {"currency": "€", "elec_price": 0.36},
    "Brazil": {"currency": "R$", "elec_price": 0.85},
    "Canada": {"currency": "C$", "elec_price": 0.16},
    "Chile": {"currency": "$", "elec_price": 140.0},
    "China": {"currency": "¥", "elec_price": 0.60},
    "Colombia": {"currency": "$", "elec_price": 750.0},
    "Czech Republic": {"currency": "Kč", "elec_price": 8.5},
    "Denmark": {"currency": "kr", "elec_price": 2.8},
    "Egypt": {"currency": "E£", "elec_price": 1.4},
    "Finland": {"currency": "€", "elec_price": 0.25},
    "France": {"currency": "€", "elec_price": 0.23},
    "Germany": {"currency": "€", "elec_price": 0.40},
    "Greece": {"currency": "€", "elec_price": 0.22},
    "Hungary": {"currency": "Ft", "elec_price": 36.0},
    "India": {"currency": "₹", "elec_price": 8.0},
    "Indonesia": {"currency": "Rp", "elec_price": 1450.0},
    "Ireland": {"currency": "€", "elec_price": 0.38},
    "Israel": {"currency": "₪", "elec_price": 0.60},
    "Italy": {"currency": "€", "elec_price": 0.35},
    "Japan": {"currency": "¥", "elec_price": 31.0},
    "Malaysia": {"currency": "RM", "elec_price": 0.50},
    "Mexico": {"currency": "$", "elec_price": 1.9},
    "Netherlands": {"currency": "€", "elec_price": 0.35},
    "New Zealand": {"currency": "NZ$", "elec_price": 0.30},
    "Norway": {"currency": "kr", "elec_price": 1.5},
    "Pakistan": {"currency": "₨", "elec_price": 45.0},
    "Philippines": {"currency": "₱", "elec_price": 11.0},
    "Poland": {"currency": "zł", "elec_price": 0.95},
    "Portugal": {"currency": "€", "elec_price": 0.24},
    "Russia": {"currency": "₽", "elec_price": 5.5},
    "Saudi Arabia": {"currency": "﷼", "elec_price": 0.18},
    "Singapore": {"currency": "S$", "elec_price": 0.30},
    "South Africa": {"currency": "R", "elec_price": 2.80},
    "South Korea": {"currency": "₩", "elec_price": 140.0},
    "Spain": {"currency": "€", "elec_price": 0.24},
    "Sweden": {"currency": "kr", "elec_price": 2.5},
    "Switzerland": {"currency": "CHF", "elec_price": 0.32},
    "Taiwan": {"currency": "NT$", "elec_price": 3.0},
    "Thailand": {"currency": "฿", "elec_price": 4.7},
    "Turkey": {"currency": "₺", "elec_price": 2.6},
    "UAE": {"currency": "AED", "elec_price": 0.30},
    "UK": {"currency": "£", "elec_price": 0.34},
    "USA": {"currency": "$", "elec_price": 0.16},
    "United Kingdom": {"currency": "£", "elec_price": 0.34},
    "United States": {"currency": "$", "elec_price": 0.16},
    "Vietnam": {"currency": "₫", "elec_price": 2000.0}
}

DEFAULT_CURRENCY = "$"
DEFAULT_ELEC_PRICE = 0.14  # Global avg approx

# ---------- HELPERS ----------
def get_country_info(country_name):
    if country_name in COUNTRY_DATA:
        return COUNTRY_DATA[country_name]
    for k, v in COUNTRY_DATA.items():
        if k.lower() in country_name.lower():
            return v
    return {"currency": DEFAULT_CURRENCY, "elec_price": DEFAULT_ELEC_PRICE}

def safe_read_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_float(x):
    try:
        return float(x)
    except:
        m = re.search(r"\d+(\.\d+)?", str(x))
        return float(m.group(0)) if m else None

def prepare_mileage(raw):
    if raw.empty:
        return pd.DataFrame(columns=["Make","Model","Fuel","kmpl"])
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

# ---------- LOAD FILES ----------
if "mileage_db" not in st.session_state:
    raw_mileage = safe_read_csv("Fuel Car Mileage.csv")
    st.session_state.mileage_db = prepare_mileage(raw_mileage)

preds_df = safe_read_csv("fuel_price_predictions_2025_2030.csv")
hist_prices_df = safe_read_csv("clean_fuel_prices.csv") 
master_prices_df = safe_read_csv("Master_Fuel_Prices_2015_2024.csv")
ev_df = safe_read_csv("clean_ev_specs.csv")
gef_df = safe_read_csv("clean_gef.csv")

# ---------- DEFAULTS ----------
SEGMENT_MILEAGE = {
    "Small Hatchback (e.g., Alto, Swift)": 18,
    "Sedan (e.g., City, Civic)": 14,
    "Compact SUV (e.g., Creta, Nexon)": 12,
    "Large SUV / Truck (e.g., Fortuner, F-150)": 9,
    "Sports Car": 7
}
EV_SEGMENT = {"Budget City EV":12, "Standard Sedan EV":15, "Performance SUV EV":22}

# ---------- STATE INITIALIZATION ----------
if "params" not in st.session_state:
    st.session_state.params = {
        "country": "India",
        "fuel": "Petrol",
        "car_mode": "By Segment (Easy)",
        "segment": list(SEGMENT_MILEAGE.keys())[0],
        "brand": "Other",
        "model": "My Car",
        "mileage": 15.0,
        "ev_mode": "Generic EV",
        "ev_choice": list(EV_SEGMENT.keys())[0],
        "daily_km": 40,
        "use_avg_elec": True,
        "elec_price": 8.0,
        "currency_symbol": "₹",
        "manual_fuel_price": 0.0
    }

# TRACKERS for Auto-Detect logic
if "last_country" not in st.session_state:
    st.session_state.last_country = st.session_state.params["country"]
if "last_fuel" not in st.session_state:
    st.session_state.last_fuel = st.session_state.params["fuel"]

# ---------- SIDEBAR (INPUTS) ----------
p = st.session_state.params

with st.sidebar:
    st.title("🚘 Simulation Parameters")
    st.subheader("1. Location")
    
    # Merge countries
    pred_countries = set(preds_df["country"].dropna().unique()) if not preds_df.empty else set()
    hist_countries = set(hist_prices_df["country"].dropna().unique()) if not hist_prices_df.empty else set()
    master_countries = set(master_prices_df["Country"].dropna().unique()) if not master_prices_df.empty else set()
    all_countries = sorted(list(pred_countries.union(hist_countries).union(master_countries)))
    
    if not all_countries: all_countries = ["India","USA","UK"]
        
    selected_country = st.selectbox("Select Country", all_countries, index=all_countries.index(p["country"]) if p["country"] in all_countries else 0)

    # Country Change Logic
    if selected_country != st.session_state.last_country:
        country_info = get_country_info(selected_country)
        p["country"] = selected_country
        p["elec_price"] = country_info["elec_price"]
        p["currency_symbol"] = country_info["currency"]
        p["manual_fuel_price"] = 0.0 # Reset to force re-detect
        st.session_state.last_country = selected_country
        st.rerun()

    st.markdown("---")
    st.subheader("2. Your Current Car")
    
    # Fuel Select
    selected_fuel = st.radio("Fuel Type", ["Petrol","Diesel"], index=0 if p["fuel"]=="Petrol" else 1, horizontal=True)

    # ⚡⚡⚡ FUEL CHANGE LOGIC (The Fix) ⚡⚡⚡
    # If fuel changed, we MUST reset manual price so it looks up the new fuel price
    if selected_fuel != st.session_state.last_fuel:
        p["fuel"] = selected_fuel
        p["manual_fuel_price"] = 0.0  # Reset!
        st.session_state.last_fuel = selected_fuel
        st.rerun()
    
    # Update param if it was just a normal interaction (though rerun handles mostly)
    p["fuel"] = selected_fuel

    # --- AUTO DETECT PRICE ---
    st.markdown("##### ⛽ Current Fuel Price")
    
    detected_price = None
    
    # 1. Master File (Priority)
    if not master_prices_df.empty:
        try:
            m_row = master_prices_df[master_prices_df["Country"].str.lower() == p["country"].lower()]
            if not m_row.empty:
                f_type_match = m_row[m_row["Fuel Type"].str.contains(p["fuel"], case=False, na=False)]
                if not f_type_match.empty:
                    # Look for most recent year data
                    for y in ["2024", "2023", "2022"]:
                        val = f_type_match.iloc[0].get(y)
                        if pd.notna(val):
                            detected_price = float(val)
                            break
        except: pass

    # 2. Predictions File (Backup)
    if detected_price is None and not preds_df.empty:
        try:
            cp = preds_df[preds_df['country'].str.lower() == p["country"].lower()]
            fp = cp[cp['fuel'].str.lower().str.contains(p["fuel"].lower(), na=False)].sort_values('year')
            if not fp.empty: detected_price = fp.iloc[0]['predicted_price']
        except: pass

    # 3. Historical Clean File (Last Resort)
    if detected_price is None and not hist_prices_df.empty:
         try:
            ch = hist_prices_df[hist_prices_df['country'].str.lower() == p["country"].lower()]
            price_col = "petrol_price" if "petrol" in p["fuel"].lower() else "diesel_price"
            if not ch.empty and price_col in ch.columns:
                detected_price = float(ch.sort_values('year', ascending=False).iloc[0][price_col])
         except: pass

    # Apply detected price if we have 0.0 (meaning reset state)
    if p["manual_fuel_price"] == 0.0:
        if detected_price:
            p["manual_fuel_price"] = detected_price
        else:
            p["manual_fuel_price"] = 100.0 # Only fallback if truly nothing found

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1: st.markdown(f"**{p['currency_symbol']}**")
    with col_f2:
        p["manual_fuel_price"] = st.number_input(
            "Price/Liter", 
            value=float(p["manual_fuel_price"]), 
            format="%.2f",
            help="This is the price used for calculations."
        )

    if detected_price:
        st.caption(f"✅ Auto-detected {p['fuel']} price")
    else:
        if p["manual_fuel_price"] == 100.0:
            st.warning("⚠️ Price not found. Please update!")

    st.markdown("---")
    
    # Car Mode
    p["car_mode"] = st.radio("Car Selection Mode", ["By Segment (Easy)", "By Exact Model (Advanced)"], index=0 if p["car_mode"]=="By Segment (Easy)" else 1)

    if p["car_mode"] == "By Segment (Easy)":
        segs = list(SEGMENT_MILEAGE.keys())
        p["segment"] = st.selectbox("Car Segment", segs, index=segs.index(p["segment"]) if p["segment"] in segs else 0)
        p["mileage"] = SEGMENT_MILEAGE[p["segment"]]
        st.info(f"Assumed mileage: {p['mileage']} km/L")
    else:
        filtered = st.session_state.mileage_db.copy()
        
        with st.expander("➕ Add Custom Car"):
            c_make = st.text_input("Make", key="new_make")
            c_model = st.text_input("Model", key="new_model")
            c_fuel = st.selectbox("Fuel", ["Petrol", "Diesel"], key="new_fuel")
            c_kmpl = st.number_input("Mileage", min_value=1.0, value=15.0, key="new_kmpl")
            if st.button("Add"):
                if c_make and c_model:
                    new_row = pd.DataFrame([{"Make": c_make, "Model": c_model, "Fuel": c_fuel, "kmpl": c_kmpl}])
                    st.session_state.mileage_db = pd.concat([st.session_state.mileage_db, new_row], ignore_index=True)
                    st.rerun()

        if "Fuel" in filtered.columns:
            mask = filtered["Fuel"].str.lower().str.contains(p["fuel"].lower(), na=False)
            if mask.any(): filtered = filtered[mask]

        brands = sorted(filtered["Make"].dropna().unique().tolist())
        brand_options = ["Other"] + brands
        
        p["brand"] = st.selectbox("Brand", brand_options, index=brand_options.index(p.get("brand","Other")) if p.get("brand") in brand_options else 0)

        if p["brand"] != "Other":
            models = sorted(filtered[filtered["Make"] == p["brand"]]["Model"].dropna().unique().tolist())
            model_options = ["My Car"] + models
        else:
            model_options = ["My Car","Other"]
        p["model"] = st.selectbox("Model", model_options, index=model_options.index(p.get("model","My Car")) if p.get("model") in model_options else 0)

        detected_kmpl = None
        if p["brand"] != "Other" and p["model"] not in ["My Car","Other"]:
            match = filtered[(filtered["Make"]==p["brand"]) & (filtered["Model"]==p["model"])]
            if not match.empty: detected_kmpl = parse_float(match.iloc[0]["kmpl"])
        
        if detected_kmpl:
            p["mileage"] = st.number_input("Mileage (km/L)", value=float(detected_kmpl))
        else:
            p["mileage"] = st.number_input("Mileage (km/L)", value=float(p["mileage"]))

    st.markdown("---")
    st.subheader("3. Target EV")
    p["ev_mode"] = st.radio("EV Selection", ["Generic EV","Specific EV"], index=0 if p["ev_mode"]=="Generic EV" else 1)
    if p["ev_mode"] == "Generic EV":
        ev_opts = list(EV_SEGMENT.keys())
        p["ev_choice"] = st.selectbox("EV Type", ev_opts, index=ev_opts.index(p["ev_choice"]) if p["ev_choice"] in ev_opts else 0)
        p["ev_eff"] = EV_SEGMENT[p["ev_choice"]]
    else:
        if not ev_df.empty:
            ev_brand_col = next((c for c in ev_df.columns if re.search("brand|make", c, re.I)), None)
            ev_model_col = next((c for c in ev_df.columns if re.search("model|name", c, re.I)), None)
            if ev_brand_col:
                brands_ev = sorted(ev_df[ev_brand_col].dropna().unique())
                p["ev_brand"] = st.selectbox("EV Brand", brands_ev, index=brands_ev.index(p.get("ev_brand", brands_ev[0])) if p.get("ev_brand") in brands_ev else 0)
                models_ev = sorted(ev_df[ev_df[ev_brand_col]==p["ev_brand"]][ev_model_col].dropna().unique())
                p["ev_model"] = st.selectbox("EV Model", models_ev, index=models_ev.index(p.get("ev_model", models_ev[0])) if p.get("ev_model") in models_ev else 0)
                row = ev_df[(ev_df[ev_brand_col]==p["ev_brand"]) & (ev_df[ev_model_col]==p["ev_model"])]
                if not row.empty and 'kwh_per_100km' in row.columns:
                    p["ev_eff"] = float(row.iloc[0]['kwh_per_100km'])
                else:
                    p["ev_eff"] = 15.0
                st.caption(f"Efficiency: {p['ev_eff']} kWh/100km")
        else:
            p["ev_eff"] = st.number_input("Efficiency", value=15.0)

    st.markdown("---")
    st.subheader("4. Driving & Costs")
    p["daily_km"] = st.slider("Daily Driving (km)", 10, 200, value=int(p.get("daily_km",40)))
    
    p["use_avg_elec"] = st.checkbox(f"Use avg price for {p['country']}?", value=p.get("use_avg_elec", True))
    
    if p["use_avg_elec"]:
        info = get_country_info(p["country"])
        p["elec_price"] = info["elec_price"]
        p["currency_symbol"] = info["currency"]
        st.markdown(f"Elec Price: **{p['currency_symbol']}{p['elec_price']} / kWh**")
    else:
        c1, c2 = st.columns([1,2])
        with c1: p["currency_symbol"] = st.text_input("Sym", value=p.get("currency_symbol","$"))
        with c2: p["elec_price"] = st.number_input("Elec Rate", value=float(p.get("elec_price",8.0)))
    
    st.markdown("---")
    run_sim = st.button("Run Simulation 🚀")

# ---------- COMPUTATION ----------
def compute_simulation(params):
    daily_km = params["daily_km"]
    current_mileage = float(params["mileage"])
    ev_eff = float(params.get("ev_eff", 15.0))
    elec_price = float(params["elec_price"])
    curr = params.get("currency_symbol", "$")
    
    # USE THE MANUAL (OR DETECTED) PRICE
    base_price = float(params["manual_fuel_price"])
    
    years = list(range(2025, 2031))
    # 4% Inflation Projection
    prices = [base_price * (1.04 ** (y - 2025)) for y in years]

    # GEF
    gef = 0.7
    if not gef_df.empty and 'country' in gef_df.columns and 'gef' in gef_df.columns:
        row = gef_df[gef_df['country'].str.lower() == params["country"].lower()]
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
            "Year": int(yr),
            "Currency": curr,
            "Fuel Price": round(price,2),
            "Liters per year": round(liters,2),
            "Cost Fuel": round(cost_fuel,2),
            "Cost EV": round(cost_ev,2),
            "Savings": round(saving,2),
            "Cumulative": round(cumulative,2),
            "CO2 Fuel (kg)": round(co2_fuel,2),
            "CO2 EV (kg)": round(co2_ev,2),
            "CO2 Avoided (kg)": round(co2_fuel - co2_ev,2)
        })

    return pd.DataFrame(records), gef

if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_gef" not in st.session_state:
    st.session_state.last_gef = None

if 'run_sim' in locals() and run_sim:
    df_results, gef_val = compute_simulation(p)
    st.session_state.last_results = df_results
    st.session_state.last_gef = gef_val

# ---------- MAIN UI ----------
col_main_left, col_main_right = st.columns([1,3])
with col_main_left:
    st.title("🌱 Eco-Switch")
    st.markdown(f"**{p['fuel']}** vs **{p.get('ev_choice', p.get('ev_name','EV'))}**")
    st.caption(f"Location: {p['country']}")

with col_main_right:
    if st.session_state.last_results is None:
        st.markdown("<div style='text-align: center; padding: 50px; opacity: 0.6;'><h2>Ready to Switch?</h2><p>Click 'Run Simulation' to start.</p></div>", unsafe_allow_html=True)
    else:
        res = st.session_state.last_results
        total_saved = res["Cumulative"].iloc[-1]
        total_co2 = res["CO2 Avoided (kg)"].sum()
        gef_val = st.session_state.last_gef or 0.7
        curr = p.get("currency_symbol", "$")

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card"><div class="kpi-label">💰 5-Year Savings</div><div class="kpi-value">{curr}{total_saved:,.0f}</div></div>
            <div class="kpi-card"><div class="kpi-label">🌍 CO₂ Reduced (kg)</div><div class="kpi-value">{total_co2:,.0f}</div></div>
            <div class="kpi-card"><div class="kpi-label">⚡ Grid Emission Factor</div><div class="kpi-value" style="-webkit-text-fill-color: #FAFAFA;">{gef_val:.3f}</div></div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["💸 Financial Savings", "🌿 CO₂ Analysis"])
        with tab1:
            fig = px.area(res, x="Year", y="Cumulative", title=f"Cumulative Savings ({curr})", color_discrete_sequence=["#00E676"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig2 = px.bar(res, x="Year", y="CO2 Avoided (kg)", title="Yearly CO₂ Avoided", color_discrete_sequence=["#00E676"])
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig2, use_container_width=True)

        if total_saved > 0:
            st.markdown(f"""<div class="success-box"><h3>✅ Switch Recommended</h3><p>Estimated savings: <b>{curr}{total_saved:,.0f}</b></p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="warning-box"><h3>⚠️ Keep Current Car</h3><p>Negative savings: <b>{curr}{total_saved:,.0f}</b></p></div>""", unsafe_allow_html=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1: st.download_button("📥 Download CSV", res.to_csv(index=False).encode("utf-8"), "eco_switch_report.csv", "text/csv")
        
        with st.expander("🔍 View Detailed Data"):
            disp = res.copy()
            for c in ["Fuel Price", "Cost Fuel", "Cost EV", "Savings", "Cumulative"]:
                disp[c] = disp[c].apply(lambda x: f"{curr}{x:,.2f}")
            st.dataframe(disp, use_container_width=True)