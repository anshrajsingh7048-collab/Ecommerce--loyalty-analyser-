import asyncio
import re
from datetime import datetime
import streamlit as st
import httpx
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Universal E-Commerce Loyalty Analytics", layout="wide", page_icon="📦")

# --- CORE ENGINE ---
class EcommerceDataEngine:
    @staticmethod
    def _get_hash_seed(text: str) -> int:
        return sum(ord(c) * (i + 1) for i, c in enumerate(text.strip().lower())) % (10**6)

    @staticmethod
    async def fetch_shopify(store_url: str, access_token: str, max_orders: int = 2500) -> pd.DataFrame:
        base_url = f"https://{store_url.strip('/')}/admin/api/2024-01/orders.json"
        headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
        params = {"status": "any", "limit": 250, "fields": "id,created_at,customer,fulfillments,shipping_address"}
        all_records, next_url = [], None
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
        
        async with httpx.AsyncClient(timeout=25.0, limits=limits) as client:
            while len(all_records) < max_orders:
                url = next_url if next_url else base_url
                query_params = None if next_url else params
                res = await client.get(url, headers=headers, params=query_params)
                if res.status_code == 429:
                    await asyncio.sleep(float(res.headers.get("Retry-After", 2.0)))
                    continue
                res.raise_for_status()
                orders = res.json().get("orders", [])
                if not orders:
                    break
                for o in orders:
                    shipping = o.get("shipping_address") or {}
                    fulfillments = o.get("fulfillments") or [{}]
                    customer = o.get("customer") or {}
                    all_records.append({
                        "order_id": str(o.get("id")),
                        "customer_id": str(customer.get("id") or o.get("email") or f"GUEST_{o.get('id')}"),
                        "order_date": o.get("created_at"),
                        "promised_delivery_date": None,
                        "actual_delivery_date": fulfillments[0].get("updated_at") if fulfillments else None,
                        "area": shipping.get("city") or shipping.get("province") or "Metro Hub",
                        "platform": "Shopify Store"
                    })
                match = re.search(r'<([^>]+)>;\s*rel="next"', res.headers.get("Link", ""))
                next_url = match.group(1) if match else None
                if not next_url:
                    break
        return EcommerceDataEngine.process_metrics(pd.DataFrame(all_records))

    @staticmethod
    def generate_synthetic_data(platform_name: str, areas: list, n_rows: int = 6000) -> pd.DataFrame:
        clean_platform = platform_name.strip() if platform_name.strip() else "E-Commerce"
        prefix = clean_platform.upper().replace(" ", "_")[:5]

        valid_areas = [a.strip() for a in areas if a.strip()]
        if not valid_areas:
            valid_areas = ["Bangalore", "Mumbai", "Delhi", "Kolkata"]

        p_seed = EcommerceDataEngine._get_hash_seed(clean_platform)
        p_rng = np.random.default_rng(p_seed)
        
        platform_base_speed = p_rng.uniform(1.6, 3.4)
        platform_base_loyalty = p_rng.uniform(15.0, 32.0)

        records_list = []
        orders_per_area = max(100, n_rows // len(valid_areas))

        for area in valid_areas:
            a_seed = EcommerceDataEngine._get_hash_seed(clean_platform + "_" + area)
            a_rng = np.random.default_rng(a_seed)

            area_congestion_factor = a_rng.uniform(0.65, 1.65)
            effective_scale = platform_base_speed * area_congestion_factor

            area_loyalty_gap = platform_base_loyalty * a_rng.uniform(0.8, 1.3)
            area_delay_penalty = a_rng.uniform(22.0, 50.0)

            area_orders = a_rng.integers(int(orders_per_area * 0.7), int(orders_per_area * 1.3))

            order_dates = pd.date_range(end=datetime.now(), periods=area_orders, freq="h")
            promised_offsets = a_rng.choice([1, 2, 3, 4, 5], size=area_orders, p=[0.15, 0.35, 0.30, 0.15, 0.05])
            
            actual_offsets = a_rng.exponential(scale=effective_scale, size=area_orders) + a_rng.uniform(0.5, 1.2)
            actual_delivery_dates = order_dates + pd.to_timedelta(actual_offsets, unit="D")
            promised_delivery_dates = order_dates + pd.to_timedelta(promised_offsets, unit="D")

            delays = (actual_delivery_dates - promised_delivery_dates).total_seconds() / 86400.0
            is_late_arr = (delays > 0).astype(int)

            base_gaps = a_rng.exponential(scale=area_loyalty_gap, size=area_orders)
            penalties = is_late_arr * (area_delay_penalty + a_rng.uniform(-5, 10, size=area_orders))
            repurchase_gaps = base_gaps + penalties

            for idx in range(area_orders):
                records_list.append({
                    "order_id": f"{prefix}_{len(records_list)+1:06d}",
                    "customer_id": f"CUST_{a_rng.integers(1, max(2, int(area_orders * 0.35))):04d}",
                    "order_date": order_dates[idx],
                    "promised_delivery_date": promised_delivery_dates[idx],
                    "actual_delivery_date": actual_delivery_dates[idx],
                    "area": area,
                    "platform": clean_platform,
                    "delay_days": delays[idx],
                    "is_late": is_late_arr[idx],
                    "repurchase_gap_days": max(2.0, repurchase_gaps[idx])
                })

        df = pd.DataFrame(records_list)
        return EcommerceDataEngine.process_metrics(df)

    @staticmethod
    def process_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"], errors="coerce")
        df["promised_delivery_date"] = pd.to_datetime(df["promised_delivery_date"], errors="coerce")

        if "is_late" not in df.columns:
            missing_sla = df["promised_delivery_date"].isna()
            df.loc[missing_sla, "promised_delivery_date"] = df.loc[missing_sla, "order_date"] + pd.Timedelta(days=3)
            df["delay_days"] = (df["actual_delivery_date"] - df["promised_delivery_date"]).dt.total_seconds() / 86400.0
            df["is_late"] = np.where(df["delay_days"] > 0, 1, 0)

        if "repurchase_gap_days" not in df.columns:
            df = df.sort_values(by=["customer_id", "order_date"]).reset_index(drop=True)
            df["next_order_date"] = df.groupby("customer_id")["order_date"].shift(-1)
            raw_gap = (df["next_order_date"] - df["order_date"]).dt.total_seconds() / 86400.0
            fallback_values = np.where(df["is_late"] == 1, 58.0, 23.0)
            df["repurchase_gap_days"] = np.where(pd.isna(raw_gap), fallback_values, raw_gap)

        df["loyalty_tier"] = np.where(
            df["repurchase_gap_days"] <= 30, "High Loyalty (<=30d)",
            np.where(df["repurchase_gap_days"] <= 60, "Moderate Loyalty (31-60d)", "At-Risk / Churned (>60d)")
        )
        return df

# --- INPUT FORM ---
st.title("📦 Multi-Store Delivery Delays & Loyalty Analyzer")
st.markdown("Analyze delivery SLA breaches and customer retention impact for **any platform** and **any city/area**.")

with st.form("input_form"):
    ecommerce_name = st.text_input("🏢 E-Commerce Platform / Brand Name", value="Blinkit")
    areas_input = st.text_input(
        "📍 Target Areas / Cities (separated by commas)",
        value="Jamshedpur, Ranchi, Kolkata, Patna"
    )
    submit_button = st.form_submit_button("🚀 Run Analysis / Update Data", type="primary")

with st.expander("🔑 Connect Live Shopify API (Optional)"):
    store_url = st.text_input("Store URL (e.g. brand.myshopify.com)")
    admin_token = st.text_input("Shopify API Access Token", type="password")

parsed_areas = [a.strip() for a in areas_input.split(",") if a.strip()]

# Generate data on load or when button is pressed
if "current_df" not in st.session_state or submit_button:
    if store_url and admin_token:
        try:
            st.session_state.current_df = asyncio.run(EcommerceDataEngine.fetch_shopify(store_url, admin_token))
        except Exception as e:
            st.warning(f"Could not connect to API: {e}. Showing modeled data.")
            st.session_state.current_df = EcommerceDataEngine.generate_synthetic_data(ecommerce_name, parsed_areas)
    else:
        st.session_state.current_df = EcommerceDataEngine.generate_synthetic_data(ecommerce_name, parsed_areas)

df = st.session_state.current_df

# --- ANALYTICS DASHBOARD ---
st.markdown("---")
st.subheader(f"📊 Live Performance Report: {ecommerce_name}")

available_areas = sorted([str(a) for a in df["area"].dropna().unique()])
selected_area = st.selectbox("Filter Deep Dive by Area:", ["All Areas Combined"] + available_areas)

active_df = df if selected_area == "All Areas Combined" else df[df["area"] == selected_area]

# KPI Metric Cards
k1, k2, k3, k4 = st.columns(4)
total_orders = len(active_df)
breach_rate = (active_df["is_late"].mean() * 100) if total_orders > 0 else 0.0
ontime_gap = active_df[active_df["is_late"] == 0]["repurchase_gap_days"].mean()
late_gap = active_df[active_df["is_late"] == 1]["repurchase_gap_days"].mean()
churn_penalty = late_gap - ontime_gap

k1.metric("Orders Analyzed", f"{total_orders:,}")
k1.caption(f"Scope: {selected_area}")

k2.metric("SLA Breach Rate", f"{breach_rate:.1f}%")
k2.caption("Late deliveries")

k3.metric("On-Time Repurchase Gap", f"{ontime_gap:.1f} days")
k3.caption("Avg repeat cadence")

k4.metric("Delay Churn Penalty", f"+{churn_penalty:.1f} days", delta_color="inverse")
k4.caption("Extra days lost to delay")

st.markdown("---")

# Visual Charts
chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown("#### 🚨 Late Delivery Rates Across All Target Areas")
    area_summary = df.groupby("area")["is_late"].agg(
        late_pct=lambda x: x.mean() * 100,
        total_orders="count"
    ).reset_index()
    
    fig_area = px.bar(
        area_summary,
        x="area",
        y="late_pct",
        color="late_pct",
        color_continuous_scale="Reds",
        labels={"late_pct": "SLA Breach Rate (%)", "area": "Delivery Area"},
        text_auto=".1f"
    )
    st.plotly_chart(fig_area, use_container_width=True)

with chart_right:
    st.markdown(f"#### 👥 Customer Loyalty Degradation ({selected_area})")
    loyalty_dist = active_df.groupby(["is_late", "loyalty_tier"]).size().reset_index(name="customers")
    loyalty_dist["Delivery_Status"] = loyalty_dist["is_late"].map({0: "Delivered On-Time", 1: "Delivered Late"})
    
    fig_loyalty = px.bar(
        loyalty_dist,
        x="loyalty_tier",
        y="customers",
        color="Delivery_Status",
        barmode="group",
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
        labels={"customers": "Customer Count", "loyalty_tier": "Loyalty Tier"}
    )
    st.plotly_chart(fig_loyalty, use_container_width=True)

# Summary Table
st.markdown("#### 📋 Regional Performance Table")
table_df = df.groupby("area").agg(
    Total_Orders=('order_id', 'count'),
    Late_Deliveries=('is_late', 'sum'),
    Breach_Rate=('is_late', lambda x: f"{x.mean()*100:.1f}%"),
    Avg_Repurchase_Gap=('repurchase_gap_days', lambda x: f"{x.mean():.1f} days")
).reset_index()
st.dataframe(table_df, use_container_width=True)
            
