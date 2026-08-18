import streamlit as st
import asyncio
import plotly.express as px
from connectors import EcommerceDataEngine

st.set_page_config(page_title="E-Commerce Delivery & Loyalty Intelligence", layout="wide")

st.title("📦 Universal Delivery Delays & Customer Loyalty Portal")

col_input1, col_input2 = st.columns(2)
with col_input1:
    ecommerce_name = st.text_input("Enter E-Commerce Platform / Store Name", value="Amazon India")
with col_input2:
    custom_areas = st.text_input("Enter Target Areas (Comma-Separated)", value="Bangalore South, Delhi NCR, Mumbai West, Hyderabad Central")

areas_list = [a.strip() for a in custom_areas.split(",") if a.strip()]

with st.expander("🔑 Connect Live Store API (Optional)"):
    store_url = st.text_input("Shopify Store URL (e.g. yourstore.myshopify.com)")
    admin_token = st.text_input("Shopify Admin API Token", type="password")

if st.button("🚀 Fetch & Analyze Data", type="primary"):
    with st.spinner("Processing delivery telemetry and calculating repurchase cadences..."):
        if store_url and admin_token:
            try:
                df = asyncio.run(EcommerceDataEngine.fetch_shopify(store_url, admin_token))
            except Exception as e:
                st.error(f"API Connection Failed: {e}. Falling back to simulation.")
                df = EcommerceDataEngine.generate_synthetic_data(ecommerce_name, areas_list)
        else:
            df = EcommerceDataEngine.generate_synthetic_data(ecommerce_name, areas_list)
        
        st.session_state["analyzed_df"] = df

if "analyzed_df" in st.session_state:
    df = st.session_state["analyzed_df"]
    
    unique_areas = sorted([str(a) for a in df["area"].dropna().unique()])
    selected_area = st.selectbox("📍 Select Target Area for Deep Dive Loyalty Analysis", ["All Combined Areas"] + unique_areas)
    
    active_df = df if selected_area == "All Combined Areas" else df[df["area"] == selected_area]
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_orders = len(active_df)
    breach_rate = (active_df["is_late"].mean() * 100) if total_orders > 0 else 0.0
    ontime_gap = active_df[active_df["is_late"] == 0]["repurchase_gap_days"].mean()
    late_gap = active_df[active_df["is_late"] == 1]["repurchase_gap_days"].mean()
    churn_diff = late_gap - ontime_gap

    kpi1.metric("Analyzed Orders", f"{total_orders:,}")
    kpi2.metric("SLA Breach Rate", f"{breach_rate:.1f}%")
    kpi3.metric("On-Time Repurchase Gap", f"{ontime_gap:.1f} days")
    kpi4.metric("Delay Churn Penalty", f"+{churn_diff:.1f} days", delta_color="inverse")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Regional SLA Breach Rates")
        area_agg = df.groupby("area")["is_late"].mean().reset_index()
        area_agg["breach_pct"] = area_agg["is_late"] * 100
        fig_area = px.bar(area_agg, x="area", y="breach_pct", color="breach_pct", color_continuous_scale="Reds")
        st.plotly_chart(fig_area, use_container_width=True)

    with c2:
        st.subheader(f"Loyalty Retention Breakdown ({selected_area})")
        loyalty_agg = active_df.groupby(["is_late", "loyalty_tier"]).size().reset_index(name="count")
        loyalty_agg["Delivery_Status"] = loyalty_agg["is_late"].map({0: "On-Time", 1: "Delayed"})
        fig_loyalty = px.bar(loyalty_agg, x="loyalty_tier", y="count", color="Delivery_Status", barmode="group",
                             color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig_loyalty, use_container_width=True)


