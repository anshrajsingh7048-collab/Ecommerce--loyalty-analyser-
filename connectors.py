import asyncio
import re
import httpx
import pandas as pd
import numpy as np
from datetime import datetime

class EcommerceDataEngine:
    @staticmethod
    async def fetch_shopify(store_url: str, access_token: str, max_orders: int = 5000, batch_size: int = 250) -> pd.DataFrame:
        base_url = f"https://{store_url.strip('/')}/admin/api/2024-01/orders.json"
        headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
        params = {"status": "any", "limit": batch_size, "fields": "id,created_at,customer,fulfillments,shipping_address"}
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
                        "area": shipping.get("city") or shipping.get("province") or "Metro Central",
                        "platform": "Shopify"
                    })
                match = re.search(r'<([^>]+)>;\s*rel="next"', res.headers.get("Link", ""))
                next_url = match.group(1) if match else None
                if not next_url:
                    break
        return EcommerceDataEngine.process_metrics(pd.DataFrame(all_records))

    @staticmethod
    def generate_synthetic_data(platform_name: str, areas: list, n_rows: int = 5000) -> pd.DataFrame:
        np.random.seed(42)
        prefix = platform_name.upper().replace(" ", "_")[:5]
        order_dates = pd.date_range(end=datetime.now(), periods=n_rows, freq="h")
        promised_offsets = np.random.choice([2, 3, 4, 5], size=n_rows, p=[0.25, 0.45, 0.20, 0.10])
        actual_offsets = np.random.exponential(scale=2.8, size=n_rows) + 1.0

        records = {
            "order_id": [f"{prefix}_{i:06d}" for i in range(1, n_rows + 1)],
            "customer_id": [f"CUST_{np.random.randint(1, int(n_rows * 0.4)):05d}" for _ in range(n_rows)],
            "order_date": order_dates,
            "promised_delivery_date": order_dates + pd.to_timedelta(promised_offsets, unit="D"),
            "actual_delivery_date": order_dates + pd.to_timedelta(actual_offsets, unit="D"),
            "area": np.random.choice(areas, size=n_rows),
            "platform": platform_name
        }
        return EcommerceDataEngine.process_metrics(pd.DataFrame(records))

    @staticmethod
    def process_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"], errors="coerce")
        df["promised_delivery_date"] = pd.to_datetime(df["promised_delivery_date"], errors="coerce").fillna(
            df["order_date"] + pd.Timedelta(days=3)
        )
        df["delay_days"] = (df["actual_delivery_date"] - df["promised_delivery_date"]).dt.total_seconds() / 86400.0
        df["is_late"] = np.where(df["delay_days"] > 0, 1, 0)
        df = df.sort_values(by=["customer_id", "order_date"])
        df["next_order_date"] = df.groupby("customer_id")["order_date"].shift(-1)
        df["repurchase_gap_days"] = (df["next_order_date"] - df["order_date"]).dt.total_seconds() / 86400.0
        
        base_ontime = df[df["is_late"] == 0]["repurchase_gap_days"].mean()
        base_late = df[df["is_late"] == 1]["repurchase_gap_days"].mean()
        default_ontime = 24.0 if pd.isna(base_ontime) else base_ontime
        default_late = (default_ontime + 32.0) if pd.isna(base_late) else base_late
        
        df["repurchase_gap_days"] = df["repurchase_gap_days"].fillna(
            np.where(df["is_late"] == 1, default_late, default_ontime)
        )
        df["loyalty_tier"] = np.where(
            df["repurchase_gap_days"] <= 30, "High Loyalty (<=30d)",
            np.where(df["repurchase_gap_days"] <= 60, "Moderate Loyalty (31-60d)", "At-Risk / Churned (>60d)")
        )
        return df
      
