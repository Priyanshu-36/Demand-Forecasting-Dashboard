import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Demand Forecasting Dashboard")


@st.cache_resource
def get_spark():
    return SparkSession.builder.getOrCreate()

spark = get_spark()


@st.cache_resource
def load_data():
    sdf = spark.read.csv("sales_data.csv", header=True)
    sdf = sdf.select(
        F.to_date(F.col("Date"), "dd-MM-yyyy").alias("date"),
        F.col("Product ID").alias("product"),
        F.col("Inventory Level").cast(IntegerType()).alias("inventory"),
        F.col("Demand").cast(IntegerType()).alias("demand")
    ).dropna()
    return sdf

sdf = load_data()
st.sidebar.success(f"{sdf.count():,} rows loaded via PySpark")


st.sidebar.header("Settings")
products = sorted([r["product"] for r in sdf.select("product").distinct().collect()])
selected = st.sidebar.selectbox("Select Product", products)
forecast_days = st.sidebar.slider("Forecast Days", min_value=3, max_value=10, value=7)


grp = (
    sdf.filter(F.col("product") == selected)
       .orderBy("date")
)


rows        = grp.select("date", "demand", "inventory").collect()
dates       = [r["date"] for r in rows]
demands     = [r["demand"] for r in rows]
inventories = [r["inventory"] for r in rows]

last_date = dates[-1]


def wma_forecast(series, forecast_days, window=7):
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    buf = list(series[-window:])
    preds = []
    for _ in range(forecast_days):
        pred = round(float(np.dot(buf[-window:], weights)), 1)
        preds.append(pred)
        buf.append(pred)
    return preds


def lr_forecast(demands, forecast_days):
    n = len(demands)
    X = np.arange(n).reshape(-1, 1)
    y = np.array(demands, dtype=float)
 
    model = LinearRegression().fit(X, y)
 
    split = int(n * 0.8)
    mae   = round(mean_absolute_error(y[split:], model.predict(X[split:])), 2) if split < n else "N/A"
 
    future_X = np.arange(n, n + forecast_days).reshape(-1, 1)
    preds    = [round(float(v), 1) for v in model.predict(future_X).clip(0)]
 
    return preds, mae


import datetime
future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]

wma_preds        = wma_forecast(demands, forecast_days)
lr_preds, mae    = lr_forecast(demands, forecast_days)


st.subheader(f"Demand Forecast — {selected}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates[-60:], demands[-60:],
        color="steelblue", linewidth=1.5, label="Historical Demand")
ax.plot(future_dates, wma_preds,
        color="orange", linewidth=2, linestyle="--", marker="o", markersize=4,
        label="WMA Forecast")
ax.plot(future_dates, lr_preds,
        color="green", linewidth=2, linestyle="-.", marker="s", markersize=4,
        label=f"LR Forecast (MAE={mae})")
ax.axvline(last_date, color="gray", linestyle=":", linewidth=1, label="Forecast Start")
ax.set_xlabel("Date")
ax.set_ylabel("Demand (units)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)


st.subheader("Forecast Summary Table")

st.table({
    "Date":         [str(d) for d in future_dates],
    "WMA Forecast": wma_preds,
    "LR Forecast":  lr_preds,
})


col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Historical Demand", f"{round(sum(demands)/len(demands), 1)} units")
col2.metric("Avg Inventory",         f"{round(sum(inventories)/len(inventories), 1)} units")
col3.metric("Avg WMA Forecast",      f"{round(sum(wma_preds)/len(wma_preds), 1)} units")
col4.metric("Avg LR Forecast",       f"{round(sum(lr_preds)/len(lr_preds), 1)} units")