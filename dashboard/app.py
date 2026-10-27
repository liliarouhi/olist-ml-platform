"""Olist sales and delivery dashboard. Run: streamlit run dashboard/app.py"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from olist_ml.db.warehouse import Filters, WarehouseRepository

st.set_page_config(page_title="Olist Dashboard", page_icon="📦", layout="wide")

MIN_DATE, MAX_DATE = date(2017, 1, 1), date(2018, 8, 31)  # valid analysis window (docs/03)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SCORE_COLORS = {"1": "#d73027", "2": "#fc8d59", "3": "#fee08b", "4": "#91cf60", "5": "#1a9850"}
repo = WarehouseRepository()


@st.cache_data(ttl=3600)
def load_options() -> tuple[list[str], list[str]]:
    return repo.options()


@st.cache_data(ttl=600)
def load(method: str, start: date, end: date, states: tuple, categories: tuple,
         previous: bool = False) -> pd.DataFrame:
    """Cached call of a repository method; identical filter combinations hit the cache."""
    filters = Filters(start, end, states, categories)
    if previous:
        filters = filters.previous_period()
    return getattr(repo, method)(filters)


# ---------- Sidebar filters ----------
all_states, all_categories = load_options()
with st.sidebar:
    st.header("Filters")
    period = st.date_input("Purchase date", (MIN_DATE, MAX_DATE), min_value=MIN_DATE, max_value=MAX_DATE)
    states = tuple(st.multiselect("Customer state", all_states))
    categories = tuple(st.multiselect("Product category", all_categories))
    st.caption("Empty selection means all. Delivered orders only.")

if not isinstance(period, tuple) or len(period) != 2:
    st.info("Select both a start and an end date.")
    st.stop()
start, end = period
args = (start, end, states, categories)

# ---------- KPI cards with period-over-period change ----------
current = load("kpis", *args).iloc[0]
previous = load("kpis", *args, previous=True).iloc[0]

st.title("📦 Olist: Sales & Delivery")
has_comparison = Filters(start, end).previous_period().start >= MIN_DATE
comparison_note = ("compared with the previous period of equal length" if has_comparison
                   else "select a shorter period to compare with the previous one")
st.caption(f"{start:%d %b %Y} – {end:%d %b %Y} · {comparison_note}")
if current["orders"] == 0:
    st.warning("No orders match these filters.")
    st.stop()


def change(key: str, relative: bool) -> str | None:
    
    if not has_comparison:
        return None
    
    now, before = current[key], previous[key]
    if pd.isna(now) or pd.isna(before) or (relative and before == 0):
        return None
    return f"{(now - before) / before:+.1%}" if relative else f"{now - before:+.2f}"


cards = st.columns(6)
cards[0].metric("Revenue (BRL)", f"{current['revenue']:,.0f}", change("revenue", True))
cards[1].metric("Orders", f"{int(current['orders']):,}", change("orders", True))
cards[2].metric("Avg order value", f"{current['avg_order_value']:,.2f}", change("avg_order_value", True))
cards[3].metric("Late deliveries", f"{current['late_pct']:.1f}%", change("late_pct", False), delta_color="inverse")
cards[4].metric("Avg review", f"{current['avg_review']:.2f} / 5", change("avg_review", False))
cards[5].metric("Avg delivery (days)", f"{current['avg_delivery_days']:.1f}",
                change("avg_delivery_days", False), delta_color="inverse")

sales_tab, delivery_tab, category_tab = st.tabs(["📈 Sales", "🚚 Delivery & satisfaction", "🏷️ Categories"])

# ---------- Sales ----------
with sales_tab:
    monthly = load("monthly", *args)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=monthly["month_start"], y=monthly["orders"], name="Orders", opacity=0.35, secondary_y=True)
    fig.add_scatter(x=monthly["month_start"], y=monthly["revenue"], name="Revenue (BRL)",
                    mode="lines+markers", secondary_y=False)
    fig.update_layout(title="Monthly revenue and orders", hovermode="x unified",
                      legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="Revenue (BRL)", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True, showgrid=False)
    st.plotly_chart(fig)

    left, right = st.columns(2)
    weekday = load("weekday", *args)
    weekday["day"] = weekday["day_of_week"].map(lambda d: WEEKDAYS[int(d) - 1])
    left.plotly_chart(px.bar(weekday, x="day", y="orders", title="Orders by weekday"))
    right.plotly_chart(px.line(monthly, x="month_start", y="late_pct", markers=True,
                               title="Late deliveries per month (%)",
                               labels={"month_start": "Month", "late_pct": "Late (%)"}))

# ---------- Delivery & satisfaction ----------
with delivery_tab:
    by_state = load("states", *args)
    fig = px.scatter(by_state, x="avg_delivery_days", y="late_pct", size="orders", color="avg_review",
                     text="state", color_continuous_scale="RdYlGn", hover_data={"revenue": ":,.0f"},
                     title="Delivery time vs late rate per state (size = orders, colour = avg review)",
                     labels={"avg_delivery_days": "Avg delivery (days)", "late_pct": "Late (%)",
                             "avg_review": "Avg review"})
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig)

    left, right = st.columns(2)
    by_days = load("review_by_delivery", *args)
    left.plotly_chart(px.line(by_days, x="delivery_days", y="avg_review", markers=True,
                              title="Average review by delivery time (40 = 40+ days)",
                              labels={"delivery_days": "Delivery (days)", "avg_review": "Avg review"}))

    dist = load("review_distribution", *args)
    dist["share"] = dist["orders"] / dist.groupby("delivery")["orders"].transform("sum")
    dist["score"] = dist["score"].astype(str)
    fig = px.bar(dist, x="delivery", y="share", color="score", color_discrete_map=SCORE_COLORS,
                 title="Review score distribution: late vs on-time orders",
                 category_orders={"score": ["1", "2", "3", "4", "5"]},
                 labels={"share": "Share of orders", "delivery": ""})
    fig.update_yaxes(tickformat=".0%")
    right.plotly_chart(fig)

# ---------- Categories ----------
with category_tab:
    top = load("top_categories", *args)
    st.plotly_chart(px.treemap(top, path=["category"], values="revenue", color="orders",
                               color_continuous_scale="Blues",
                               title="Revenue by category (top 15, colour = orders)"))

    heat = load("category_month", *args)
    pivot = heat.pivot(index="category", columns="month_start", values="revenue").fillna(0)
    st.plotly_chart(px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis",
                              labels={"color": "Revenue (BRL)", "x": "Month", "y": ""},
                              title="Monthly revenue of the top 8 categories"))

st.caption("Source: PostgreSQL star schema `dw` · queries cached for 10 minutes")