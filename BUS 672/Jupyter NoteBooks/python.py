#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Playbook — Good / Bad / Important (Context Edition)
NO plot titles or axis labels inside figures. Page titles only (above image).
- 2×2 layout (two top, one bottom spanning two columns)
- Figures shifted down 0.5" to avoid title overlap
- Notes typed below each image in the PDF
"""

import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import networkx as nx

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# -----------------------------
# Globals & Colors
# -----------------------------
np.random.seed(42)
OUT_DIR = Path("output_playbook")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_WIDTH = 1700
EXPORT_HEIGHT = 980

GOOD = "#2e7d32"     # green
BAD  = "#b71c1c"     # red
IMP  = "#f9a825"     # gold
BLUE = "#1e88e5"     # neutral accent
GRAY = "#9e9e9e"

COLOR_BY_FOCUS = {"Good": GOOD, "Bad": BAD, "Important": IMP}

# -----------------------------
# Helpers (export, PDF, layout)
# -----------------------------
def save_fig(fig: go.Figure, filename: str) -> str:
    # Remove any figure title; we show page titles in PDF instead
    fig.update_layout(
        width=EXPORT_WIDTH, height=EXPORT_HEIGHT,
        margin=dict(l=70, r=50, t=40, b=60),
        title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    # Hide all axis titles & tick labels globally
    fig.update_xaxes(showticklabels=False, title_text=None)
    fig.update_yaxes(showticklabels=False, title_text=None)

    path = str(OUT_DIR / filename)
    pio.write_image(fig, path, scale=2)
    return path

def wrap_lines(text: str, width_chars: int = 150):
    for raw in text.strip().split("\n"):
        raw = raw.rstrip()
        if not raw:
            yield ""
        else:
            for w in textwrap.wrap(raw, width=width_chars):
                yield w

def draw_page_with_notes(c: canvas.Canvas, png_path: str, page_title: str, notes_block: str):
    w, h = landscape(letter)
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(w/2, h - 0.6*inch, page_title)
    # Image
    img = ImageReader(png_path)
    img_w, img_h = img.getSize()
    bottom_reserved = 2.0 * inch
    max_w = w - 1.0*inch
    max_h = h - 2.0*inch - bottom_reserved
    scale = min(max_w / img_w, max_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    x = (w - draw_w) / 2
    y = (h - draw_h) / 2 + (bottom_reserved/2) - 0.5*inch  # shift down 0.5"
    c.drawImage(img, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
    # Notes
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.9*inch, 1.05*inch, "Notes")
    c.setFont("Helvetica", 11)
    y_text = 0.85 * inch
    for line in wrap_lines(notes_block, width_chars=160):
        if not line:
            y_text -= 0.12 * inch
        else:
            c.drawString(0.9*inch, y_text, line)
            y_text -= 0.15 * inch
    c.showPage()

def make_grid_3plots(use_domain_col2=False, specs_override=None):
    """
    2×2 grid; bottom spans two columns.
    - If use_domain_col2=True, (1,2) is 'domain' (for Sankey).
    - If specs_override is provided, it replaces default specs (e.g., spatial).
    - NO subplot titles (removed by request).
    """
    if specs_override is None:
        col2_type = "domain" if use_domain_col2 else "xy"
        specs = [
            [ {"type": "xy"}, {"type": col2_type} ],
            [ {"type": "xy", "colspan": 2}, None ]
        ]
    else:
        specs = specs_override

    fig = make_subplots(
        rows=2, cols=2, specs=specs,
        horizontal_spacing=0.08, vertical_spacing=0.14,
        subplot_titles=("", "", "")  # explicitly blank
    )
    # (No layout title; page title handles it)
    fig.update_layout(title=None)
    return fig

# Utility
def corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.std() == 0 or y.std() == 0: return 0.0
    return float(np.corrcoef(x, y)[0,1])

# -----------------------------
# SPOOFED DATA with contextual fields
# -----------------------------
# Categorical context: “Quarterly Revenue by Category” with a company target
categories = ["Alpha","Bravo","Charlie","Delta","Echo","Foxtrot"]
target_rev = 75  # revenue target
vals_good = np.random.randint(80, 120, size=len(categories))   # > target
vals_bad  = np.random.randint(30, 60,  size=len(categories))   # < target
vals_imp  = np.random.randint(60, 90,  size=len(categories))   # around target

g_2024_g = np.random.randint(70, 120, len(categories))
g_2025_i = g_2024_g + np.random.randint(-15, 25, len(categories))
g_under_b = np.maximum(0, g_2024_g - np.random.randint(10, 40, len(categories)))

# Pareto contributions
pareto_good = np.random.randint(5, 25, size=len(categories))
pareto_bad  = np.random.randint(10, 35, size=len(categories))
pareto_imp  = np.random.randint(8, 28, size=len(categories))

# Ordinal
edu_levels = ["High School","Associate","Bachelor","Master","Doctorate"]
edu_good = np.sort(np.random.randint(70, 95,  len(edu_levels)))
edu_bad  = np.sort(np.random.randint(40, 65,  len(edu_levels)))
edu_imp  = np.sort(np.random.randint(60, 80,  len(edu_levels)))

likert_levels = ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"]
lik_good = np.array([5, 8, 18, 45, 70]) + np.random.randint(-3, 3, 5)
lik_bad  = np.array([40, 52, 15, 7, 4]) + np.random.randint(-3, 3, 5)
lik_imp  = np.array([10, 18, 55, 20, 10]) + np.random.randint(-3, 3, 5)

rank_items = ["Team A","Team B","Team C","Team D","Team E","Team F"]
rank_t1 = np.random.permutation([1,2,3,4,5,6])
rank_t2 = rank_t1.copy()
rank_t2[[0,1]] = np.maximum(1, rank_t1[[0,1]] - 1)  # improve (Good)
rank_t2[[2]]   = np.minimum(6, rank_t1[[2]] + 1)    # worsen (Bad)

# Quantitative
n_points = 360
x_g = np.random.normal(55, 12, size=n_points//3)  # KPI
x_b = np.random.normal(45, 15, size=n_points//3)
x_i = np.random.normal(50, 10, size=n_points//3)

y_g = 0.7*x_g + 15 + np.random.normal(0, 7, size=x_g.size)   # Cost
y_b = 0.2*x_b +  5 + np.random.normal(0,12, size=x_b.size)
y_i = 0.5*x_i + 10 + np.random.normal(0, 9, size=x_i.size)

box_g = np.random.normal(70, 8, 120)
box_b = np.random.normal(45,10, 120)
box_i = np.random.normal(55, 9, 120)

# Temporal
dates = pd.date_range("2024-01-01", periods=24, freq="MS")
s_good = np.cumsum(np.random.randint(40, 90, size=len(dates))) + 150
s_bad  = np.cumsum(np.random.randint(10, 40, size=len(dates))) + 60
s_imp  = np.cumsum(np.random.randint(25, 60, size=len(dates))) + 100

a_good = (np.sin(np.linspace(0, 3*np.pi, len(dates))) * 100 + 500).astype(int)
a_bad  = (np.sin(np.linspace(0, 3*np.pi, len(dates))) *  40 + 200).astype(int)
a_imp  = (np.sin(np.linspace(0, 3*np.pi, len(dates))) *  70 + 350).astype(int)

m_good = s_good + np.random.randint(-15, 15, len(dates))
m_bad  = s_bad  + np.random.randint(-10, 10, len(dates))
m_imp  = s_imp  + np.random.randint(-12, 12, len(dates))

events = pd.to_datetime(["2024-07-01","2025-01-01"])
event_labels = ["Promo Launch","Pricing Update"]

# Relational
G = nx.barabasi_albert_graph(20, 2, seed=7)
layout_pos = nx.spring_layout(G, seed=7, k=0.75)
bet = nx.betweenness_centrality(G)
degrees = dict(G.degree())
deg_vals = np.array([degrees[n] for n in G.nodes()])
th_hi = np.percentile(deg_vals, 75)
th_lo = np.percentile(deg_vals, 25)
def node_class(n):
    d = degrees[n]
    if d >= th_hi: return "Important"
    if d <= th_lo: return "Bad"
    return "Good"

labels = ["Source G","Source B","Source I","Stage 1","Stage 2","Output"]
source = [0,1,2,3,3,4]
target = [3,3,3,4,5,5]
value  = [8,4,6,12,10,6]

# Spatial with rates
states = ["CA","TX","FL","NY","IL","PA","OH","GA","NC","MI","WA","AZ","MA","TN","IN"]
population = np.random.randint(1_000_000, 30_000_000, size=len(states))
sales_state = np.random.randint(200, 2000, size=len(states))
rate_per_100k = (sales_state / population) * 100_000
q33, q66 = np.quantile(rate_per_100k, [0.33, 0.66])
state_bins = ["Good" if r >= q66 else "Bad" if r <= q33 else "Important" for r in rate_per_100k]

num_locs = 30
lats = np.random.uniform(25, 49, size=num_locs)
lons = np.random.uniform(-124, -66, size=num_locs)
sizes = np.random.randint(10, 200, size=num_locs)
cat_sym = np.random.choice(["Good","Bad","Important"], size=num_locs, p=[0.4,0.3,0.3])

grid_x, grid_y = np.meshgrid(np.arange(0, 30), np.arange(0, 20))
heat = (
    50
    + 30*np.exp(-((grid_x-10)**2 + (grid_y-6)**2)/50)
    + 20*np.exp(-((grid_x-22)**2 + (grid_y-14)**2)/30)
    + np.random.normal(0, 2, size=grid_x.shape)
)
lvl_good = np.percentile(heat, 85)
lvl_imp  = np.percentile(heat, 60)
lvl_bad  = np.percentile(heat, 35)

# -----------------------------
# BUILDERS (Good / Bad / Important pages)
# -----------------------------
def build_categorical(mode: str) -> go.Figure:
    color = COLOR_BY_FOCUS[mode]
    fig = make_grid_3plots()

    # A) Sorted bars with target line
    ymap = {"Good": vals_good, "Bad": vals_bad, "Important": vals_imp}
    vals = ymap[mode]
    order = np.argsort(vals)[::-1]
    xs = np.array(categories)[order]; ys = vals[order]
    fig.add_trace(go.Bar(
        x=xs, y=ys, marker_color=color,
        hovertemplate="%{x}<br>Value: %{y}<extra></extra>", name=mode
    ), row=1, col=1)
    fig.add_hline(y=target_rev, line_dash="dot", line_color=BLUE, row=1, col=1)
    fig.add_annotation(xref="x1", yref="y1", x=xs[-1], y=target_rev,
                       text=f"Target={target_rev}", showarrow=False, font=dict(size=11), bgcolor="white")

    # B) Grouped “spotlight” + target zones
    ymap_g = {"Good": g_2024_g, "Bad": g_under_b, "Important": g_2025_i}
    fig.add_trace(go.Bar(x=categories, y=ymap_g[mode], marker_color=color,
                         hovertemplate="%{x}<br>Value: %{y}<extra></extra>", name=mode),
                  row=1, col=2)
    fig.add_hrect(y0=target_rev, y1=target_rev*1.3, line_width=0, fillcolor="rgba(46,125,50,0.08)",
                  row=1, col=2)
    fig.add_hrect(y0=0, y1=target_rev*0.85, line_width=0, fillcolor="rgba(183,28,28,0.08)",
                  row=1, col=2)
    fig.add_hline(y=target_rev, line_dash="dot", line_color=BLUE, row=1, col=2)

    # C) Pareto chart (bars + cumulative % line)
    pareto_map = {"Good": pareto_good, "Bad": pareto_bad, "Important": pareto_imp}
    vals_p = pareto_map[mode]
    order_p = np.argsort(vals_p)[::-1]
    xs_p = np.array(categories)[order_p]; ys_p = vals_p[order_p]
    cum = np.cumsum(ys_p) / ys_p.sum() * 100
    fig.add_trace(go.Bar(x=xs_p, y=ys_p, marker_color=color, name="Count",
                         hovertemplate="%{x}<br>Count: %{y}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=xs_p, y=cum, yaxis="y3", mode="lines+markers",
                             name="Cumulative %", line=dict(width=3, color=BLUE),
                             hovertemplate="%{x}<br>Cum: %{y:.1f}%<extra></extra>"),
                  row=2, col=1)

    # Strip axis labels/titles handled globally in save_fig
    return fig

def build_ordinal(mode: str) -> go.Figure:
    color = COLOR_BY_FOCUS[mode]
    fig = make_grid_3plots()

    # A) Ordered bars (natural order, add median line)
    ymap = {"Good": edu_good, "Bad": edu_bad, "Important": edu_imp}
    vals = ymap[mode]
    fig.add_trace(
        go.Bar(
            x=edu_levels, y=vals, marker_color=color,
            hovertemplate="%{x}<br>Score: %{y}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_hline(y=float(np.median(vals)), line_dash="dot", line_color=BLUE, row=1, col=1)

    # B) Diverging Likert (neg left, pos right)
    lik_map = {"Good": lik_good, "Bad": lik_bad, "Important": lik_imp}
    counts = lik_map[mode]
    neg = counts[0] + counts[1]; neu = counts[2]; pos = counts[3] + counts[4]
    fig.add_trace(
        go.Bar(x=[f"{mode} Q"], y=[-neg], marker_color=color,
               hovertemplate="Disagree: %{y}<extra></extra>", showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=[f"{mode} Q"], y=[ neu], marker_color=GRAY,
               hovertemplate="Neutral: %{y}<extra></extra>", showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=[f"{mode} Q"], y=[ pos], marker_color=color, opacity=0.8,
               hovertemplate="Agree: %{y}<extra></extra>", showlegend=False),
        row=1, col=2
    )
    fig.update_layout(barmode="relative")

    # C) Slope with Δ labels and correct hover (customdata)
    for i, item in enumerate(rank_items):
        improved = rank_t2[i] < rank_t1[i]
        worsened = rank_t2[i] > rank_t1[i]
        cls = "Good" if improved else "Bad" if worsened else "Important"
        if cls == mode:
            dy = rank_t1[i] - rank_t2[i]
            arrow = "▲" if dy > 0 else "▼" if dy < 0 else "•"
            t1, t2 = rank_t1[i], rank_t2[i]
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[t1, t2],
                    mode="lines+markers+text",
                    text=[f"{item}", f"{item} ({arrow}{abs(dy)})"],
                    textposition="top center",
                    line=dict(width=3, color=color),
                    marker=dict(size=8, color=color),
                    customdata=[[t1, t2], [t1, t2]],
                    hovertemplate=f"{item}<br>T1: %{{customdata[0]}} → T2: %{{customdata[1]}}<extra></extra>",
                    showlegend=False
                ),
                row=2, col=1
            )
    # No axis labels/titles by global save_fig
    return fig

def build_quantitative(mode: str) -> go.Figure:
    color = COLOR_BY_FOCUS[mode]
    fig = make_grid_3plots()

    # Select data
    if mode == "Good":
        X, Y, Yname = x_g, y_g, "Cost"
    elif mode == "Bad":
        X, Y, Yname = x_b, y_b, "Cost"
    else:
        X, Y, Yname = x_i, y_i, "Cost"

    # A) Scatter with “good region” (KPI high, cost low)
    thresh_x = np.percentile(X, 60)
    thresh_y = np.percentile(Y, 40)
    fig.add_shape(type="rect", xref="x1", yref="y1",
                  x0=thresh_x, y0=min(Y)-5, x1=max(X)+5, y1=thresh_y,
                  fillcolor="rgba(46,125,50,0.10)", line_width=0, row=1, col=1)
    r = corr(X, Y)
    fig.add_trace(go.Scatter(
        x=X, y=Y, mode="markers", marker=dict(size=6, opacity=0.8, color=color),
        hovertemplate=f"KPI: %{{x:.1f}}<br>{Yname}: %{{y:.1f}}<extra></extra>",
        name=mode
    ), row=1, col=1)
    m, b0 = np.polyfit(X, Y, 1)
    xline = np.linspace(X.min(), X.max(), 80)
    fig.add_trace(go.Scatter(x=xline, y=m*xline+b0, mode="lines",
                             name=f"Trend r={r:.2f}", line=dict(width=3, color=BLUE)),
                  row=1, col=1)

    # B) Histogram with adaptive bins + 90th pct
    nb = max(12, int(np.sqrt(X.size)))
    p90 = np.percentile(X, 90)
    fig.add_trace(go.Histogram(x=X, nbinsx=nb, marker_color=color, opacity=0.8,
                               hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>"),
                  row=1, col=2)
    fig.add_vline(x=p90, line_dash="dot", line_color=BLUE, row=1, col=2)
    fig.add_annotation(xref="x2", yref="paper", x=p90, y=0.98,
                       text="90th pct", showarrow=False, font=dict(size=11), bgcolor="white")

    # C) Boxplot + outlier count
    bymap = {"Good": box_g, "Bad": box_b, "Important": box_i}
    data = bymap[mode]
    q1, q3 = np.percentile(data, [25, 75]); iqr = q3-q1
    lo = q1 - 1.5*iqr; hi = q3 + 1.5*iqr
    outliers = ((data < lo) | (data > hi)).sum()
    fig.add_trace(go.Box(y=data, name=f"{mode} (outliers={outliers})",
                         marker_color=color, boxmean=True, hovertemplate="Value: %{y}<extra></extra>"),
                  row=2, col=1)
    return fig

def build_temporal(mode: str) -> go.Figure:
    color = COLOR_BY_FOCUS[mode]
    fig = make_grid_3plots()

    smap = {"Good": s_good, "Bad": s_bad, "Important": s_imp}
    amap = {"Good": a_good, "Bad": a_bad, "Important": a_imp}
    mmap = {"Good": m_good, "Bad": m_bad, "Important": m_imp}

    # A) Line + moving average
    s = pd.Series(smap[mode], index=dates)
    ma = s.rolling(3, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=dates, y=s, mode="lines+markers", name=mode,
                             line=dict(width=2, color=color),
                             hovertemplate="%{x|%b %Y}<br>Value: %{y}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma, mode="lines", name="3-mo MA",
                             line=dict(width=3, color=BLUE, dash="dot")),
                  row=1, col=1)

    # B) Seasonal area
    a = amap[mode]
    fig.add_trace(go.Scatter(x=dates, y=a, fill="tozeroy", mode="lines",
                             name=f"{mode} Area", line=dict(color=color)),
                  row=1, col=2)

    # C) YoY % change + events
    m = pd.Series(mmap[mode], index=dates)
    yoy = m.pct_change(12) * 100
    fig.add_trace(go.Scatter(x=dates, y=yoy, mode="lines+markers",
                             name="YoY %", line=dict(color=color, width=3),
                             hovertemplate="%{x|%b %Y}<br>YoY: %{y:.1f}%<extra></extra>"),
                  row=2, col=1)
    for dt, label in zip(events, event_labels):
        fig.add_vline(x=dt, line_dash="dash", line_color=BLUE, row=2, col=1)
        fig.add_annotation(x=dt, yref="y3", y=yoy.max() if np.isfinite(yoy.max()) else 0,
                           text=label, showarrow=True, arrowhead=2, yshift=20,
                           bgcolor="white", font=dict(size=10))
    return fig

def build_relational(mode: str) -> go.Figure:
    color = COLOR_BY_FOCUS[mode]
    fig = make_grid_3plots(use_domain_col2=True)

    def is_focus(n): return node_class(n) == mode

    # A) Network — only focus nodes & edges; node size by betweenness
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if is_focus(u) and is_focus(v):
            x0,y0 = layout_pos[u]; x1,y1 = layout_pos[v]
            edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    if edge_x:
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=1, color="#cfd8dc"),
                                 hoverinfo="skip", showlegend=False),
                      row=1, col=1)
    focus_nodes = [n for n in G.nodes() if is_focus(n)]
    if focus_nodes:
        bvals = np.array([bet[n] for n in focus_nodes])
        if bvals.max() > 0:
            bnorm = (bvals - bvals.min())/(bvals.max()-bvals.min()+1e-6)
        else:
            bnorm = np.zeros_like(bvals)
        for n, b in zip(focus_nodes, bnorm):
            x,y = layout_pos[n]
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                text=[f"N{n}"], textposition="top center",
                marker=dict(size=10 + 20*b, color=color, line=dict(width=1, color="#333")),
                hovertext=f"Node {n}<br>deg={degrees[n]}<br>bet={bet[n]:.3f}",
                hoverinfo="text", showlegend=False
            ), row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # B) Sankey — spotlight; show % in hover
    total = sum(value)
    fig.add_trace(go.Sankey(
        node=dict(label=labels, color=[color]*len(labels)),
        link=dict(
            source=source, target=target, value=value,
            color=[color]*len(value),
            hovertemplate="Flow: %{value} (%{customdata:.1f}%)<extra></extra>",
            customdata=np.array(value)/total*100
        )
    ), row=1, col=2)

    # C) Hubs by betweenness (focus subset only)
    edge_x2, edge_y2 = [], []
    for u, v in G.edges():
        if is_focus(u) and is_focus(v):
            x0,y0 = layout_pos[u]; x1,y1 = layout_pos[v]
            edge_x2 += [x0,x1,None]; edge_y2 += [y0,y1,None]
    if edge_x2:
        fig.add_trace(go.Scatter(x=edge_x2, y=edge_y2, mode="lines",
                                 line=dict(width=1, color="#b0bec5"),
                                 hoverinfo="skip", showlegend=False), row=2, col=1)
    if focus_nodes:
        focus_sorted = sorted(focus_nodes, key=lambda n: bet[n], reverse=True)[:3]
        max_bet = max(bet.values()) if bet else 0
        for n in focus_nodes:
            x,y = layout_pos[n]
            size = 12 + 24*(bet[n]/(max_bet+1e-6)) if max_bet > 0 else 12
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                text=[f"N{n}" + (" ★" if n in focus_sorted else "")], textposition="top center",
                marker=dict(size=size, color=color, line=dict(width=1, color="#333")),
                hovertext=f"Node {n}<br>deg={degrees[n]}<br>bet={bet[n]:.3f}",
                hoverinfo="text", showlegend=False
            ), row=2, col=1)
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    return fig

def build_spatial(mode: str) -> go.Figure:
    color = COLOR_BY_FOCUS[mode]
    fig = make_grid_3plots(
        specs_override=[
            [ {"type": "choropleth"}, {"type": "geo"} ],
            [ {"type": "heatmap", "colspan": 2}, None ]
        ]
    )

    # A) Choropleth by rate (focus states only)
    mask = np.array([b == mode for b in state_bins])
    if mask.any():
        fig.add_trace(go.Choropleth(
            locations=np.array(states)[mask],
            z=np.array(rate_per_100k)[mask],
            locationmode="USA-states",
            colorscale=[[0, color],[1, color]],
            zmin=rate_per_100k.min(), zmax=rate_per_100k.max(),
            showscale=False,
            hovertemplate="%{location}<br>Rate: %{z:.1f}/100k<extra></extra>",
            name=mode
        ), row=1, col=1)
    fig.update_geos(scope="usa", row=1, col=1)

    # B) Proportional symbols — only focus locations
    m = (cat_sym == mode)
    fig.add_trace(go.Scattergeo(
        lon=lons[m], lat=lats[m],
        marker=dict(
            size=(sizes[m]/6).clip(4, 22), opacity=0.8, line=dict(width=1, color="#333"),
            color=color
        ),
        hovertemplate="Lon: %{lon:.2f}, Lat: %{lat:.2f}<br>Volume~%{marker.size:.1f}<extra></extra>",
        mode="markers", name=mode
    ), row=1, col=2)
    fig.update_geos(scope="usa", row=1, col=2)

    # C) Heatmap + focus threshold contour
    thr = {"Good": lvl_good, "Important": lvl_imp, "Bad": lvl_bad}[mode]
    fig.add_trace(go.Heatmap(z=heat, colorscale="Viridis", colorbar_title="Density",
                             hovertemplate="(%{x}, %{y}) → %{z:.1f}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Contour(z=heat, contours=dict(value=thr, showlabels=True),
                             showscale=False, line=dict(color=color, width=3),
                             name=f"{mode} threshold"),
                  row=2, col=1)
    return fig

# -----------------------------
# NOTES (contextual)
# -----------------------------
BASE_NOTES = {
    "Categorical": (
        "Best: bar charts; grouped bars; stacked/Pareto for contribution\n"
        "Use cases: category comparisons; spotting 80/20 drivers vs target\n"
        "Pitfalls: too many categories; truncated axes; color overload"
    ),
    "Ordinal": (
        "Best: ordered bars; diverging Likert; slope for two-time comparisons\n"
        "Use cases: survey scales; natural-order rankings; change over two points\n"
        "Pitfalls: confirm order; avoid clutter; label clearly"
    ),
    "Quantitative": (
        "Best: scatter (add fit & good region); histograms (adaptive bins); boxplots (outliers)\n"
        "Use cases: correlation; distribution shape; outlier detection\n"
        "Pitfalls: wrong bins; overplotting; unlabeled axes"
    ),
    "Temporal": (
        "Best: lines with moving averages; YoY% for comparability; event markers; seasonal areas\n"
        "Use cases: trends; seasonality; multi-series comparatives\n"
        "Pitfalls: bar charts for continuous series; uneven intervals"
    ),
    "Relational": (
        "Best: node-link with centrality; Sankey for flows (add %);\n"
        "Use cases: social/flow networks; process bottlenecks; key hubs\n"
        "Pitfalls: 'hairball' networks — filter/cluster and highlight"
    ),
    "Spatial": (
        "Best: rates (per 100k) choropleths; symbols for volume; heatmaps for density\n"
        "Use cases: geographic rate comparisons; density hot spots; location patterns\n"
        "Pitfalls: misleading projections; rainbow scales; no normalization"
    ),
}

def focus_preface(mode: str) -> str:
    return (f"This page spotlights {mode} patterns only—other classes are omitted. "
            f"Context added: targets, rates, moving averages, centrality, Pareto, thresholds. "
            f"Axis titles & tick labels are intentionally removed inside figures.")

# -----------------------------
# BUILD ALL PAGES (6 × 3)
# -----------------------------
builders = [
    ("Categorical", build_categorical),
    ("Ordinal", build_ordinal),
    ("Quantitative", build_quantitative),
    ("Temporal", build_temporal),
    ("Relational", build_relational),
    ("Spatial", build_spatial),
]
focuses = ["Good", "Bad", "Important"]

exported = []  # (title, png_path, notes)

for cat_name, builder in builders:
    for mode in focuses:
        fig = builder(mode)
        fname = f"{cat_name.lower()}_{mode.lower()}_context_noaxes.png"
        png_path = save_fig(fig, fname)
        title = f"{cat_name} — {mode} Focus"
        notes = focus_preface(mode) + "\n" + BASE_NOTES[cat_name]
        exported.append((title, png_path, notes))

# -----------------------------
# PDF Assembly
# -----------------------------
pdf_path = OUT_DIR / "Visualization_Playbook_Focused_Views_NoAxes.pdf"
c = canvas.Canvas(str(pdf_path), pagesize=landscape(letter))

# Cover
w, h = landscape(letter)
c.setFont("Helvetica-Bold", 22)
c.drawCentredString(w/2, h - 0.8*inch, "Visualization Playbook — Good / Bad / Important (No-Axes Edition)")
c.setFont("Helvetica", 11)
cover = (
    "All plot titles and x/y labels inside figures have been removed for a cleaner teaching handout.\n"
    "Context is conveyed via page titles, visual encodings, and the notes below each figure."
)
y = h - 1.2*inch
for line in wrap_lines(cover, 160):
    c.drawString(0.9*inch, y, line)
    y -= 0.18*inch

# Legend
c.setFont("Helvetica-Bold", 12)
c.drawString(0.9*inch, 0.95*inch, "Legend:")
c.setFillColorRGB(0.25, 0.49, 0.38); c.rect(1.65*inch, 0.93*inch, 0.35*inch, 0.16*inch, fill=1, stroke=0)
c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 11); c.drawString(2.05*inch, 0.96*inch, "Good")
c.setFillColorRGB(0.72, 0.20, 0.20); c.rect(2.9*inch, 0.93*inch, 0.35*inch, 0.16*inch, fill=1, stroke=0)
c.setFillColorRGB(0, 0, 0); c.drawString(3.3*inch, 0.96*inch, "Bad")
c.setFillColorRGB(0.98, 0.76, 0.15); c.rect(4.15*inch, 0.93*inch, 0.35*inch, 0.16*inch, fill=1, stroke=0)
c.setFillColorRGB(0, 0, 0); c.drawString(4.55*inch, 0.96*inch, "Important")
c.showPage()

# Pages
for title, png_path, notes in exported:
    draw_page_with_notes(c, png_path, title, notes)

c.save()

print("Done. Exported:")
print(" ", pdf_path)
for _, p, _ in exported:
    print(" ", p)