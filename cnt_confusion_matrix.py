"""
CNT MLR Confusion Matrix Dashboard
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression

BASE         = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV = os.path.join(BASE, "training_data.csv")
NEW_ENTRIES  = os.path.join(BASE, "new_entries.csv")

FEAT_COLS = [
    'V1_MaxPulse_Voltage_V', 'Number_of_Pulses', 'Applied_energy_J',
    'Average_Power_P', 'Pth_Vth_times_Ith', 'Ith_Trigger_Current',
    'Vth_TriggerVoltage', 'N_scale', 'ton', 'toff', 'T_total_period',
    'n_P_over_Pth', 'Rb_Resistance_Before',
]
CLASS_LABELS = ['≥1000%', '40–999%', '5–39%', '0–4%', '-5–-1%', '-20–-6%', '<-20%']

def _ycat(Y):
    return np.select(
        [Y >= 1000, Y >= 40, Y >= 5, Y >= 0, Y >= -5, Y >= -20],
        [1, 2, 3, 4, 5, 6], default=7,
    )

def get_new_entries():
    if not os.path.exists(NEW_ENTRIES):
        return pd.DataFrame()
    try:
        return pd.read_csv(NEW_ENTRIES)
    except Exception:
        return pd.DataFrame()

def _build_cm(X, Ycat, normalise=True):
    if normalise:
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0, ddof=1)
        X_fit  = (X - X_mean) / X_std
    else:
        X_fit  = X
    clf = LogisticRegression(solver='lbfgs', max_iter=10000, C=1e9)
    clf.fit(X_fit, Ycat)
    pred = clf.predict(X_fit)
    K  = 7
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(Ycat, pred):
        cm[t - 1, p - 1] += 1
    return cm, (pred == Ycat).mean()

@st.cache_data(ttl=60)
def load_confusion_matrix(selected_rowids=None):
    if not os.path.exists(TRAINING_CSV):
        return None, None, None, None, None, ['training_data.csv']

    data = pd.read_csv(TRAINING_CSV)

    if selected_rowids:
        new_df = get_new_entries()
        if not new_df.empty and 'rowid' in new_df.columns:
            subset = new_df[new_df['rowid'].isin(selected_rowids)]
            shared = [c for c in FEAT_COLS + ['Ra_minus_Rb_over_Rb_percent']
                      if c in subset.columns]
            if shared:
                data = pd.concat([data, subset[shared]], ignore_index=True)

    data  = data.dropna(subset=FEAT_COLS + ['Ra_minus_Rb_over_Rb_percent'])
    X     = data[FEAT_COLS].values.astype(float)
    Y     = data['Ra_minus_Rb_over_Rb_percent'].values
    Ycat  = _ycat(Y)
    n     = len(Y)

    cm_norm, acc_norm   = _build_cm(X, Ycat, normalise=True)
    cm_raw,  acc_raw    = _build_cm(X, Ycat, normalise=False)

    return cm_norm, acc_norm, cm_raw, acc_raw, n, []

def confusion_matrix_fig(cm, acc, n, title_suffix=""):
    K       = cm.shape[0]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    text    = [[f"{cm[i,j]}<br>{cm_norm[i,j]*100:.0f}%" for j in range(K)] for i in range(K)]
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=CLASS_LABELS, y=CLASS_LABELS,
        text=text, texttemplate="%{text}",
        colorscale='Blues', showscale=True, zmin=0, zmax=1,
    ))
    fig.update_layout(
        title=dict(
            text=f"MLR Confusion Matrix  |  Accuracy {acc*100:.1f}%  |  n = {n}  |  {title_suffix}",
            font=dict(size=12, family="Times New Roman, serif"),
        ),
        xaxis=dict(title="Predicted class", side="bottom",
                   tickfont=dict(size=10, family="Times New Roman, serif")),
        yaxis=dict(title="Actual class", autorange="reversed",
                   tickfont=dict(size=10, family="Times New Roman, serif")),
        font=dict(family="Times New Roman, serif", size=10),
        margin=dict(l=110, r=30, t=70, b=100),
        height=500, plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CNT Confusion Matrix", layout="centered")
st.title("CNT Pulse Outcome Prediction")
st.divider()

# ── New entry selector ────────────────────────────────────────────────────────
selected_rowids = []
new_df = get_new_entries()

if not new_df.empty:
    st.subheader("Include new experiments")
    st.caption("Tick entries to add them to the confusion matrix.")
    for _, row in new_df.iterrows():
        label = (f"{row.get('Chip and Device', 'Unknown')} | "
                 f"V1={row.get('V1_MaxPulse_Voltage_V','?')}  "
                 f"N={int(row.get('Number_of_Pulses', 0))}  "
                 f"(row {int(row['rowid'])})")
        if st.checkbox(label, key=f"row_{int(row['rowid'])}"):
            selected_rowids.append(int(row['rowid']))
    st.divider()

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm_norm, acc_norm, cm_raw, acc_raw, n, missing = load_confusion_matrix(
    selected_rowids=tuple(selected_rowids) if selected_rowids else None
)

if missing:
    st.error(f"Missing files: {', '.join(missing)}")
elif cm_norm is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Normalised X** (X − mean) / std")
        st.plotly_chart(confusion_matrix_fig(cm_norm, acc_norm, n, "Normalised X"),
                        use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.markdown("**Raw X** (no normalisation)")
        st.plotly_chart(confusion_matrix_fig(cm_raw, acc_raw, n, "Raw X"),
                        use_container_width=True, config={"displayModeBar": False})
else:
    st.warning("Could not load model data.")

st.divider()
st.caption(
    "Class = ΔR/Rb (%) change in CNT resistance after electromigration pulsing.  "
    "Model trained using MATLAB mnrfit, replicated in Python."
)
