"""
CNT MLR Confusion Matrix Dashboard
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE         = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE, "mlr_weights.csv")
NORM_PATH    = os.path.join(BASE, "mlr_norm_params.csv")
TRAINING_CSV = os.path.join(BASE, "training_data.csv")
NEW_ENTRIES  = os.path.join(BASE, "new_entries.csv")

FEAT_COLS = [
    'V1_MaxPulse_Voltage_V', 'Number_of_Pulses', 'Applied_energy_J',
    'Average_Power_P', 'Pth_Vth_times_Ith', 'Ith_Trigger_Current',
    'Vth_TriggerVoltage', 'N_scale', 'ton', 'toff', 'T_total_period',
    'n_P_over_Pth', 'Rb_Resistance_Before',
]
CLASS_LABELS = ['≥1000%', '40–999%', '5–39%', '0–4%', '-5–-1%', '-20–-6%', '<-20%']

def mnrval_py(wm, X_norm):
    X_aug  = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
    scores = X_aug @ wm
    exp_s  = np.exp(np.clip(scores, -500, 500))
    denom  = 1 + exp_s.sum(axis=1, keepdims=True)
    return np.hstack([exp_s / denom, 1 / denom])

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

@st.cache_data(ttl=60)
def load_confusion_matrix(selected_rowids=None):
    missing = [p for p in [WEIGHTS_PATH, NORM_PATH, TRAINING_CSV] if not os.path.exists(p)]
    if missing:
        return None, None, None, [os.path.basename(p) for p in missing]

    wm_df   = pd.read_csv(WEIGHTS_PATH, index_col=0)
    norm_df = pd.read_csv(NORM_PATH,    index_col=0)
    wm      = wm_df.values.astype(float)
    X_mean  = norm_df.loc['mean', FEAT_COLS].values.astype(float)
    X_std   = norm_df.loc['std',  FEAT_COLS].values.astype(float)

    data = pd.read_csv(TRAINING_CSV)

    if selected_rowids:
        new_df = get_new_entries()
        if not new_df.empty and 'rowid' in new_df.columns:
            subset = new_df[new_df['rowid'].isin(selected_rowids)]
            shared = [c for c in FEAT_COLS + ['Ra_minus_Rb_over_Rb_percent']
                      if c in subset.columns]
            if shared:
                data = pd.concat([data, subset[shared]], ignore_index=True)

    data   = data.dropna(subset=FEAT_COLS + ['Ra_minus_Rb_over_Rb_percent'])
    X      = data[FEAT_COLS].values.astype(float)
    Y      = data['Ra_minus_Rb_over_Rb_percent'].values
    Ycat   = _ycat(Y)
    X_norm = (X - X_mean) / X_std
    pm     = mnrval_py(wm, X_norm)
    pred   = pm.argmax(axis=1) + 1

    K  = 7
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(Ycat, pred):
        cm[t - 1, p - 1] += 1
    acc = (pred == Ycat).mean()
    return cm, acc, len(Y), []

def confusion_matrix_fig(cm, acc, n):
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
            text=f"CNT Pulse Outcome — MLR Confusion Matrix  |  "
                 f"Accuracy {acc*100:.1f}%  |  n = {n}",
            font=dict(size=13, family="Times New Roman, serif"),
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
st.caption("Multinomial logistic regression — nanotube electromigration data.")
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
cm, acc, n, missing = load_confusion_matrix(
    selected_rowids=tuple(selected_rowids) if selected_rowids else None
)

if missing:
    st.error(f"Missing files: {', '.join(missing)}")
elif cm is not None:
    st.plotly_chart(confusion_matrix_fig(cm, acc, n),
                    use_container_width=True, config={"displayModeBar": False})

    K       = cm.shape[0]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    rows    = [{"Class": CLASS_LABELS[i],
                "Samples": int(cm[i].sum()),
                "Correct": int(cm[i, i]),
                "Accuracy (%)": f"{cm_norm[i, i]*100:.1f}"}
               for i in range(K)]
    st.subheader("Per-class breakdown")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.warning("Could not load model data.")

st.divider()
st.caption(
    "Class = ΔR/Rb (%) change in CNT resistance after electromigration pulsing.  "
    "Model trained using MATLAB mnrfit, replicated in Python."
)
