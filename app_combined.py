import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="BA-NLS Predictor", layout="wide")

# ════════════════════════════════════════════════════════════
# 1.  GBTM MODEL PARAMETERS
# ════════════════════════════════════════════════════════════

# DBil – quadratic  (intercept, linear, quadratic)
DBIL_PARAMS = np.array([
    [174.8771,  -77.7399,   9.4778],
    [178.2627,  -74.1496,  15.4246],
    [149.7439,  -11.6684,  -4.2360],
    [310.8862, -234.4296,  55.2103],
    [159.157,    53.316,  -18.028 ],
])

# TBA – cubic  (intercept, linear, quadratic, cubic)
TBA_PARAMS = np.array([
    [205.862,  -140.332,   54.958,   -7.189],
    [ -4.943,   253.300, -146.564,   26.698],
    [145.1073,  -43.2551,  13.4262,   0.0  ],
    [318.078,  -207.331,   84.979,  -11.307],
    [ 81.980,   154.438,  -55.981,    7.915],
])

N_GROUPS    = 5
COLORS      = ['#1F77B4', '#2CA02C', '#9467BD', '#E377C2', '#FF7F0E']
TIME_LABELS = ["Baseline\n(Pre-op)", "2 Weeks\nPost-op",
               "1 Month\nPost-op",   "3 Months\nPost-op"]
TIME_ORIG   = np.array([1, 2, 3, 4])
TIME_SMOOTH = np.linspace(1, 4, 100)

# ════════════════════════════════════════════════════════════
# 2.  COX MODEL PARAMETERS
#     Model: Surv(NLStimeYH, NLS) ~ group.y + group.x
#     Reference levels: group.y = 1,  group.x = 1
# ════════════════════════════════════════════════════════════

COX_BETA = {
    'group.y2': 1.9017764, 'group.y3': 0.8491976,
    'group.y4': 2.3251954, 'group.y5': 1.3356971,
    'group.x2': 0.9446093, 'group.x3': 0.3395522,
    'group.x4': 0.3408952, 'group.x5': 0.9619062,
}

# Baseline cumulative hazard H0(t)  from basehaz(m4, centered=FALSE)
BASEHAZ_TIME = np.array([
     4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 55, 56,
    57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 84, 86, 89, 109, 112,
], dtype=float)

BASEHAZ_H0 = np.array([
    0.004453921, 0.006409470, 0.008831771, 0.015090783, 0.034104186,
    0.043191788, 0.052627500, 0.064430556, 0.074617911, 0.077946853,
    0.082253516, 0.085776166, 0.094957892, 0.095921008, 0.098837173,
    0.098837173, 0.100904251, 0.104074122, 0.109558935, 0.110703835,
    0.111861199, 0.116554278, 0.117742579, 0.120146044, 0.121373574,
    0.123911977, 0.125218633, 0.130609603, 0.133350866, 0.134762093,
    0.134762093, 0.134762093, 0.136226256, 0.137704752, 0.137704752,
    0.137704752, 0.140790189, 0.140790189, 0.140790189, 0.142456782,
    0.144181944, 0.145984379, 0.147832718, 0.147832718, 0.151732087,
    0.151732087, 0.151732087, 0.151732087, 0.153864084, 0.153864084,
    0.153864084, 0.156063412, 0.156063412, 0.156063412, 0.158353552,
    0.163073187, 0.170891511, 0.182638381, 0.185703004, 0.192313194,
    0.206504247, 0.214250705, 0.218347233, 0.236141788, 0.236141788,
    0.241183674, 0.247370929, 0.247370929, 0.247370929, 0.254943990,
    0.263295929, 0.263295929, 0.281998351, 0.281998351, 0.295796765,
    0.295796765, 0.295796765, 0.314061009, 0.314061009, 0.337873053,
    0.379942634,
], dtype=float)

_h0_interp = interp1d(
    BASEHAZ_TIME, BASEHAZ_H0, kind='previous',
    bounds_error=False, fill_value=(BASEHAZ_H0[0], BASEHAZ_H0[-1])
)

# ════════════════════════════════════════════════════════════
# 3.  CORE FUNCTIONS
# ════════════════════════════════════════════════════════════

def predict_quadratic(t, p):
    return p[0] + p[1]*t + p[2]*t**2

def predict_cubic(t, p):
    return p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3

def calc_group(values, param_matrix, poly='quadratic'):
    fn  = predict_quadratic if poly == 'quadratic' else predict_cubic
    mse = np.array([
        np.mean((np.array(values) - fn(TIME_ORIG, param_matrix[g]))**2)
        for g in range(N_GROUPS)
    ])
    if mse.min() == 0:
        probs = np.zeros(N_GROUPS); probs[mse.argmin()] = 1.0
    else:
        w = 1.0 / (mse + 1e-10); probs = w / w.sum()
    return int(mse.argmin()) + 1, probs

def lp_value(gy: int, gx: int) -> float:
    lp = 0.0
    if gy > 1: lp += COX_BETA.get(f'group.y{gy}', 0.0)
    if gx > 1: lp += COX_BETA.get(f'group.x{gx}', 0.0)
    return lp

def survival_curve(lp: float, t_grid: np.ndarray) -> np.ndarray:
    return np.exp(-_h0_interp(t_grid) * np.exp(lp))

def nls_prob(lp: float, months: float) -> float:
    h0 = float(_h0_interp(np.array([months])))
    return float(1.0 - np.exp(-h0 * np.exp(lp)))

def draw_trajectory(ax, values, param_matrix, poly, ylabel, title, pt_label):
    fn = predict_quadratic if poly == 'quadratic' else predict_cubic
    for g in range(N_GROUPS):
        ax.plot(TIME_SMOOTH, fn(TIME_SMOOTH, param_matrix[g]),
                color=COLORS[g], linestyle='--', linewidth=1.5,
                alpha=0.65, label=f'Trajectory {g+1}')
    ax.plot(TIME_ORIG, values, 'ko-', linewidth=2, markersize=8,
            label=pt_label, zorder=5)
    ax.set_xlabel('Follow-up Time', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(TIME_ORIG); ax.set_xticklabels(TIME_LABELS, fontsize=8)
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.25)

# ════════════════════════════════════════════════════════════
# 4.  PAGE LAYOUT
# ════════════════════════════════════════════════════════════

st.title("BA-NLS Predictor")
st.markdown(
    "Prognostic stratification based on postoperative **DBil** and **TBA** "
    "trajectories in biliary atresia — powered by GBTM + Cox regression."
)
st.divider()

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header("Patient Data Input")
INPUT_LABELS = ["Baseline (Pre-op)", "2 Weeks Post-op",
                "1 Month Post-op",   "3 Months Post-op"]

st.sidebar.subheader("DBil Values (μmol/L)")
dbil_vals = [
    st.sidebar.number_input(lbl, 0.0, 900.0, float(i*20+10),
                            1.0, format="%.1f", key=f"d{i}")
    for i, lbl in enumerate(INPUT_LABELS)
]
st.sidebar.divider()
st.sidebar.subheader("TBA Values (μmol/L)")
tba_vals = [
    st.sidebar.number_input(lbl, 0.0, 900.0, float(i*15+8),
                            1.0, format="%.1f", key=f"t{i}")
    for i, lbl in enumerate(INPUT_LABELS)
]
run = st.sidebar.button("▶  Run Prediction", use_container_width=True)

# ── Trajectory charts (always visible) ──────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("DBil Trajectory")
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.8))
    draw_trajectory(ax1, dbil_vals, DBIL_PARAMS, 'quadratic',
                    'DBil (μmol/L)', 'DBil — GBTM 5-Group', 'This patient')
    plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

with c2:
    st.subheader("TBA Trajectory")
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    draw_trajectory(ax2, tba_vals, TBA_PARAMS, 'cubic',
                    'TBA (μmol/L)', 'TBA — GBTM 5-Group', 'This patient')
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ════════════════════════════════════════════════════════════
# 5.  PREDICTION RESULTS
# ════════════════════════════════════════════════════════════

if run:
    st.divider()

    # ── 5a. GBTM grouping ────────────────────────────────────
    gy, dbil_probs = calc_group(dbil_vals, DBIL_PARAMS, 'quadratic')
    gx, tba_probs  = calc_group(tba_vals,  TBA_PARAMS,  'cubic')
    lp             = lp_value(gy, gx)

    col_d, col_t, col_cp = st.columns(3)

    with col_d:
        st.markdown("#### DBil Trajectory Group")
        st.success(f"**Trajectory {gy}**  (group.y = {gy})")
        st.dataframe(pd.DataFrame({
            'Group':           [f'Trajectory {i+1}' for i in range(N_GROUPS)],
            'Probability (%)': [round(p*100, 1) for p in dbil_probs],
        }), hide_index=True, use_container_width=True)

    with col_t:
        st.markdown("#### TBA Trajectory Group")
        st.success(f"**Trajectory {gx}**  (group.x = {gx})")
        st.dataframe(pd.DataFrame({
            'Group':           [f'Trajectory {i+1}' for i in range(N_GROUPS)],
            'Probability (%)': [round(p*100, 1) for p in tba_probs],
        }), hide_index=True, use_container_width=True)

    with col_cp:
        st.markdown("#### NLS Cumulative Probability")
        checkpoints = [12, 24, 36, 60]
        st.dataframe(pd.DataFrame({
            'Time Point': [f'{m} months' for m in checkpoints],
            'NLS Prob (%)': [f'{nls_prob(lp, m)*100:.1f}%' for m in checkpoints],
            'NLS-Free (%)': [f'{(1-nls_prob(lp, m))*100:.1f}%' for m in checkpoints],
        }), hide_index=True, use_container_width=True)
        st.caption(f"LP = {lp:.4f}  |  ref: group.y=1, group.x=1")

    # ── 5b. Survival curve ───────────────────────────────────
    st.divider()
    st.subheader("Cox Survival Curve — NLS-Free Survival")

    t_grid = np.linspace(4, 112, 400)
    fig3, ax3 = plt.subplots(figsize=(9, 4.8))

    # Reference trajectories (group.x=1, group.y varies)
    ref_cfg = [
        (1, '#1F77B4', '-',  'group.y=1 (ref)'),
        (2, '#D62728', '--', 'group.y=2'),
        (3, '#2CA02C', '--', 'group.y=3'),
        (4, '#9467BD', '--', 'group.y=4'),
        (5, '#FF7F0E', '--', 'group.y=5'),
    ]
    for _gy, col, ls, lbl in ref_cfg:
        _lp = lp_value(_gy, 1)
        ax3.plot(t_grid, survival_curve(_lp, t_grid)*100,
                 color=col, linestyle=ls, linewidth=1.1,
                 alpha=0.45, label=lbl)

    # This patient
    s_pt = survival_curve(lp, t_grid)
    ax3.plot(t_grid, s_pt*100, color='black', linewidth=2.5, zorder=5,
             label=f'This patient (group.y={gy}, group.x={gx})')

    # Annotate key time points
    for mo in [12, 36, 60]:
        sv = (1 - nls_prob(lp, mo)) * 100
        ax3.axvline(mo, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax3.annotate(
            f'{sv:.1f}%',
            xy=(mo, sv), xytext=(mo+2.5, sv+3.5),
            fontsize=8, color='black',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
        )

    ax3.set_xlabel('Time (months)', fontsize=11)
    ax3.set_ylabel('NLS-Free Survival (%)', fontsize=11)
    ax3.set_title('Cox Model: NLS-Free Survival Probability', fontsize=12,
                  fontweight='bold')
    ax3.set_xlim(0, 115); ax3.set_ylim(0, 105)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax3.legend(fontsize=8, loc='lower left')
    ax3.grid(True, alpha=0.25)
    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    # ── 5c. Full risk comparison table ──────────────────────
    st.divider()
    st.subheader("Risk Comparison — All Trajectory Group Combinations")

    rows = []
    for _gy in range(1, 6):
        for _gx in range(1, 6):
            _lp  = lp_value(_gy, _gx)
            rows.append({
                'group.y (DBil)': f'Traj {_gy}',
                'group.x (TBA)':  f'Traj {_gx}',
                'NLS @ 12m':      f'{nls_prob(_lp,12)*100:.1f}%',
                'NLS @ 36m':      f'{nls_prob(_lp,36)*100:.1f}%',
                'NLS @ 60m':      f'{nls_prob(_lp,60)*100:.1f}%',
                '':               '◀ This patient' if (_gy==gy and _gx==gx) else '',
            })

    df_cmp = pd.DataFrame(rows)

    def hl(row):
        is_pt = row[''] == '◀ This patient'
        return ['background-color:#fff3cd;font-weight:bold']*len(row) if is_pt else ['']*len(row)

    st.dataframe(
        df_cmp.style.apply(hl, axis=1),
        use_container_width=True, hide_index=True, height=340
    )
    st.caption(
        "NLS probability = 1 − exp(−H₀(t) · exp(LP))  "
        "| LP = Σ β · group indicator"
    )

# ── Input summary (always visible) ──────────────────────────
st.divider()
st.subheader("Input Data Summary")
st.dataframe(pd.DataFrame({
    'Time Point':    INPUT_LABELS,
    'DBil (μmol/L)': dbil_vals,
    'TBA (μmol/L)':  tba_vals,
}), hide_index=True, use_container_width=True)
