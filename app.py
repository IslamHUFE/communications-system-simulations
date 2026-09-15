import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from python_simulations.constellation import (
    compute_Pe,
    compute_ber_curve,
    compute_dmin,
    compute_Es,
    get_constellation,
)

st.set_page_config(
    page_title="Constellation Diagram Simulator",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Constellation Diagram Simulator")
st.caption("Interactive web version of the Python constellation simulation.")

# Keep the original simulator's dark visual language.
st.markdown(
    """
    <style>
    .stApp { background: #0d0d0d; }
    [data-testid="stSidebar"] { background: #161616; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Simulation")

    modulation = st.radio(
        "Modulation",
        ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"],
        index=1,
    )

    amplitude = st.slider(
        "Amplitude",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
    )

    sigma2 = st.slider(
        "Noise σ²",
        min_value=0.001,
        max_value=10.0,
        value=1.0,
        step=0.001,
    )

    n_samples = st.slider(
        "Samples",
        min_value=10,
        max_value=10000,
        value=100,
        step=10,
    )

    st.divider()
    st.subheader("Custom constellation")
    use_custom = st.checkbox("Use custom points")

    if "custom_points" not in st.session_state:
        st.session_state.custom_points = []

    if use_custom:
        col1, col2 = st.columns(2)
        with col1:
            custom_i = st.number_input("I", value=0.0, step=0.1)
        with col2:
            custom_q = st.number_input("Q", value=0.0, step=0.1)

        if st.button("+ Add point", use_container_width=True):
            st.session_state.custom_points.append([custom_i, custom_q])
            st.rerun()

        if st.button("Clear points", use_container_width=True):
            st.session_state.custom_points = []
            st.rerun()

# Generate constellation points.
if use_custom and st.session_state.custom_points:
    pts = np.asarray(st.session_state.custom_points, dtype=float)
    Es0 = np.mean(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    if Es0 > 0:
        pts = pts / np.sqrt(Es0) * amplitude
    mod_label = f"Custom ({len(pts)} pts)"
else:
    pts = get_constellation(modulation, amplitude)
    mod_label = modulation

M = len(pts)

# Generate noisy received samples.
sigma = np.sqrt(sigma2)
rng = np.random.default_rng()
idx = rng.integers(0, M, n_samples)
rx = pts[idx] + rng.normal(0, sigma, (n_samples, 2))

dmin = compute_dmin(pts)[0] if M >= 2 else 0.0
Pe = compute_Pe(pts, sigma2)
Es = compute_Es(pts)
bits = np.log2(M) if M > 1 else 0.0

left, right = st.columns([2.7, 1])

with left:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#161616")

    ax.scatter(
        rx[:, 0], rx[:, 1],
        s=12, alpha=0.30, color="#3A8EE6",
        label="Received (noisy)", rasterized=True,
    )

    if M >= 2:
        radius = dmin / 2
        for p in pts:
            ax.add_patch(
                plt.Circle(
                    p, radius, fill=False, edgecolor="#2e2e2e",
                    linewidth=0.8, linestyle="--",
                )
            )

        pair = compute_dmin(pts)[1]
        p0, p1 = pts[pair[0]], pts[pair[1]]
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict(
                arrowstyle="<->", color="#E8A020", lw=1.8,
                mutation_scale=14,
            ),
        )
        mid = (p0 + p1) / 2
        diff = p1 - p0
        perp = np.array([-diff[1], diff[0]])
        norm = np.linalg.norm(perp)
        if norm > 0:
            perp = perp / norm * amplitude * 0.09
        ax.text(
            mid[0] + perp[0], mid[1] + perp[1],
            f"d_min = {dmin:.3f}",
            color="#E8A020", fontsize=9, ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#0d0d0d",
                edgecolor="#E8A020", linewidth=0.8, alpha=0.9,
            ),
        )

    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=160, color="#1DB974", edgecolors="#0a3d25",
        linewidths=1.5, zorder=5, label="Symbols",
    )
    for i, p in enumerate(pts):
        ax.text(
            p[0], p[1], str(i), color="white", fontsize=7.5,
            ha="center", va="center", zorder=6, fontweight="bold",
        )

    ax.axhline(0, color="#2e2e2e", linewidth=0.9)
    ax.axvline(0, color="#2e2e2e", linewidth=0.9)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_xlabel("In-phase (I)", color="#888888")
    ax.set_ylabel("Quadrature (Q)", color="#888888")
    ax.tick_params(colors="#888888")
    ax.set_title(
        f"{mod_label}  |  Amplitude = {amplitude:.2f}  |  σ² = {sigma2:.3f}",
        color="#e8e8e8",
    )
    margin = amplitude * 1.65
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(facecolor="#1e1e1e", edgecolor="#2e2e2e", labelcolor="#e8e8e8")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with right:
    st.subheader("Metrics")
    st.metric("d_min", f"{dmin:.4f}" if M >= 2 else "—")
    st.metric("Pe", f"{Pe:.2e}" if Pe > 1e-15 else "< 1e-15")
    st.metric("Es", f"{Es:.4f}")
    st.metric("M", str(M))
    st.metric("Bits / symbol", f"{bits:.1f}")

    st.divider()
    st.subheader("BER vs Eb/N0")

    if M >= 2:
        eb_range = np.arange(-2, 22, 0.5)
        ber_t, ber_s = compute_ber_curve(pts, eb_range)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("#0d0d0d")
        ax2.set_facecolor("#161616")
        ax2.grid(True, which="both", alpha=0.2, linewidth=0.6)
        ax2.semilogy(eb_range, np.maximum(ber_t, 1e-15), color="#1DB974", lw=2, label="Theoretical")
        simulated = np.where(ber_s > 0, ber_s, np.nan)
        ax2.semilogy(eb_range, simulated, "o--", color="#E8A020", lw=1.3, markersize=4, label="Monte Carlo")
        ax2.set_xlabel("Eb/N0 (dB)", color="#888888")
        ax2.set_ylabel("BER", color="#888888")
        ax2.tick_params(colors="#888888")
        ax2.set_title("BER Curve", color="#e8e8e8")
        ax2.legend(facecolor="#1e1e1e", edgecolor="#2e2e2e", labelcolor="#e8e8e8", fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

st.caption("The simulation runs on the server; visitors only need a web browser.")
