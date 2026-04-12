import warnings

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from scipy.special import erfc

warnings.filterwarnings("ignore")

# ─── Color palette ───────────────────────────────────────────
C_BG = "#0d0d0d"
C_PANEL = "#161616"
C_CARD = "#1e1e1e"
C_BORDER = "#2e2e2e"
C_TEXT = "#e8e8e8"
C_MUTED = "#888888"
C_GREEN = "#1DB974"
C_BLUE = "#3A8EE6"
C_AMBER = "#E8A020"
C_RED = "#D94F4F"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": C_BG,
        "axes.facecolor": C_PANEL,
        "axes.edgecolor": C_BORDER,
        "axes.labelcolor": C_MUTED,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "grid.color": C_BORDER,
        "grid.linewidth": 0.6,
        "text.color": C_TEXT,
    }
)

# ─────────────────────────────────────────────
#  Constellation definitions
# ─────────────────────────────────────────────


def get_constellation(mod, amplitude=1.0):
    if mod == "BPSK":
        pts = np.array([[-1.0, 0.0], [1.0, 0.0]])
    elif mod == "QPSK":
        angles = np.pi / 4 + np.arange(4) * np.pi / 2
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
    elif mod == "8-PSK":
        angles = np.arange(8) * 2 * np.pi / 8
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
    elif mod == "16-QAM":
        levels = np.array([-3, -1, 1, 3])
        I, Q = np.meshgrid(levels, levels)
        pts = np.column_stack([I.ravel(), Q.ravel()]) / 3.0
    elif mod == "64-QAM":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        I, Q = np.meshgrid(levels, levels)
        pts = np.column_stack([I.ravel(), Q.ravel()]) / 7.0
    else:
        pts = np.zeros((1, 2))

    Es = np.mean(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    if Es > 0:
        pts = pts / np.sqrt(Es)
    return pts * amplitude


# ─────────────────────────────────────────────
#  Math helpers
# ─────────────────────────────────────────────


def qfunc(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def compute_dmin(pts):
    if len(pts) < 2:
        return 0.0, (0, 0)
    dmin, pair = np.inf, (0, 1)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < dmin:
                dmin, pair = d, (i, j)
    return float(dmin), pair


def compute_Pe(pts, sigma2):
    if len(pts) < 2 or sigma2 <= 0:
        return 0.0
    dmin, _ = compute_dmin(pts)
    return float(qfunc(dmin / (2 * np.sqrt(sigma2))))


def compute_Es(pts):
    return float(np.mean(pts[:, 0] ** 2 + pts[:, 1] ** 2))


def compute_ber_curve(pts, eb_n0_db_range, n_symbols=4000):
    M = len(pts)
    if M < 2:
        return np.zeros_like(eb_n0_db_range), np.zeros_like(eb_n0_db_range)

    k = max(1.0, np.log2(M))
    Es = compute_Es(pts)
    ber_t, ber_s = [], []

    for eb_n0_db in eb_n0_db_range:
        eb_n0 = 10 ** (eb_n0_db / 10.0)
        es_n0 = eb_n0 * k
        sigma2 = Es / (2 * es_n0) if es_n0 > 0 else 1.0
        sigma = np.sqrt(sigma2)

        dmin_val, _ = compute_dmin(pts)
        Pe = float(qfunc(dmin_val / (2 * sigma))) if sigma > 0 else 0.0
        ber_t.append(Pe / k)

        # Monte-Carlo
        idx_tx = np.random.randint(0, M, n_symbols)
        tx = pts[idx_tx]
        rx = tx + np.random.normal(0, sigma, tx.shape)
        diffs = rx[:, None, :] - pts[None, :, :]
        dists = np.sum(diffs**2, axis=2)
        idx_rx = np.argmin(dists, axis=1)
        ber_s.append(np.sum(idx_rx != idx_tx) / n_symbols / k)

    return np.array(ber_t), np.array(ber_s)


# ─────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────


class ConstellationApp:
    def __init__(self):
        self.mod = "QPSK"
        self.amplitude = 1.0
        self.sigma2 = 0.05
        self.n_samples = 500
        self.custom_pts = []

        self.fig = plt.figure(figsize=(15, 8.5))
        self.fig.canvas.manager.set_window_title("Constellation Diagram Simulator")
        self._build_layout()
        self._draw()
        plt.show()

    # ── Layout ────────────────────────────────

    def _build_layout(self):
        # Outer: left controls | right plot
        outer = gridspec.GridSpec(
            1,
            2,
            figure=self.fig,
            left=0.05,
            right=0.98,
            top=0.96,
            bottom=0.05,
            width_ratios=[1, 2.3],
            wspace=0.07,
        )

        # Left column: 10 rows
        left = gridspec.GridSpecFromSubplotSpec(
            10,
            1,
            subplot_spec=outer[0],
            hspace=0.55,
            height_ratios=[2.4, 0.85, 0.85, 0.85, 0.4, 0.75, 0.75, 0.4, 0.85, 0.95],
        )

        def section_label(ax, txt):
            ax.set_facecolor(C_BG)
            ax.axis("off")
            ax.text(
                0.0,
                0.5,
                txt,
                transform=ax.transAxes,
                color=C_MUTED,
                fontsize=8,
                va="center",
                ha="left",
                style="italic",
            )

        # ── Modulation radio ─────────────────────────────────────
        self.ax_radio = self.fig.add_subplot(left[0])
        self.ax_radio.set_facecolor(C_CARD)
        for sp in self.ax_radio.spines.values():
            sp.set_edgecolor(C_BORDER)
        self.ax_radio.set_title(
            "  Modulation", color=C_MUTED, fontsize=9, pad=5, loc="left"
        )

        mods = ("BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM")
        self.radio = RadioButtons(self.ax_radio, mods, active=1, activecolor=C_GREEN)
        for lbl in self.radio.labels:
            lbl.set_color(C_TEXT)
            lbl.set_fontsize(9.5)
        # set_radio_props available in mpl >= 3.7
        try:
            self.radio.set_radio_props(facecolor=C_MUTED, edgecolor=C_MUTED, s=36)
        except Exception:
            pass
        self.radio.on_clicked(self._on_mod)

        # ── Sliders ───────────────────────────────────────────────
        ####################################################Amplitude##########################################
        self.ax_amp = self.fig.add_subplot(left[1])
        self.sl_amp = Slider(
            self.ax_amp,
            "Amplitude",
            0.1,
            100,
            valinit=1.0,
            valstep=0.1,
            color=C_GREEN,
            facecolor=C_BORDER,
        )
        self._style_slider(self.sl_amp)
        self.sl_amp.on_changed(self._on_change)

        #####################################################noise##########################################

        self.ax_noise = self.fig.add_subplot(left[2])
        self.sl_noise = Slider(
            self.ax_noise,
            "Noise  σ²",
            0.001,
            10,
            valinit=1,
            valstep=0.001,
            color=C_AMBER,
            facecolor=C_BORDER,
        )
        self._style_slider(self.sl_noise)
        self.sl_noise.on_changed(self._on_change)
        #####################################################samples##########################################
        self.ax_samp = self.fig.add_subplot(left[3])
        self.sl_samp = Slider(
            self.ax_samp,
            "Samples",
            10,
            10000,
            valinit=100,
            valstep=10,
            color=C_BLUE,
            facecolor=C_BORDER,
        )
        self._style_slider(self.sl_samp)
        self.sl_samp.on_changed(self._on_change)

        # ── Custom section ────────────────────────────────────────
        self.ax_ctitle = self.fig.add_subplot(left[4])
        section_label(self.ax_ctitle, "── Custom point  (I, Q)")

        self.ax_cx = self.fig.add_subplot(left[5])
        self.tb_cx = TextBox(
            self.ax_cx, "I", initial="0.0", color=C_CARD, hovercolor=C_BORDER
        )
        self._style_textbox(self.tb_cx)

        self.ax_cy = self.fig.add_subplot(left[6])
        self.tb_cy = TextBox(
            self.ax_cy, "Q", initial="0.0", color=C_CARD, hovercolor=C_BORDER
        )
        self._style_textbox(self.tb_cy)

        # ── Actions section ───────────────────────────────────────
        self.ax_btitle = self.fig.add_subplot(left[7])
        section_label(self.ax_btitle, "── Actions")

        btn_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=left[8], wspace=0.22
        )
        self.ax_badd = self.fig.add_subplot(btn_gs[0])
        self.ax_bclr = self.fig.add_subplot(btn_gs[1])
        self.ax_bber = self.fig.add_subplot(btn_gs[2])

        self.btn_add = Button(self.ax_badd, "+ Add", color=C_CARD, hovercolor="#1a3d28")
        self.btn_clr = Button(self.ax_bclr, "Clear", color=C_CARD, hovercolor="#3d1a1a")
        self.btn_ber = Button(
            self.ax_bber, "BER Plot", color=C_CARD, hovercolor="#1a2a3d"
        )

        for btn in (self.btn_add, self.btn_clr, self.btn_ber):
            btn.label.set_color(C_TEXT)
            btn.label.set_fontsize(9)
            for sp in btn.ax.spines.values():
                sp.set_edgecolor(C_BORDER)

        self.btn_add.on_clicked(self._add_custom)
        self.btn_clr.on_clicked(self._clear_custom)
        self.btn_ber.on_clicked(self._show_ber)

        # ── Metrics card ──────────────────────────────────────────
        self.ax_metrics = self.fig.add_subplot(left[9])
        self.ax_metrics.set_facecolor(C_CARD)
        for sp in self.ax_metrics.spines.values():
            sp.set_edgecolor(C_BORDER)
        self.ax_metrics.axis("off")
        self.metrics_txt = self.ax_metrics.text(
            0.5,
            0.5,
            "",
            transform=self.ax_metrics.transAxes,
            ha="center",
            va="center",
            color=C_TEXT,
            fontsize=8.5,
            fontfamily="monospace",
            linespacing=1.9,
        )

        # ── Main plot ─────────────────────────────────────────────
        self.ax_main = self.fig.add_subplot(outer[1])
        for sp in self.ax_main.spines.values():
            sp.set_edgecolor(C_BORDER)

    def _style_slider(self, sl):
        sl.label.set_color(C_TEXT)
        sl.label.set_fontsize(9)
        sl.valtext.set_color(C_MUTED)
        sl.valtext.set_fontsize(9)

    def _style_textbox(self, tb):
        tb.label.set_color(C_TEXT)
        tb.label.set_fontsize(9)
        tb.text_disp.set_color(C_TEXT)
        tb.text_disp.set_fontsize(9)

    # ── Points ────────────────────────────────

    def _get_pts(self):
        if self.custom_pts:
            pts = np.array(self.custom_pts, dtype=float)
            Es = np.mean(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            if Es > 0:
                pts = pts / np.sqrt(Es) * self.amplitude
            return pts
        return get_constellation(self.mod, self.amplitude)

    # ── Main draw ─────────────────────────────

    def _draw(self):
        ax = self.ax_main
        ax.cla()
        ax.set_facecolor(C_PANEL)
        ax.grid(True, alpha=0.18, linewidth=0.6, zorder=0)
        ax.axhline(0, color=C_BORDER, linewidth=0.9, zorder=1)
        ax.axvline(0, color=C_BORDER, linewidth=0.9, zorder=1)
        ax.set_xlabel("In-phase  (I)", color=C_MUTED, fontsize=10, labelpad=8)
        ax.set_ylabel("Quadrature  (Q)", color=C_MUTED, fontsize=10, labelpad=10)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)

        mod_label = (
            self.mod if not self.custom_pts else f"Custom ({len(self.custom_pts)} pts)"
        )
        ax.set_title(
            f"{mod_label}   |   Amplitude = {self.amplitude:.2f}"
            f"   |   σ² = {self.sigma2:.3f}",
            color=C_TEXT,
            fontsize=10.5,
            pad=10,
        )

        pts = self._get_pts()
        M = len(pts)
        if M == 0:
            self.fig.canvas.draw_idle()
            return

        # Noisy samples
        sigma = np.sqrt(self.sigma2)
        idx = np.random.randint(0, M, self.n_samples)
        rx = pts[idx] + np.random.normal(0, sigma, (self.n_samples, 2))
        ax.scatter(
            rx[:, 0],
            rx[:, 1],
            s=9,
            alpha=0.28,
            color=C_BLUE,
            zorder=2,
            label="Received",
            rasterized=True,
        )

        # Decision-region circles + dmin
        dmin_val, pair = 0.0, (0, 1)
        if M >= 2:
            dmin_val, pair = compute_dmin(pts)
            r = dmin_val / 2
            for p in pts:
                circ = plt.Circle(
                    p,
                    r,
                    fill=False,
                    edgecolor=C_BORDER,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=3,
                )
                ax.add_patch(circ)

            # dmin arrow between the closest pair
            p0, p1 = pts[pair[0]], pts[pair[1]]
            mid = (p0 + p1) / 2

            # perpendicular offset for label
            diff = p1 - p0
            perp = np.array([-diff[1], diff[0]])
            pnorm = np.linalg.norm(perp)
            if pnorm > 0:
                perp = perp / pnorm * (self.amplitude * 0.09)

            ax.annotate(
                "",
                xy=p1,
                xytext=p0,
                arrowprops=dict(
                    arrowstyle="<->", color=C_AMBER, lw=1.8, mutation_scale=14
                ),
                zorder=6,
            )
            ax.text(
                mid[0] + perp[0],
                mid[1] + perp[1],
                f"d_min = {dmin_val:.3f}",
                color=C_AMBER,
                fontsize=8.5,
                ha="center",
                va="center",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=C_BG,
                    edgecolor=C_AMBER,
                    linewidth=0.8,
                    alpha=0.88,
                ),
            )

        # Constellation points
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=160,
            color=C_GREEN,
            zorder=8,
            edgecolors="#0a3d25",
            linewidths=1.5,
            label="Symbols",
        )
        for i, p in enumerate(pts):
            ax.text(
                p[0],
                p[1],
                str(i),
                color="white",
                fontsize=7.5,
                ha="center",
                va="center",
                zorder=9,
                fontweight="bold",
            )

        # Legend
        legend_els = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=C_GREEN,
                markeredgecolor="#0a3d25",
                markersize=9,
                label="Symbols",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=C_BLUE,
                alpha=0.6,
                markersize=6,
                label="Received (noisy)",
            ),
            Line2D([0], [0], color=C_AMBER, lw=1.5, label="d_min"),
            Line2D(
                [0],
                [0],
                color=C_BORDER,
                lw=1,
                linestyle="--",
                label="Decision boundary",
            ),
        ]
        ax.legend(
            handles=legend_els,
            loc="upper right",
            fontsize=8.5,
            facecolor=C_CARD,
            edgecolor=C_BORDER,
            labelcolor=C_TEXT,
            framealpha=0.92,
            handlelength=1.8,
        )

        # Axis limits
        margin = self.amplitude * 1.65
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal", adjustable="box")

        # Metrics
        Pe = compute_Pe(pts, self.sigma2)
        Es = compute_Es(pts)
        bits = np.log2(M) if M > 1 else 0
        Pe_str = f"{Pe:.2e}" if Pe > 1e-15 else "< 1e-15"
        dm_str = f"{dmin_val:.4f}" if M >= 2 else "—"

        self.metrics_txt.set_text(
            f"d_min = {dm_str}    Pe = {Pe_str}\n"
            f"Es = {Es:.4f}       M = {M}  ({bits:.1f} bit/sym)"
        )

        self.fig.canvas.draw_idle()

    # ── BER window ────────────────────────────

    def _show_ber(self, event):
        pts = self._get_pts()
        if len(pts) < 2:
            return

        eb_range = np.arange(-2, 22, 0.5)
        ber_t, ber_s = compute_ber_curve(pts, eb_range)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(C_BG)
        ax2.set_facecolor(C_PANEL)
        for sp in ax2.spines.values():
            sp.set_edgecolor(C_BORDER)
        ax2.grid(True, which="both", alpha=0.2, linewidth=0.6)
        ax2.tick_params(colors=C_MUTED, labelsize=9)
        ax2.set_xlabel("Eb / N0  (dB)", color=C_MUTED, fontsize=10, labelpad=8)
        ax2.set_ylabel("BER", color=C_MUTED, fontsize=10, labelpad=8)

        mod_label = self.mod if not self.custom_pts else f"Custom ({len(pts)} pts)"
        ax2.set_title(
            f"BER vs Eb/N0   |   {mod_label}   A = {self.amplitude:.2f}",
            color=C_TEXT,
            fontsize=11,
            pad=10,
        )

        vt = ber_t > 1e-12
        vs = ber_s > 0

        if vt.any():
            ax2.semilogy(
                eb_range[vt],
                ber_t[vt],
                "-",
                color=C_GREEN,
                lw=2,
                label="Theoretical BER",
            )
        if vs.any():
            ax2.semilogy(
                eb_range[vs],
                ber_s[vs],
                "o--",
                color=C_AMBER,
                lw=1.5,
                markersize=5,
                markerfacecolor=C_BG,
                markeredgewidth=1.5,
                label="Simulated BER (Monte Carlo)",
            )

        ax2.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=9)
        fig2.tight_layout(pad=1.5)
        plt.show()

    # ── Event handlers ────────────────────────

    def _on_mod(self, label):
        self.mod = label
        self.custom_pts = []
        self._draw()

    def _on_change(self, _val):
        self.amplitude = float(self.sl_amp.val)
        self.sigma2 = float(self.sl_noise.val)
        self.n_samples = int(self.sl_samp.val)
        self._draw()

    def _add_custom(self, _event):
        try:
            x = float(self.tb_cx.text)
            y = float(self.tb_cy.text)
            self.custom_pts.append([x, y])
            self._draw()
        except ValueError:
            pass

    def _clear_custom(self, _event):
        self.custom_pts = []
        self._draw()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ConstellationApp()
