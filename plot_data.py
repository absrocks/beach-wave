#!/usr/bin/env python3

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def _sci_label(t):
    exp = int(math.floor(math.log10(abs(t))))
    mant = t / 10 ** exp
    if abs(mant - round(mant)) < 1e-9:
        mant = int(round(mant))
    if mant == 1:
        return rf"$10^{{{exp}}}$"
    return rf"${mant}\times10^{{{exp}}}$"

INPUT_PARAMETERS = {
    'base_directory': [
        '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/eroded_full_init/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/bar_nourishment_full_init/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/bar_nourishment_full_final/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/berm_nourishment_full_init/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/berm_nourishment_full_final/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/profile_full_scale_init/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/profile_full_scale_final/logs',
      
    ],
    'spanwise_spectra': {
        'enabled': False, 
        'case': '/Users/abhishek/work/free_surface_2025/wave_tank/full_scale/bar_nourishment_full_10mm/logs/spanwise',
    },
    'csv_list': [
        'erode_full_init.csv',
        'bar_full_init.csv',
        'bar_full_final.csv',
        'berm_full_init.csv',
        'berm_full_final.csv',
        'profile_full_init.csv',
        'profile_full_final.csv',
    ],
    'add_csv': True,
    'csv_append_x': '33.25',
    # Profile (dotted, secondary axis) fills this bottom fraction of the plot
    # height, so it stays below the parameter curves. 0.5 = bottom half.
    'profile_axis_fraction': 0.5,
    'variable': 'TKE_epsilon_turb_pw_avg_Y_epsilon_pw_avg_Y_depthavg', # or 'TKE', or 'epsilon_turb'
    'output_parameters': ['TKE_avg', 'epsilon_turb_pw_avg_Y_avg'],# 'epsilon_pw_avg_Y_avg'], # columns (besides X) to read from .dat
    't_start': 28.3,
    't_end': 38,
    'window': False,
    'window_periods': [31.6],
    'Xmask': None, #[39.25, 39.4], # range of X to mask out (e.g. near the wall)
    'streamwise_avg': True,
    'streamwise_int': False,
    'streamwise_avg_range': [40, 48], # range of X to average over for streamwise averaging
    'case_name_index': 7,
    'exclude_zero': False,
    'streamwise_cum_avg': False,
    'smooth_sigma': 2,
    'value_cap': 1e2,
    'x_bin': 0.04,
    'plot_x_range': [34, 48],
}

cfg = dict(INPUT_PARAMETERS)


def _bad_mask(vals):
    """Boolean mask of values to EXCLUDE from any average:
      - NaN / inf  (always)
      - |value| > cfg['value_cap']  (unphysical blow-ups, e.g. eps 1e+56)
      - value == 0  if cfg['exclude_zero'] is True
    """
    vals = np.asarray(vals, dtype=float)
    bad = ~np.isfinite(vals)
    cap = cfg.get('value_cap', None)
    if cap is not None:
        bad |= np.abs(vals) > float(cap)
    if bool(cfg.get('exclude_zero', False)):
        bad |= (vals == 0.0)
    return bad


def _smooth(y):
    """Gaussian-smooth a 1-D array using cfg['smooth_sigma'] (0 disables)."""
    sigma = float(cfg.get('smooth_sigma', 0) or 0)
    if sigma <= 0:
        return np.asarray(y, dtype=float)
    return gaussian_filter1d(np.asarray(y, dtype=float), sigma=sigma)


def _apply_plot_filters(x_array, eps_avg):
    """Apply Xmask (remove a range) then plot_x_range (keep a range) to the
    line-plot data."""
    x_array = np.asarray(x_array, dtype=float)
    eps_avg = np.asarray(eps_avg, dtype=float)
    xm = cfg.get('Xmask')
    if xm is not None:
        keep = ~((x_array >= xm[0]) & (x_array <= xm[1]))
        x_array, eps_avg = x_array[keep], eps_avg[keep]
    pr = cfg.get('plot_x_range')
    if pr is not None:
        keep = (x_array >= pr[0]) & (x_array <= pr[1])
        x_array, eps_avg = x_array[keep], eps_avg[keep]
    return x_array, eps_avg


def _read_profile_csv(path):
    """Read a beach-profile CSV (x, z) — comma or whitespace delimited, with an
    optional header row. Returns (x, z) numpy arrays."""
    cand = path
    if not os.path.isabs(path) and not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(alt):
            cand = alt
    xs, zs = [], []
    with open(cand) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r'[,\s]+', line)
            try:
                x = float(parts[0]); z = float(parts[1])
            except (ValueError, IndexError):
                continue  # header / malformed row
            xs.append(x); zs.append(z)
    if not xs:
        raise ValueError(f"no numeric (x,z) rows found in {cand}")
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(zs)[order]


def _overlay_profile(case_index, color):
    """Overlay the beach profile for a case as a dotted line in `color` on a
    shared secondary Y-axis (bed elevation), so the bathymetry can be read
    alongside the TKE/eps curve. CSV is csv_list[case_index] (same order as
    base_directory); its X is offset by cfg['csv_append_x']."""
    if not cfg.get('add_csv'):
        return
    csv_list = cfg.get('csv_list') or []
    if case_index is None or case_index >= len(csv_list):
        if cfg.get('add_csv'):
            print(f"[add_csv] no csv for case_index={case_index} "
                  f"(csv_list has {len(csv_list)} entries)")
        return
    csv_name = csv_list[case_index]
    try:
        px, pz = _read_profile_csv(csv_name)
    except Exception as e:
        print(f"[add_csv] could not read '{csv_name}': {e}")
        return
    try:
        x_off = float(cfg.get('csv_append_x', 0) or 0)
    except (TypeError, ValueError):
        x_off = 0.0
    px = px + x_off

    ax = plt.gca()
    fig = ax.figure
    ax2 = getattr(fig, '_profile_ax', None)
    if ax2 is None:
        ax2 = ax.twinx()
        ax2.set_ylabel('Bed elevation z (m)')
        fig._profile_ax = ax2
        fig._profile_primary = ax          # remember the parameter axes
        fig._profile_zrange = [float('inf'), float('-inf')]
    ax2.plot(px, pz, linestyle=':', linewidth=1.5, color=color)

    # Keep the profile in the LOWER part of the plot so it doesn't overlap the
    # parameter curves: make the secondary axis span (z-data span / fraction),
    # i.e. with fraction=0.5 the bathymetry fills only the bottom half.
    zr = fig._profile_zrange
    zr[0] = min(zr[0], float(np.min(pz)))
    zr[1] = max(zr[1], float(np.max(pz)))
    zmin, zmax = zr
    span = max(zmax - zmin, 1e-9)
    frac = float(cfg.get('profile_axis_fraction', 0.5) or 0.5)
    ax2.set_ylim(zmin, zmin + span / frac)

    # Tick labels only up to the real bed max (~1 m); the rest of the axis (the
    # upper half) is left blank. Axis still extends to keep the profile low.
    step = 0.2
    lo_tick = np.floor(min(zmin, 0.0) / step) * step
    top_tick = np.ceil(zmax / step) * step      # e.g. 0.99 -> 1.0
    ticks = np.arange(lo_tick, top_tick + 1e-9, step)
    ax2.set_yticks(ticks)

    # Restore the PRIMARY (parameter) axes as current. ax.twinx() switched the
    # current axes to the secondary one, which would make the next case's
    # plt.plot() draw its parameter line on the secondary axis with a restarted
    # color cycle (duplicate colors + wrong scale). This puts it back.
    plt.sca(fig._profile_primary)


#  REFERENCE_CASES   — case names always added to every bar group (e.g. the
#                      "erode" baseline you compare every treatment against).
#  CASE_GROUPS       — list of treatment groups; each sublist is the set of
#                      case names that should be bundled together along with
#                      the reference cases.
#
#  Both are DERIVED from cfg['base_directory'] so renaming a case dir only
#  requires editing base_directory. The case name is the path segment at
#  cfg['case_name_index']; cases sharing a stem (name minus a trailing
#  '_init'/'_final') form one group, ordered init -> final. Any case whose
#  name contains 'erode' is treated as the reference and prepended to every
#  group. Groups keep base_directory order, e.g.:
#      Group 1:  eroded + bar_init  + bar_final
#      Group 2:  eroded + berm_init + berm_final
#      Group 3:  eroded + profile_init + profile_final
def _derive_case_groups(dirs, seg_idx):
    if isinstance(dirs, str):
        dirs = [dirs]
    names = []
    for d in dirs:
        n = Path(d).parts[seg_idx]
        if n not in names:
            names.append(n)

    def _stem(n):
        return re.sub(r'_(init|final)$', '', n)

    refs, groups, order = [], {}, []
    for n in names:
        if 'erode' in n:
            refs.append(n)
            continue
        s = _stem(n)
        if s not in groups:
            groups[s] = []
            order.append(s)
        groups[s].append(n)

    def _rank(n):
        return 0 if n.endswith('_init') else (1 if n.endswith('_final') else 2)

    return refs, [sorted(groups[s], key=_rank) for s in order]

REFERENCE_CASES, CASE_GROUPS = _derive_case_groups(
    cfg['base_directory'], cfg.get('case_name_index', 7))

def _grouped_bar_chart(results, ylabel, title, ylim=None, group_gap=1.0, value_fmt='%.3g'):
    """
    Bar chart: one cluster per entry in CASE_GROUPS. Each cluster holds the
    REFERENCE_CASES bars (in order) followed by that group's treatment bars.

    `results` is a list of (case_name, value) tuples produced by the main
    loop; values for case names not in any group are silently ignored.

    If CASE_GROUPS is empty or None, falls back to the old behavior of one
    cluster per non-reference case (compared against all references).
    """
    by_case = dict(results)
    refs_present = [r for r in REFERENCE_CASES if r in by_case]

    groups = []
    if CASE_GROUPS:
        for treatment_cases in CASE_GROUPS:
            present = [c for c in treatment_cases if c in by_case]
            if present:
                groups.append(refs_present + present)

    # Fall back to one-cluster-per-case when CASE_GROUPS is empty OR when none
    # of the results matched any group (e.g. a one-off case like
    # 'bar_nourishment_full' that isn't listed in CASE_GROUPS). Without this
    # the bar chart would silently draw nothing.
    if not groups:
        ordered = [c for c, _ in results if c in by_case]
        if not ordered:
            return
        groups = [[c] for c in ordered]

    max_per_group = max(len(g) for g in groups)
    bar_w = 0.8

    palette = plt.get_cmap('tab10')
    all_cases = []
    for g in groups:
        for c in g:
            if c not in all_cases:
                all_cases.append(c)
    color_for = {case: palette(i) for i, case in enumerate(all_cases)}

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(groups)), 5))
    seen = set()
    for g_idx, group_cases in enumerate(groups):
        group_origin = g_idx * (max_per_group * bar_w + group_gap)
        for i, case in enumerate(group_cases):
            x_pos = group_origin + i * bar_w
            lbl = case if case not in seen else None
            container = ax.bar(x_pos, by_case[case], width=bar_w,
                               color=color_for[case], label=lbl,
                               edgecolor='black')
            ax.bar_label(container, fmt=value_fmt, padding=2, fontsize=8)
            seen.add(case)

    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Auto-headroom so the legend (loc='best', usually top region) and the
        # per-bar value annotations don't collide with the tallest bar.
        vals = [by_case[c] for g in groups for c in g if c in by_case]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if vals:
            top = max(vals)
            bot = min(min(vals), 0.0)
            ax.set_ylim(bot, top * 1.4 if top > 0 else top)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
    ax.grid(False)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def _plot_labels(var_name):
    if 'TKE' in var_name:
        return r'k (${m}^2/{s}^2$)', 'Spatial distribution of time averaged and depth averaged turbulent kinetic energy'
    if 'epsilon_turb_avg' in var_name:
        return r'$\epsilon$ (${m}^2/{s}^3$)', 'Spatial distribution of time averaged and depth averaged turbulent dissipation rate'
    if 'epsilon_avg' in var_name:
        return r'$\epsilon$ (${m}^2/{s}^3$)', 'Spatial distribution of time averaged and depth averaged dissipation rate'

    return var_name, var_name

def _read_spanwise_velocity(path):
    """Read a spanwise velocity .dat (# y U_x U_y U_z [alpha] [valid]).
    Returns (y, u, v, w, alpha) — alpha is NaN-filled if absent."""
    ys, us, vs, ws, al = [], [], [], [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = line.split()
        try:
            ys.append(float(p[0])); us.append(float(p[1]))
            vs.append(float(p[2])); ws.append(float(p[3]))
        except (ValueError, IndexError):
            continue
        al.append(float(p[4]) if len(p) > 4 else float('nan'))
    y = np.asarray(ys); o = np.argsort(y)
    return (y[o], np.asarray(us)[o], np.asarray(vs)[o],
            np.asarray(ws)[o], np.asarray(al)[o])


def _spectrum_1d(sig, dy):
    """1D power spectral density of a real signal sampled at spacing dy.
    Returns (kappa, E) with kappa in rad/m. (TODO.md step 5.)"""
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    kappa = np.fft.rfftfreq(n, d=dy) * 2.0 * np.pi
    E = (np.abs(np.fft.rfft(sig)) ** 2) * dy / n
    return kappa, E


def _interp_nan(vals, x):
    """Fill a few NaN points by linear interpolation over x (so the FFT has a
    complete signal). Returns vals unchanged if no NaN or too few good points."""
    vals = np.asarray(vals, dtype=float)
    bad = ~np.isfinite(vals)
    if not bad.any():
        return vals
    good = ~bad
    if good.sum() < 2:
        return vals
    out = vals.copy()
    out[bad] = np.interp(x[bad], x[good], vals[good])
    return out


def _autocorr(sig):
    """Normalized spanwise autocorrelation R(r), positive lags only. The mean
    removed here IS the spanwise average, so this is the autocorrelation of the
    fluctuation u' = u - <u>_y. (TODO.md step 2.) R(0)=1."""
    s = np.asarray(sig, dtype=float)
    s = s - np.mean(s)
    R = np.correlate(s, s, mode='full')
    R = R[s.size - 1:]                    # positive lags
    if R[0] != 0:
        R = R / R[0]
    return R


def _first_zero_l0(R, dy):
    """Characteristic eddy size l0 = first zero crossing of R(r) (linear
    interp), and kappa_l0 = 2*pi/l0. (TODO.md step 2.)"""
    R = np.asarray(R, dtype=float)
    r = np.arange(R.size) * dy
    sc = np.where(np.diff(np.sign(R)))[0]
    if sc.size == 0:
        return np.nan, np.nan
    i = int(sc[0])
    denom = (R[i + 1] - R[i])
    if denom == 0:
        l0 = r[i]
    else:
        l0 = r[i] - R[i] * (r[i + 1] - r[i]) / denom
    kappa_l0 = (2.0 * np.pi / l0) if l0 > 0 else np.nan
    return l0, kappa_l0


def _fit_powerlaw(kappa, E, k_lo, k_hi):
    """Best-fit power-law slope of E(kappa) over [k_lo, k_hi] in log-log, with
    R^2. (TODO.md step 1.) Returns (slope_m, R2, intercept_b, n_pts)."""
    kappa = np.asarray(kappa, dtype=float)
    E = np.asarray(E, dtype=float)
    mask = (kappa > k_lo) & (kappa < k_hi) & (E > 0) & np.isfinite(E)
    if int(mask.sum()) < 3:
        return np.nan, np.nan, np.nan, int(mask.sum())
    lk = np.log10(kappa[mask]); le = np.log10(E[mask])
    m, b = np.polyfit(lk, le, 1)
    fit = m * lk + b
    ss_res = np.sum((le - fit) ** 2)
    ss_tot = np.sum((le - np.mean(le)) ** 2)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return float(m), float(r2), float(b), int(mask.sum())


def spanwise_spectra_plot():
    """Read the spanwise velocity profiles written by
    extract_spanwise_profiles(), compute 1D wavenumber spectra E_uu/E_vv/E_ww
    (FFT along the span), average over the snapshots per probe, and plot each
    probe loglog with -5/3 and -3 reference slopes and the grid cutoff kappa.
    (TODO.md steps 5-7.) Standalone: caller exits after this.

    cfg['spanwise_spectra']['case'] is the 'spanwise' output dir containing one
    folder per probe (e.g. P3_x39_z0.54/) with <label>_t<t>.dat velocity files
    (the *_alphaZ_*/*_alphaX_* files are alpha checks and are ignored here).
    """
    sp = cfg.get('spanwise_spectra') or {}
    case = sp.get('case')
    if not case or not os.path.isdir(case):
        print(f"[spectra] case dir not found: {case}")
        return

    # probe folders, sorted by label
    folders = sorted(
        d for d in (os.path.join(case, e) for e in os.listdir(case))
        if os.path.isdir(d)
    )
    if not folders:
        print(f"[spectra] no probe folders under {case}")
        return

    pat = re.compile(r"(?P<label>[^_/]+)_x(?P<x>[0-9.]+)(?:_z(?P<z>[0-9.]+))?$")
    for folder in folders:
        base = os.path.basename(folder)
        m = pat.match(base)
        label = m.group('label') if m else base
        xloc = m.group('x') if m else '?'

        # velocity snapshot files only (skip alphaZ / alphaX)
        vfiles = sorted(
            p for p in Path(folder).glob("*_t*.dat")
            if 'alphaZ' not in p.name and 'alphaX' not in p.name
        )
        if not vfiles:
            print(f"[spectra] {label}: no velocity files in {folder}")
            continue

        acc = None      # {n, dy, Euu, Evv, Eww, count}
        n_skipped_nan = 0
        for vf in vfiles:
            y, u, v, w, a = _read_spanwise_velocity(str(vf))
            n = y.size
            if n < 8:
                continue
            # Reject broken snapshots (probe returned no data -> NaN U). A few
            # NaN points (e.g. span edges in air) are linearly interpolated.
            nan_frac = float((~np.isfinite(u)).mean())
            if nan_frac > 0.05:
                n_skipped_nan += 1
                continue
            if nan_frac > 0:
                u = _interp_nan(u, y); v = _interp_nan(v, y); w = _interp_nan(w, y)
            dy = (y[-1] - y[0]) / (n - 1)
            ku, Euu = _spectrum_1d(u, dy)
            _,  Evv = _spectrum_1d(v, dy)
            _,  Eww = _spectrum_1d(w, dy)
            # autocorrelations (mean removed = spanwise average -> u' etc.)
            Ruu = _autocorr(u); Rvv = _autocorr(v); Rww = _autocorr(w)
            if acc is None:
                acc = {'n': n, 'dy': dy, 'k': ku,
                       'Euu': np.zeros_like(Euu), 'Evv': np.zeros_like(Evv),
                       'Eww': np.zeros_like(Eww),
                       'Ruu': np.zeros_like(Ruu), 'Rvv': np.zeros_like(Rvv),
                       'Rww': np.zeros_like(Rww), 'count': 0}
            if n != acc['n']:
                print(f"[spectra] {label}: {vf.name} has {n} pts != {acc['n']}; skipped")
                continue
            acc['Euu'] += Euu; acc['Evv'] += Evv; acc['Eww'] += Eww
            acc['Ruu'] += Ruu; acc['Rvv'] += Rvv; acc['Rww'] += Rww
            acc['count'] += 1
            n_dry = int(np.sum(np.isfinite(a) & (a <= 0.1)))
            if n_dry:
                print(f"[spectra] {label}: {vf.name} has {n_dry}/{n} pts with alpha<=0.1 (air)")

        if not acc or acc['count'] == 0:
            if n_skipped_nan:
                print(f"[spectra] {label}: ALL {n_skipped_nan} snapshots are NaN "
                      f"(probe found no data) — re-run extract_spanwise_profiles "
                      f"with the fixed pvpython code, and check the probe z is in water.")
            else:
                print(f"[spectra] {label}: no usable snapshots")
            continue

        k = acc['k']
        dy = acc['dy']
        cnt = acc['count']
        Euu = acc['Euu'] / cnt;  Evv = acc['Evv'] / cnt;  Eww = acc['Eww'] / cnt
        Ruu = acc['Ruu'] / cnt;  Rvv = acc['Rvv'] / cnt;  Rww = acc['Rww'] / cnt
        kappa_cut = np.pi / dy

        # --- Step 2: characteristic eddy size l0 (from E_uu autocorrelation) ---
        l0_uu, kl0_uu = _first_zero_l0(Ruu, dy)
        l0_vv, _ = _first_zero_l0(Rvv, dy)
        l0_ww, _ = _first_zero_l0(Rww, dy)
        # inertial-range lower bound: 2*pi/l0 of the streamwise component
        k_lo = kl0_uu if np.isfinite(kl0_uu) else max(k[1], 2.0 * k[1])

        # --- Step 1: best-fit power-law slope + R^2 over [k_lo, kappa_cut] ---
        comps = [('E_{uu}', Euu), ('E_{vv}', Evv), ('E_{ww}', Eww)]
        fits = {}
        for name, E in comps:
            m, r2, b, npt = _fit_powerlaw(k, E, k_lo, kappa_cut)
            fits[name] = (m, r2, b, npt)
            print(f"[spectra] {label}: {name}  slope m={m:.3f}  R^2={r2:.3f}  "
                  f"(fit over kappa in [{k_lo:.1f},{kappa_cut:.1f}], {npt} pts)")
        print(f"[spectra] {label}: l0(uu)={l0_uu:.3f} m  (kappa_l0={kl0_uu:.1f}); "
              f"l0(vv)={l0_vv:.3f}; l0(ww)={l0_ww:.3f}")

        # --- figure: spectrum (left) + autocorrelation (right) ---
        _, (axE, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

        axE.loglog(k[1:], Euu[1:], 'o', ms=4, color='C0', label=r'$E_{uu}$ (streamwise)')
        axE.loglog(k[1:], Evv[1:], 's', ms=4, mfc='none', color='C1', label=r'$E_{vv}$ (spanwise)')
        axE.loglog(k[1:], Eww[1:], '+', ms=6, color='C2', label=r'$E_{ww}$ (vertical)')

        # best-fit line for E_uu over the inertial range
        m_uu, r2_uu, b_uu, _ = fits['E_{uu}']
        if np.isfinite(m_uu):
            kf = np.logspace(np.log10(k_lo), np.log10(kappa_cut), 50)
            axE.loglog(kf, 10.0 ** (m_uu * np.log10(kf) + b_uu), 'C0-', lw=2,
                       label=rf'fit $E_{{uu}}$: m={m_uu:.2f}, $R^2$={r2_uu:.2f}')

        # reference slopes anchored to E_uu / E_ww at k_lo
        ai = int(np.argmin(np.abs(k - k_lo)))
        ai = max(ai, 1)
        k_ref = np.logspace(np.log10(k[ai]), np.log10(kappa_cut), 50)
        c53 = Euu[ai] / (k[ai] ** (-5.0 / 3.0))
        c3 = Eww[ai] / (k[ai] ** (-3.0))
        axE.loglog(k_ref, c53 * k_ref ** (-5.0 / 3.0), 'b--', lw=1.2, label=r'$\kappa^{-5/3}$')
        axE.loglog(k_ref, c3 * k_ref ** (-3.0), 'k--', lw=1.2, label=r'$\kappa^{-3}$')

        if np.isfinite(kl0_uu):
            axE.axvline(kl0_uu, color='g', ls='-.', lw=1.3,
                        label=rf'$\kappa_{{l_0}}=2\pi/l_0$ ({kl0_uu:.0f})')
        axE.axvline(kappa_cut, color='r', ls=':', lw=1.5,
                    label=r'$\kappa_{cut}=\pi/\Delta y$')

        axE.set_xlabel(r'$\kappa_y$ [rad/m]')
        axE.set_ylabel(r'$E_{ii}(\kappa_y)$ [m$^3$/s$^2$]')
        axE.set_title(f'{label}  (x = {xloc} m)  — {cnt} snapshots')
        axE.legend(loc='best', fontsize=7)
        axE.grid(True, which='both', ls=':', alpha=0.4)

        # autocorrelation R_ii(r)
        r = np.arange(Ruu.size) * dy
        axR.plot(r, Ruu, 'C0-', label=r'$R_{uu}$')
        axR.plot(r, Rvv, 'C1-', label=r'$R_{vv}$')
        axR.plot(r, Rww, 'C2-', label=r'$R_{ww}$')
        axR.axhline(0.0, color='k', lw=0.6)
        if np.isfinite(l0_uu):
            axR.axvline(l0_uu, color='g', ls='-.', lw=1.3,
                        label=rf'$l_0$={l0_uu:.3f} m')
        axR.set_xlabel(r'lag $r$ [m]')
        axR.set_ylabel(r'$R_{ii}(r)$')
        axR.set_title(f'{label}: spanwise autocorrelation')
        axR.legend(loc='best', fontsize=8)
        axR.grid(True, ls=':', alpha=0.4)

        plt.tight_layout()
        out_png = os.path.join(case, f"spectrum_{label}.png")
        try:
            plt.savefig(out_png, dpi=200)
            print(f"[spectra] {label}: averaged {cnt} snapshots -> {out_png}")
        except Exception as e:
            print(f"[spectra] {label}: savefig failed: {e}")
        plt.show()


def main():
    # Spanwise wavenumber-spectra mode: FFT the spanwise velocity profiles,
    # plot per probe, and exit without running the streamwise/bar analysis.
    if (cfg.get('spanwise_spectra') or {}).get('enabled'):
        spanwise_spectra_plot()
        return

    dirs = cfg['base_directory']
    if isinstance(dirs, str):
        dirs = [dirs]
    output_params = cfg.get('output_parameters') or []
    if not output_params:
        raise ValueError("No 'output_parameters' set in INPUT_PARAMETERS")

    seg_idx = cfg.get('case_name_index', 7)
    def _case_of(base_dir):
        return Path(base_dir).parts[seg_idx]

    # Decide whether to use the grouped layout. If CASE_GROUPS is set but NONE
    # of the supplied dirs has a case_name in any group (e.g. a one-off case
    # like 'bar_nourishment_full'), fall back to the single-plot path so the
    # data still gets plotted instead of being silently skipped.
    _known_group_names = set(REFERENCE_CASES) | {c for g in CASE_GROUPS for c in g}
    _matched_any = any(_case_of(d) in _known_group_names for d in dirs)
    use_groups = bool(CASE_GROUPS) and _matched_any
    if CASE_GROUPS and not _matched_any:
        present = sorted({_case_of(d) for d in dirs})
        print(f"[warn] None of the input dirs match CASE_GROUPS/REFERENCE_CASES "
              f"(case_name_index={seg_idx}).")
        print(f"       Dir case names:    {present}")
        print(f"       Known group names: {sorted(_known_group_names)}")
        print(f"       Falling back to a single combined plot.")

    for var_name in output_params:
        plot_ylabel, plot_title = _plot_labels(var_name)
        sw_results = []
        sw_int_results = []
        seen_cases = set()  # dedupe across groups for the bar charts

        if use_groups:
            # One streamwise plot per group: refs + that group's treatments.
            for grp_idx, treatment_cases in enumerate(CASE_GROUPS):
                grp_names = list(REFERENCE_CASES) + list(treatment_cases)
                grp_dirs = [d for d in dirs if _case_of(d) in grp_names]
                if not grp_dirs:
                    continue

                plt.figure(figsize=(8, 5))
                for base_dir in grp_dirs:
                    data_dir = os.path.join(base_dir, cfg["variable"])
                    case_name, sw_val, sw_int_val = epsilon_plot(
                        data_dir, cfg, base_dir, var_name)

                    if case_name not in seen_cases:
                        if sw_val is not None:
                            sw_results.append((case_name, sw_val))
                        if sw_int_val is not None:
                            sw_int_results.append((case_name, sw_int_val))
                        seen_cases.add(case_name)

                grp_label = " + ".join(treatment_cases)
                plt.xlabel('X (m)')
                plt.ylabel(plot_ylabel)
                plt.title(f'{plot_title}\n[group {grp_idx + 1}: {grp_label}]')
                if 'eps' in var_name:
                    plt.yscale("log")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        else:
            # Legacy single-plot path: all dirs on one figure.
            plt.figure(figsize=(8, 5))
            for base_dir in dirs:
                data_dir = os.path.join(base_dir, cfg["variable"])
                case_name, sw_val, sw_int_val = epsilon_plot(
                    data_dir, cfg, base_dir, var_name)
                if case_name not in seen_cases:
                    if sw_val is not None:
                        sw_results.append((case_name, sw_val))
                    if sw_int_val is not None:
                        sw_int_results.append((case_name, sw_int_val))
                    seen_cases.add(case_name)

            plt.xlabel('X (m)')
            plt.ylabel(plot_ylabel)
            plt.title(plot_title)
            if 'eps' in var_name:
                plt.yscale("log")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        print("sw_val", sw_results)
        print("sw_int", sw_int_results)

        # Streamwise-averaged grouped bar chart for this parameter
        if cfg.get('streamwise_avg') and sw_results:
            xr = cfg['streamwise_avg_range']
            _grouped_bar_chart(
                sw_results,
                ylabel=plot_ylabel,
                title=f'Streamwise averaged ({xr[0]}m - {xr[1]}m) {var_name}',
            )

        # Streamwise-integrated grouped bar chart (per-case x range)
        if cfg.get('streamwise_int') and sw_int_results:
            int_ylabel, _ = _plot_labels(var_name)
            _grouped_bar_chart(
                sw_int_results,
                ylabel=rf'$\int$ {int_ylabel} dX',
                title=f'Streamwise integral of {var_name}',
            )

def time_avg(pairs, var_name, window=None, output_params=None):

    if window is not None:
        tmin, tmax = window[0], window[1]
    else:
        tmin, tmax = pairs[0][0], pairs[-1][0]

    # The depth-average writes a slightly DIFFERENT X grid each timestep (its
    # grid follows the per-timestep water bounds), so keying by exact float X
    # gives ~one sample per X and no real time-average. Bin X onto a common
    # grid of width cfg['x_bin'] so all timesteps' values at nearby X are
    # grouped and averaged together.
    x_bin = float(cfg.get('x_bin', 0.05) or 0.05)

    def _xkey(x_pos):
        return round(round(float(x_pos) / x_bin) * x_bin, 6)

    # Collect all data with X positions
    data_dict = {}  # dict of {x_bin_center: [values_across_times_and_subgrid]}

    for ti, p in pairs:

        if ti >= tmin and ti <= tmax:
            cols = load_cols(p, output_params=output_params)

            eps_turb = np.array(cols[var_name])
            x = np.array(cols["X"])

            # Store values by binned X position
            for x_pos, eps_val in zip(x, eps_turb):
                k = _xkey(x_pos)
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append(eps_val)
    
    # Convert to sorted arrays
    x_positions = sorted(data_dict.keys())
    x_array = np.array(x_positions)
    
    # Calculate average for each X position
    eps_avg = []
    for x_pos in x_positions:
        values = np.array(data_dict[x_pos], dtype=float)
        # Drop NaN/inf, capped blow-ups, and (optionally) zeros from BOTH the
        # sum and the count by setting them to NaN, then nanmean.
        values = np.where(_bad_mask(values), np.nan, values)
        if np.all(np.isnan(values)):
            avg_val = np.nan
        else:
            avg_val = np.nanmean(values)
        eps_avg.append(avg_val)
    
    eps_avg = np.array(eps_avg)
    
    return eps_avg, x_array
    
def streamwise_average(eps_avg, x_array, x_range):
    """Average parameter values over the given streamwise X range.

    If cfg['exclude_zero'] is True, drops (x, value) pairs where value==0
    (or NaN) before computing the mean, so the denominator shrinks
    accordingly.
    """
    mask = (x_array >= x_range[0]) & (x_array <= x_range[1])
    vals = np.asarray(eps_avg, dtype=float)[mask]
    vals = vals[~_bad_mask(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))

def streamwise_integrate(eps_avg, x_array, x_range=None):
    """Trapezoidal integral over X. If x_range is None, use the case's full available X extent.

    If cfg['exclude_zero'] is True, (x, value) pairs with value==0 (or NaN)
    are dropped before integration. The remaining points are still
    integrated in X-order with their actual (possibly non-uniform) spacing.
    """
    x_array = np.asarray(x_array, dtype=float)
    eps_avg = np.asarray(eps_avg, dtype=float)
    if x_range is not None:
        mask = (x_array >= x_range[0]) & (x_array <= x_range[1])
        x_array = x_array[mask]
        eps_avg = eps_avg[mask]
    keep = ~_bad_mask(eps_avg)
    x_array = x_array[keep]
    eps_avg = eps_avg[keep]
    if len(x_array) < 2:
        return np.nan
    order = np.argsort(x_array)
    return float(np.trapz(eps_avg[order], x_array[order]))

def streamwise_cumulative_average(eps_avg, x_array, x_range=None):
    """Running (cumulative) average vs X.

    At each kept X[i], returns the mean of eps_avg[0..i] after filtering out
    NaN and (if cfg['exclude_zero']) zero values. The result therefore has
    the same length as the *kept* X array, which may be shorter than the
    input if zeros/NaNs were present.

    Returns (x_kept, cum_avg). If everything is filtered out, returns two
    empty arrays.
    """
    x_array = np.asarray(x_array, dtype=float)
    eps_avg = np.asarray(eps_avg, dtype=float)
    if x_range is not None:
        mask = (x_array >= x_range[0]) & (x_array <= x_range[1])
        x_array = x_array[mask]
        eps_avg = eps_avg[mask]
    keep = ~_bad_mask(eps_avg)
    x_array = x_array[keep]
    eps_avg = eps_avg[keep]
    if eps_avg.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(x_array)
    x_sorted = x_array[order]
    e_sorted = eps_avg[order]
    cum_sum = np.cumsum(e_sorted)
    cum_count = np.arange(1, e_sorted.size + 1, dtype=float)
    cum_avg = cum_sum / cum_count
    return x_sorted, cum_avg

def cleanup(eps,x):
    eps = np.asarray(eps, dtype=float)
    x = np.asarray(x, dtype=float)
    # Keep only finite, in-cap, >=1e-9 values (drops NaN/inf and capped blow-ups).
    keep = (~_bad_mask(eps)) & (eps >= 10 ** -9)
    x_array = x[keep]
    eps_avg = eps[keep]
    mask = np.where((x_array < 4) & (eps_avg >= 10**-4))
    x_array = np.delete(x_array, mask)
    eps_avg = np.delete(eps_avg, mask)
    return eps_avg, x_array, mask
def load_cols(path, output_params=None):
    # Load data, skipping comment lines
    data = np.loadtxt(path, comments='#')

    # Read header from comment line
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                header_text = line[1:].strip()
                break
        else:
            raise ValueError(f"No header found in {path}")

    # Remove anything in parentheses (units)
    header_text = re.sub(r'\([^)]*\)', '', header_text)

    # Split by whitespace and remove empty strings
    headers = [h for h in header_text.split() if h]

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Verify column count matches
    if len(headers) != data.shape[1]:
        raise ValueError(f"Header has {len(headers)} columns {headers} but data has {data.shape[1]} columns in {path}")

    # First column is always X coords; remaining columns optionally filtered by output_params
    if output_params is None:
        selected = headers
    else:
        missing = [p for p in output_params if p not in headers[1:]]
        if missing:
            raise ValueError(f"output_parameters {missing} not in header {headers[1:]} of {path}")
        selected = [headers[0]] + [h for h in headers[1:] if h in output_params]

    # Create dictionary using original column index from headers
    cols = {}
    for h in selected:
        i = headers.index(h)
        cols[h] = data[:, i].tolist()

    return cols

def epsilon_plot(path, cfg, base_dir, var_name):
    folder = Path(path)
    pat = re.compile(r"_([0-9.]+)\.dat$")
    print("folder:", folder)
    pairs = []
    for p in folder.glob("*.dat"):

        m = pat.search(p.name)
        if m:
            pairs.append((float(m.group(1)), p))
        else:
            print(f"Not matched: {p.name}")
    pairs.sort(key=lambda tp: tp[0])  # sort by time

    if not pairs:
        raise FileNotFoundError("No matching .dat files found.")

    _parts = Path(base_dir).parts
    _seg = cfg.get('case_name_index', 7)
    try:
        case_name = _parts[_seg]
    except IndexError:
        raise ValueError(
            f"case_name_index={_seg} is out of range for path with "
            f"{len(_parts)} segments: {base_dir}"
        )

    # Index of this case in base_directory == index of its CSV in csv_list.
    try:
        case_index = list(cfg.get('base_directory') or []).index(base_dir)
    except ValueError:
        case_index = None

    all_eps = []
    final_eps = np.array([])
    final_x = np.array([])
    line_color = None   # color of THIS case's parameter line; reused for profile
    if cfg.get('window'):
        t_list = cfg['window_periods']
        for i in range(1, len(t_list)):
            eps_avg, x_array = time_avg(pairs, var_name, window=[t_list[0], t_list[i]], output_params=cfg.get('output_parameters'))
            eps_avg, x_array, mask = cleanup(eps_avg, x_array)
            x_array, eps_avg = _apply_plot_filters(x_array, eps_avg)
            if len(eps_avg) > 0:
                _ln, = plt.plot(x_array, _smooth(eps_avg), linestyle='-',
                                label=f'Time Average Window-t={t_list[0]}s-{t_list[i]}s')
                line_color = _ln.get_color()
                all_eps.extend(eps_avg)
                final_eps = eps_avg
                final_x = x_array
    else:
        t_start = cfg['t_start']
        t_end = cfg['t_end']
        eps_avg, x_array = time_avg(pairs, var_name, window=[t_start, t_end], output_params=cfg.get('output_parameters'))
        eps_avg, x_array, mask = cleanup(eps_avg, x_array)
        x_array, eps_avg = _apply_plot_filters(x_array, eps_avg)
        if len(eps_avg) > 0:
            _ln, = plt.plot(x_array, _smooth(eps_avg), linestyle='-',
                            label=case_name)
            line_color = _ln.get_color()
            all_eps.extend(eps_avg)
            final_eps = eps_avg
            final_x = x_array

    # Overlay the beach profile (dotted, same color) on a secondary Y-axis.
    if line_color is not None:
        _overlay_profile(case_index, line_color)

    if cfg.get('streamwise_cum_avg') and len(final_eps) > 0:
        x_cum, cum_avg = streamwise_cumulative_average(final_eps, final_x)
        if cum_avg.size > 0:
            plt.plot(x_cum, cum_avg, linestyle='--', linewidth=1.2,
                     label=f'{case_name} cum.avg (final={cum_avg[-1]:.3g})')
            print(f"[cum_avg] {case_name} {var_name}: final={cum_avg[-1]:.6g} "
                  f"over {cum_avg.size} kept points")

    sw_val = None
    if cfg.get('streamwise_avg') and cfg.get('streamwise_avg_range') and len(final_eps) > 0:
        sw_val = streamwise_average(final_eps, final_x, cfg['streamwise_avg_range'])

    sw_int_val = None
    if cfg.get('streamwise_int') and len(final_eps) > 0:
        sw_int_val = streamwise_integrate(final_eps, final_x, x_range=None)
        print(f"[streamwise_int] {case_name} {var_name}: integral={sw_int_val:.6g} over X=[{final_x.min():.3f},{final_x.max():.3f}]")

    return case_name, sw_val, sw_int_val

if __name__ == "__main__":
    main()