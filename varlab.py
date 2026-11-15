"""VaR Model Validation Lab: five Value-at-Risk / Expected Shortfall models backtested
on daily US equity market returns since 1926 (Kenneth French Data Library).

Models (1-day horizon, forecasts use only data up to the previous day):
  HS      historical simulation, 250-day window (Basel minimum)
  Normal  variance-covariance, 250-day window
  EWMA    RiskMetrics, lambda = 0.94, normal
  GARCH-t GARCH(1,1) with Student-t errors, refit every 21 days on 1,000 days
  FHS     filtered historical simulation on the GARCH-t standardised residuals

Backtests: Kupiec (coverage), Christoffersen (independence, conditional coverage),
Basel traffic-light zones, a McNeil-Frey-type ES test, and crisis-window exceedances.

    python varlab.py     -> results/
"""
import io
import pathlib
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
WINDOW, GARCH_WINDOW, REFIT = 250, 1000, 21
LEVELS = (0.99, 0.975)
CRISES = {
    "Great Depression (Nov 1929-Jul 1932)": ("1929-11-12", "1932-07-31"),  # forecasts start 1929-11-12
    "Black Monday (Oct-Dec 1987)": ("1987-10-01", "1987-12-31"),
    "Global Financial Crisis (Sep 2008-Mar 2009)": ("2008-09-01", "2009-03-31"),
    "COVID crash (Feb-Apr 2020)": ("2020-02-15", "2020-04-30"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
}


# ---------- data ----------
def load_returns() -> pd.Series:
    """Daily US market return in % (Mkt-RF + RF), from the Fama-French daily factor file."""
    raw = zipfile.ZipFile(DATA / "ff_daily.zip").read("F-F_Research_Data_Factors_daily.csv").decode("latin1")
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].strip())  # table ends at first blank line
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    return (df["Mkt-RF"] + df["RF"]).rename("ret")


# ---------- risk formulas ----------
def normal_var_es(mu, sigma, a):
    z = stats.norm.ppf(a)
    return -mu + sigma * z, -mu + sigma * stats.norm.pdf(z) / (1 - a)


def t_var_es(mu, sigma, nu, a):
    """VaR/ES for loss = -(mu + sigma*Z), Z a Student-t scaled to unit variance."""
    s = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(a, nu)
    es_std = stats.t.pdf(q, nu) / (1 - a) * (nu + q ** 2) / (nu - 1)
    return -mu + sigma * q * s, -mu + sigma * es_std * s


def hs_var_es(losses, a):
    var = np.quantile(losses, a)
    return var, losses[losses >= var].mean()


# ---------- models ----------
def forecast(r: pd.Series) -> pd.DataFrame:
    from arch import arch_model

    x = r.to_numpy()
    n, first = len(x), GARCH_WINDOW
    out = {f"{m}_{k}{int(a * 1000)}": np.full(n, np.nan)
           for m in ("HS", "Normal", "EWMA", "GARCH-t", "FHS") for k in ("var", "es") for a in LEVELS}

    ewma = np.var(x[:WINDOW])
    for t in range(WINDOW, n):  # EWMA variance for day t uses returns up to t-1
        ewma = 0.94 * ewma + 0.06 * x[t - 1] ** 2
        if t < first:
            continue
        w = x[t - WINDOW:t]
        for a in LEVELS:
            tag = int(a * 1000)
            out[f"HS_var{tag}"][t], out[f"HS_es{tag}"][t] = hs_var_es(-w, a)
            out[f"Normal_var{tag}"][t], out[f"Normal_es{tag}"][t] = normal_var_es(w.mean(), w.std(ddof=1), a)
            out[f"EWMA_var{tag}"][t], out[f"EWMA_es{tag}"][t] = normal_var_es(0.0, np.sqrt(ewma), a)

    for k in range(first, n, REFIT):
        fit = arch_model(x[k - GARCH_WINDOW:k], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        mu, om, al, be, nu = (fit.params[p] for p in ("mu", "omega", "alpha[1]", "beta[1]", "nu"))
        z = fit.std_resid
        sig2 = fit.conditional_volatility[-1] ** 2
        eps = x[k - 1] - mu
        for t in range(k, min(k + REFIT, n)):
            sig2 = om + al * eps ** 2 + be * sig2  # one-step-ahead variance with frozen parameters
            sig = np.sqrt(sig2)
            for a in LEVELS:
                tag = int(a * 1000)
                out[f"GARCH-t_var{tag}"][t], out[f"GARCH-t_es{tag}"][t] = t_var_es(mu, sig, nu, a)
                zv, ze = hs_var_es(-z, a)
                out[f"FHS_var{tag}"][t], out[f"FHS_es{tag}"][t] = -mu + sig * zv, -mu + sig * ze
            eps = x[t] - mu
    return pd.DataFrame(out, index=r.index).iloc[first:]


# ---------- backtests ----------
def kupiec(hits: np.ndarray, a: float) -> float:
    n, x, p = len(hits), int(hits.sum()), 1 - a
    if x in (0, n):
        ll1 = 0.0
    else:
        ll1 = (n - x) * np.log(1 - x / n) + x * np.log(x / n)
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    return stats.chi2.sf(-2 * (ll0 - ll1), 1)


def christoffersen(hits: np.ndarray) -> float:
    """p-value of the independence test (are exceedances clustered?)."""
    h0, h1 = hits[:-1], hits[1:]
    n00, n01 = np.sum((h0 == 0) & (h1 == 0)), np.sum((h0 == 0) & (h1 == 1))
    n10, n11 = np.sum((h0 == 1) & (h1 == 0)), np.sum((h0 == 1) & (h1 == 1))
    p01, p11, p = n01 / (n00 + n01), n11 / max(n10 + n11, 1), (n01 + n11) / len(h1)

    def ll(q, a, b):  # a "stays", b "switches", log-likelihood with 0*log(0) = 0
        return (a * np.log(1 - q) if a else 0) + (b * np.log(q) if b else 0)

    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return stats.chi2.sf(lr, 1)


def traffic_light(hits99: pd.Series) -> dict:
    """Share of days whose trailing-250-day VaR99 exceedance count is green (<=4), yellow (5-9), red (>=10)."""
    c = hits99.rolling(250).sum().dropna()
    return {"green %": 100 * (c <= 4).mean(), "yellow %": 100 * ((c >= 5) & (c <= 9)).mean(), "red %": 100 * (c >= 10).mean()}


def es_test(loss: np.ndarray, var: np.ndarray, es: np.ndarray):
    """McNeil-Frey-type check on VaR exceedance days: is loss - ES centred on zero? (one-sided t-test, no bootstrap)"""
    e = (loss - es)[loss > var]
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))
    return (loss[loss > var] / es[loss > var]).mean(), stats.t.sf(t, len(e) - 1)


# ---------- run ----------
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(exist_ok=True)
    r = load_returns()
    f = forecast(r)
    loss = -r.loc[f.index]
    models = ("HS", "Normal", "EWMA", "GARCH-t", "FHS")

    rows = []
    for m in models:
        row = {"model": m}
        for a in LEVELS:
            tag = int(a * 1000)
            hits = (loss > f[f"{m}_var{tag}"]).astype(int).to_numpy()
            row[f"exceed% {a:.1%}"] = 100 * hits.mean()
            row[f"Kupiec p {a:.1%}"] = kupiec(hits, a)
            row[f"Christoffersen p {a:.1%}"] = christoffersen(hits)
        row.update(traffic_light((loss > f[f"{m}_var990"]).astype(int)))
        ratio, p = es_test(loss.to_numpy(), f[f"{m}_var975"].to_numpy(), f[f"{m}_es975"].to_numpy())
        row["ES ratio 97.5%"], row["ES test p"] = ratio, p
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model")
    summary.round(4).to_csv(RESULTS / "backtest_summary.csv")

    crisis = pd.DataFrame({name: {m: int((loss.loc[s:e] > f[f"{m}_var990"].loc[s:e]).sum()) for m in models}
                           for name, (s, e) in CRISES.items()}).T
    crisis["trading days"] = [len(loss.loc[s:e]) for s, e in CRISES.values()]
    crisis["expected at 1%"] = (crisis["trading days"] * 0.01).round(1)
    crisis.to_csv(RESULTS / "crisis_exceedances.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (name, (s, e)) in zip(axes, [("Global Financial Crisis", ("2008-07-01", "2009-06-30")),
                                         ("COVID crash", ("2020-01-15", "2020-06-30"))]):
        ax.bar(loss.loc[s:e].index, loss.loc[s:e], color="#9aa5b1", width=1.5, label="Daily loss (%)")
        for m, c in (("HS", "#d35400"), ("GARCH-t", "#1f4e79")):
            ax.plot(f.loc[s:e, f"{m}_var990"], color=c, lw=1.4, label=f"{m} 99% VaR")
        ax.set_title(name); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "var_in_crises.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, summary["exceed% 99.0%"], color="#1f4e79")
    ax.axhline(1.0, color="#b03a2e", ls="--", label="Target 1%")
    ax.set_ylabel("Exceedance rate (%)"); ax.set_title(f"99% VaR exceedances, {f.index[0]:%Y}-{f.index[-1]:%Y}")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "exceedance_rates.png", dpi=150); plt.close(fig)

    text = (f"# Results (generated by `python varlab.py`)\n\nOut-of-sample period: {f.index[0]:%Y-%m-%d} to "
            f"{f.index[-1]:%Y-%m-%d} ({len(f):,} trading days).\n\n## Backtest summary\n```\n"
            f"{summary.round(3).to_string()}\n```\n\n## 99% VaR exceedances in crisis windows\n```\n{crisis.to_string()}\n```\n")
    (RESULTS / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
