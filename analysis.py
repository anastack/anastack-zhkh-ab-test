"""Reproduce the housing-payment A/B test from the case workbook."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm

GROUPS = ["A", "B"]
STEPS = ["step1_opened", "step2_entered", "step3_confirmed", "step4_success"]
START = pd.Timestamp("2024-04-01")
END = pd.Timestamp("2024-05-01")
PALETTE = ["#8B98A9", "#2571C9"]


def prepare(workbook):
    users, payments = (pd.read_excel(workbook, sheet_name=name)
                       for name in ["Пользователи", "Платежи"])
    users["city"] = users["city"].fillna("Неизвестно")
    users["age"] = users["age"].where(users["age"].between(0, 120))
    if users["user_id"].duplicated().any() or payments["payment_id"].duplicated().any():
        raise ValueError("Expected unique user_id and payment_id in the source tables")
    for step in STEPS:
        payments[step] = pd.to_datetime(payments[step], errors="raise")
    merged = payments.rename(columns={"group": "payment_group"}).merge(
        users, on="user_id", how="left", validate="many_to_one"
    )
    valid = merged["user_id"].notna() & merged["group"].isin(GROUPS)
    conflict = valid & merged["payment_group"].ne(merged["group"])
    in_period = merged[STEPS[0]].ge(START) & merged[STEPS[0]].lt(END)
    chronology = pd.Series(True, index=merged.index)
    for before, after in zip(STEPS, STEPS[1:]):
        chronology &= merged[after].isna() | merged[after].ge(merged[before])
    clean = merged.loc[valid & ~conflict & in_period & chronology].copy()
    clean["success"] = clean[STEPS[-1]].notna()
    assigned = users["group"].value_counts().reindex(GROUPS, fill_value=0)
    audit = {
        "users": len(users), "payments": len(payments), "included": len(clean),
        "unknown_user_or_group": int((~valid).sum()),
        "group_conflicts": int(conflict.sum()),
        "outside_window_or_missing_open": int((~in_period).sum()),
        "invalid_step_order": int((~chronology).sum()),
        "assigned_by_group": assigned.to_dict(),
        "assignment_srm_p_value": float(chisquare(assigned).pvalue),
    }
    return clean, merged.loc[conflict, "user_id"], audit


def estimate(clean, iterations=30_000):
    clients = clean.groupby(["group", "user_id"]).agg(
        attempts=("payment_id", "size"), successes=("success", "sum")
    )
    summary, simulations = {}, {}
    rng = np.random.default_rng(42)
    for group in GROUPS:
        sample = clients.loc[group]
        successes = sample["successes"].to_numpy()
        attempts = sample["attempts"].to_numpy()
        rate = successes.sum() / attempts.sum()
        residual = successes - rate * attempts
        # Variance of a ratio of two means over independent client clusters.
        variance = residual.var(ddof=1) / len(sample) / attempts.mean() ** 2
        funnel = clean.loc[clean["group"].eq(group), STEPS].notna().sum()
        summary[group] = {
            "clients_opened": len(sample), "attempts": int(attempts.sum()),
            "successes": int(successes.sum()), "payment_conversion": float(rate),
            "cluster_variance": float(variance),
            "clients_with_success": int((successes > 0).sum()),
            "client_conversion": float((successes > 0).mean()),
            "funnel_counts": funnel.astype(int).tolist(),
        }
        draws = []
        for start in range(0, iterations, 1000):
            idx = rng.integers(0, len(sample), size=(min(1000, iterations - start), len(sample)))
            draws.append(successes[idx].sum(axis=1) / attempts[idx].sum(axis=1))
        simulations[group] = np.concatenate(draws)
    a, b = (summary[g] for g in GROUPS)
    effect = b["payment_conversion"] - a["payment_conversion"]
    se = np.sqrt(a["cluster_variance"] + b["cluster_variance"])
    z = effect / se
    return {
        "groups": summary, "effect": effect, "standard_error": float(se),
        "z": float(z), "p_value_two_sided": float(2 * norm.sf(abs(z))),
        "delta_ci_95": [float(effect - norm.ppf(.975) * se), float(effect + norm.ppf(.975) * se)],
        "bootstrap_ci_95": np.quantile(simulations["B"] - simulations["A"], [.025, .975]).tolist(),
        "bootstrap_iterations": iterations, "bootstrap_seed": 42,
    }


def charts(out, result, segments):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})
    a, b = (result["groups"][g] for g in GROUPS)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for idx, (g, stats, col) in enumerate(zip(GROUPS, [a, b], PALETTE)):
        counts = np.array(stats["funnel_counts"])
        values = 100 * counts / stats["attempts"]
        bars = ax.bar(np.arange(4) + (idx - .5) * .35, values, width=.35, color=col, label=g)
        ax.bar_label(bars, [f"{v:.1f}%\n{n:,}" for v, n in zip(values, counts)], padding=3, fontsize=9)
    ax.set(ylim=(0, 119), ylabel="% от открытий формы", title="Воронка оплаты ЖКХ")
    ax.set_xticks(range(4), ["Открыли", "Ввели реквизиты", "Подтвердили", "Оплатили"])
    ax.grid(axis="y", alpha=.18)
    ax.legend()
    fig.savefig(out / "funnel.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 2.6), layout="constrained")
    effect = result["effect"] * 100
    low, high = np.array(result["delta_ci_95"]) * 100
    ax.errorbar(effect, 0, xerr=[[effect - low], [high - effect]], fmt="o", capsize=7, color=PALETTE[1])
    ax.axvline(0, color="#888", linestyle="--")
    ax.set(xlim=(-1, 21), ylim=(-.4, .4), yticks=[], xlabel="B − A, п.п.",
           title=f"Прирост {effect:+.2f} п.п.  |  95% ДИ [{low:+.2f}; {high:+.2f}]")
    fig.savefig(out / "effect_ci.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    shown = segments.loc[segments.index.get_level_values("segment") != "Неизвестно"]
    bars = ax.barh(shown.index.get_level_values("segment"), shown["effect_pp"], color=PALETTE[1])
    ax.bar_label(bars, fmt="%+.2f", padding=4)
    ax.invert_yaxis()
    ax.set(xlabel="B − A, п.п.", title="Эффект по платформам и городам", xlim=(0, 22))
    fig.savefig(out / "segments.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    clean, conflicts, audit = prepare(args.workbook)
    result = estimate(clean)
    result["audit"] = audit
    sensitivity = clean.loc[~clean["user_id"].isin(conflicts)].groupby("group")["success"].mean()
    result["effect_without_conflict_clients"] = float(sensitivity["B"] - sensitivity["A"])
    tables = {}
    for dimension in ["device_type", "city"]:
        grouped = clean.groupby([dimension, "group"])["success"].agg(["size", "mean"]).unstack()
        grouped.columns = ["_".join(col) for col in grouped.columns]
        grouped["effect_pp"] = (grouped["mean_B"] - grouped["mean_A"]) * 100
        tables[dimension] = grouped
    segments = pd.concat(tables, names=["dimension", "segment"])
    segments.to_csv(args.out / "segments.csv")
    (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    charts(args.out, result, segments)
    print(f"Effect: {result['effect'] * 100:+.2f} pp; p = {result['p_value_two_sided']:.3g}")


if __name__ == "__main__":
    main()
