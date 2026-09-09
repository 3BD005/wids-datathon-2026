import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HORIZONS = [12, 24, 48, 72]
SEEDS = [0, 1, 2, 3, 4]
NFOLDS = 5

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "wids datathon dataset"))
ap.add_argument("--out", default=HERE)
a = ap.parse_args()

train = pd.read_csv(os.path.join(a.data, "train.csv"))
test = pd.read_csv(os.path.join(a.data, "test.csv"))
sub = pd.read_csv(os.path.join(a.data, "sample_submission.csv"))
feats = [c for c in test.columns if c != "event_id"]

print("=" * 60)
print("EDA")
print("=" * 60)
print("train", train.shape, " test", test.shape, " features", len(feats))
print("missing train:", int(train.isna().sum().sum()), " missing test:", int(test.isna().sum().sum()))
print("duplicate ids:", int(train["event_id"].duplicated().sum()))

t = train["time_to_hit_hours"]
e = train["event"]
print("\nhit within 72h : %3d  (%.1f%%)" % (e.sum(), 100 * e.mean()))
print("censored       : %3d  (%.1f%%)" % ((1 - e).sum(), 100 * (1 - e).mean()))
print("\ntime_to_hit_hours by outcome:")
print(train.groupby("event")["time_to_hit_hours"].describe()[["mean", "50%", "min", "max"]].round(2).to_string())

Y = np.zeros((len(train), len(HORIZONS)), dtype=int)
print("\nhorizon  positives   rate   censored_before")
for i, h in enumerate(HORIZONS):
    Y[:, i] = ((e == 1) & (t <= h)).astype(int)
    print("   %2dh     %3d      %.3f        %3d" % (h, Y[:, i].sum(), Y[:, i].mean(), ((e == 0) & (t < h)).sum()))
print("\n%d of %d hits happen inside 12h" % (((e == 1) & (t <= 12)).sum(), e.sum()))

uni = []
for c in feats:
    v = train[c].fillna(train[c].median())
    if v.nunique() > 1:
        uni.append((c, roc_auc_score(Y[:, 1], v)))
uni = pd.DataFrame(uni, columns=["feature", "auc"])
uni["power"] = (uni["auc"] - 0.5).abs()
print("\ntop 10 single features (AUC vs 24h label):")
print(uni.sort_values("power", ascending=False).head(10).round(3).to_string(index=False))

key = ["dist_min_ci_0_5h", "closing_speed_m_per_h", "alignment_abs", "centroid_speed_m_per_h"]
print("\nmean by outcome:")
print(train.groupby("event")[key].mean().round(2).to_string())

plotdir = os.path.join(a.out, "plots")
os.makedirs(plotdir, exist_ok=True)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

ax[0, 0].bar([str(h) + "h" for h in HORIZONS], Y.sum(axis=0), color="#c0392b")
ax[0, 0].set_title("Positives per horizon")
ax[0, 0].set_ylabel("count")

ax[0, 1].hist([t[e == 1], t[e == 0]], bins=20, stacked=True, color=["#c0392b", "#95a5a6"], label=["hit", "censored"])
ax[0, 1].set_title("Observed time by outcome")
ax[0, 1].set_xlabel("time_to_hit_hours")
ax[0, 1].legend()

grid = np.linspace(0, 72, 145)
ax[1, 0].step(grid, [1 - ((e == 1) & (t <= g)).mean() for g in grid], color="#2c3e50")
ax[1, 0].set_title("Survival curve")
ax[1, 0].set_xlabel("hours")
ax[1, 0].set_ylabel("fraction not yet hit")
ax[1, 0].grid(alpha=0.3)

for v, c, lab in [(1, "#c0392b", "hit"), (0, "#95a5a6", "censored")]:
    ax[1, 1].scatter(
        train.loc[e == v, "dist_min_ci_0_5h"] / 1000,
        train.loc[e == v, "closing_speed_m_per_h"],
        s=22, c=c, alpha=0.75, label=lab, edgecolors="none",
    )
ax[1, 1].set_xscale("symlog")
ax[1, 1].set_title("Distance vs closing speed")
ax[1, 1].set_xlabel("min distance (km, symlog)")
ax[1, 1].set_ylabel("closing speed (m/h)")
ax[1, 1].legend()
ax[1, 1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(plotdir, "01_overview.png"), dpi=110)
plt.close(fig)

top = uni.sort_values("power").tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top["feature"], top["auc"] - 0.5, color=np.where(top["auc"] > 0.5, "#c0392b", "#2980b9"))
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Single-feature AUC - 0.5 (24h label)")
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "02_univariate.png"), dpi=110)
plt.close(fig)

cols = uni.sort_values("power", ascending=False).head(15)["feature"].tolist()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(train[cols].corr().values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=7)
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(cols, fontsize=7)
ax.set_title("Correlation among top features")
fig.colorbar(im, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "03_correlation.png"), dpi=110)
plt.close(fig)

print("\nplots saved to", plotdir)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


def fe(d):
    x = d.copy()
    dist = x["dist_min_ci_0_5h"]
    close = x["closing_speed_m_per_h"].clip(lower=0)
    radial = x["radial_growth_rate_m_per_h"].clip(lower=0)
    cent = x["centroid_speed_m_per_h"].clip(lower=0)
    align = x["alignment_abs"]
    x["log_dist"] = np.log1p(dist.clip(lower=0))
    x["eta_close"] = np.log1p(dist / close.replace(0, np.nan))
    x["eta_all"] = np.log1p(dist / (close + radial + cent).replace(0, np.nan))
    x["eta_align"] = np.log1p(dist / (close * align).replace(0, np.nan))
    x["reach72"] = ((close + radial) * 72 >= dist).astype(int)
    x["speed_align"] = x["closing_speed_m_per_h"] * x["alignment_cos"]
    x["risk"] = (close + radial) * align / x["log_dist"].replace(0, np.nan)
    x["advance_ratio"] = x["projected_advance_m"] / dist.replace(0, np.nan)
    x["hour_sin"] = np.sin(2 * np.pi * x["event_start_hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["event_start_hour"] / 24)
    return x.replace([np.inf, -np.inf], np.nan)


X = fe(train[feats])
Xte = fe(test[feats])
new = [c for c in X.columns if c not in feats]
print("added %d features: %s" % (len(new), ", ".join(new)))
print("total features:", X.shape[1])

check = pd.DataFrame([(c, roc_auc_score(Y[:, 1], X[c].fillna(X[c].median()))) for c in new], columns=["feature", "auc"])
check["power"] = (check["auc"] - 0.5).abs()
print("\nnew features ranked:")
print(check.sort_values("power", ascending=False).round(3).to_string(index=False))

srt = check.sort_values("power")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(srt["feature"], srt["auc"] - 0.5, color=np.where(srt["auc"] > 0.5, "#16a085", "#2980b9"))
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Engineered features: AUC - 0.5 (24h label)")
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "04_new_features.png"), dpi=110)
plt.close(fig)

print("\n" + "=" * 60)
print("XGBOOST")
print("=" * 60)

params = dict(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.6,
    reg_lambda=2.0,
    gamma=0.05,
    eval_metric="auc",
    early_stopping_rounds=100,
    n_jobs=4,
)

oof = np.zeros((len(X), len(HORIZONS)))
pred = np.zeros((len(Xte), len(HORIZONS)))
gain = np.zeros(X.shape[1])

for i, h in enumerate(HORIZONS):
    y = Y[:, i]
    for s in SEEDS:
        for tr, va in StratifiedKFold(NFOLDS, shuffle=True, random_state=s).split(X, y):
            m = XGBClassifier(random_state=s, **params)
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
            oof[va, i] += m.predict_proba(X.iloc[va])[:, 1] / len(SEEDS)
            pred[:, i] += m.predict_proba(Xte)[:, 1] / (len(SEEDS) * NFOLDS)
            if h == 24:
                gain += m.feature_importances_ / (len(SEEDS) * NFOLDS)
    print("h=%2d  positives %3d  OOF AUC %.5f" % (h, y.sum(), roc_auc_score(y, oof[:, i])))

order = np.argsort(gain)[-20:]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([X.columns[j] for j in order], gain[order], color="#8e44ad")
ax.set_title("XGBoost feature importance (24h model)")
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "05_importance.png"), dpi=110)
plt.close(fig)

oof = np.maximum.accumulate(oof, axis=1)
pred = np.maximum.accumulate(pred, axis=1)
aucs = [roc_auc_score(Y[:, i], oof[:, i]) for i in range(len(HORIZONS))]

print("")
print("CV mean-AUC: %.5f +/- %.5f" % (np.mean(aucs), np.std(aucs)))

for i, h in enumerate(HORIZONS):
    sub["prob_%dh" % h] = pred[:, i]
path = os.path.join(a.out, "submission.csv")
sub.to_csv(path, index=False)
print("wrote", path)
print(sub.head().round(4).to_string(index=False))
