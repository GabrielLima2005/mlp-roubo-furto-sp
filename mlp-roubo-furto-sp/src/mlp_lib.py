"""Pré-processamento, treino da MLP (com histórico por época), avaliação e experimentos."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

CAT = ["bairro", "tipo_local", "dia_semana", "periodo"]
BIN = ["hora_informada"]          # 0/1: o BO traz a hora exata (1) ou só o período (0)
FEATURES = CAT + BIN
CLASSES = ["furto", "roubo"]            # y = 0 -> furto ; y = 1 -> roubo
SEED = 42

# ---- Configurações de arquitetura comparadas (≥ 3 exigidas) ---------------------------------
CONFIGS = {
    "A_simples (16)":            dict(hidden=(16,),         activation="relu", lr=1e-3, alpha=1e-4),
    "B_media (64-32)":           dict(hidden=(64, 32),      activation="relu", lr=1e-3, alpha=1e-4),
    "C_profunda (128-64-32)":    dict(hidden=(128, 64, 32), activation="relu", lr=1e-3, alpha=1e-5),
    "D_profunda_L2 (128-64-32)": dict(hidden=(128, 64, 32), activation="relu", lr=1e-3, alpha=1e-1),
    "E_tanh (64-32)":            dict(hidden=(64, 32),      activation="tanh", lr=1e-3, alpha=1e-4),
}
BATCH, MAX_EPOCHS, PATIENCE = 128, 80, 8


def make_preprocessor(min_freq: int = 30) -> ColumnTransformer:
    """One-hot nas categóricas (categorias raras viram 'infrequente'); a binária passa direto (já é 0/1,
    não precisa de escalonamento).
    Ajustado SOMENTE no treino (fit) e aplicado em validação/teste/web (transform)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq,
                              sparse_output=False), CAT),
        ("bin", "passthrough", BIN),
    ])


def split_data(df: pd.DataFrame, seed: int = SEED):
    """70% treino / 15% validação / 15% teste, estratificado pela classe."""
    X, y = df[FEATURES], df["y"].to_numpy()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def metrics(y_true, y_pred) -> dict:
    """Métricas para a classe positiva = ROUBO (y=1)."""
    return dict(accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0))


def train_mlp(cfg: dict, Xtr, ytr, Xva, yva, seed: int = SEED):
    """Treina 1 época por vez (partial_fit) para registrar curvas de perda/acurácia de treino e
    validação. Early stopping pela perda de validação (restaura os melhores pesos)."""
    mlp = MLPClassifier(hidden_layer_sizes=cfg["hidden"], activation=cfg["activation"], solver="adam",
                        alpha=cfg["alpha"], learning_rate_init=cfg["lr"], batch_size=BATCH,
                        shuffle=True, random_state=seed)
    hist = {k: [] for k in ("loss", "val_loss", "acc", "val_acc")}
    best, best_w, wait = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        mlp.partial_fit(Xtr, ytr, classes=[0, 1])
        p_tr, p_va = mlp.predict_proba(Xtr)[:, 1], mlp.predict_proba(Xva)[:, 1]
        hist["loss"].append(log_loss(ytr, p_tr)); hist["val_loss"].append(log_loss(yva, p_va))
        hist["acc"].append(accuracy_score(ytr, p_tr > .5)); hist["val_acc"].append(accuracy_score(yva, p_va > .5))
        if hist["val_loss"][-1] < best - 1e-5:
            best, wait = hist["val_loss"][-1], 0
            best_w = (copy.deepcopy(mlp.coefs_), copy.deepcopy(mlp.intercepts_))
            hist["best_epoch"] = epoch + 1
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    mlp.coefs_, mlp.intercepts_ = best_w          # restaura o melhor ponto
    hist["epochs_run"] = len(hist["loss"])
    return mlp, hist


def run_experiments(Xtr, ytr, Xva, yva, configs: dict = CONFIGS):
    """Treina todas as configurações; devolve {nome: (modelo, histórico, métricas_treino, métricas_val)}."""
    out = {}
    for name, cfg in configs.items():
        mlp, h = train_mlp(cfg, Xtr, ytr, Xva, yva)
        out[name] = dict(model=mlp, hist=h, cfg=cfg,
                         train=metrics(ytr, mlp.predict(Xtr)), val=metrics(yva, mlp.predict(Xva)),
                         val_loss=float(min(h["val_loss"])))
    return out


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({"modelo": name, "parametros": sum(w.size for w in r["model"].coefs_),
                     "epocas": r["hist"]["epochs_run"], "melhor_epoca": r["hist"]["best_epoch"],
                     "acc_treino": r["train"]["accuracy"], "acc_val": r["val"]["accuracy"],
                     "precision_val": r["val"]["precision"], "recall_val": r["val"]["recall"],
                     "f1_val": r["val"]["f1"], "loss_val": r["val_loss"]})
    return pd.DataFrame(rows).set_index("modelo")


def select_best(table: pd.DataFrame, tol: float = 0.005) -> str:
    """Critério: maior F1 na VALIDAÇÃO; entre os que ficam a até `tol` do melhor, o de menos parâmetros
    (parcimônia: diferenças tão pequenas estão dentro do ruído entre execuções)."""
    ok = table[table["f1_val"] >= table["f1_val"].max() - tol]
    return ok["parametros"].idxmin()


# ---- Gráficos --------------------------------------------------------------------------------
def plot_curves(hist: dict, title: str, path: str | Path | None = None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = np.arange(1, len(hist["loss"]) + 1)
    ax[0].plot(ep, hist["loss"], label="treino"); ax[0].plot(ep, hist["val_loss"], label="validação")
    ax[0].axvline(hist["best_epoch"], ls="--", c="gray", lw=1, label=f"melhor época ({hist['best_epoch']})")
    ax[0].set(title="Perda (log-loss)", xlabel="época"); ax[0].legend()
    ax[1].plot(ep, hist["acc"], label="treino"); ax[1].plot(ep, hist["val_acc"], label="validação")
    ax[1].set(title="Acurácia", xlabel="época"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig


def plot_confusion(y_true, y_pred, title: str, path: str | Path | None = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Previsto", ylabel="Real", title=title)
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig


def plot_comparison(table: pd.DataFrame, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    table[["acc_val", "precision_val", "recall_val", "f1_val"]].plot.bar(ax=ax, rot=15)
    lo = float(table[["acc_val", "precision_val", "recall_val", "f1_val"]].min().min())
    ax.set(ylim=(max(0, lo - 0.05), 1), title="Comparação das arquiteturas (validação)", xlabel="")
    ax.legend(loc="lower right", ncol=4, fontsize=8); fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig


# ---- Persistência para a aplicação web --------------------------------------------------------
def form_options(pre: ColumnTransformer, df_train: pd.DataFrame) -> dict:
    """Opções do formulário = categorias FREQUENTES vistas no treino (as raras viram 'OUTRO')."""
    enc = pre.named_transformers_["cat"]
    opts = {}
    for col, cats, infreq in zip(CAT, enc.categories_, enc.infrequent_categories_):
        keep = [c for c in cats if infreq is None or c not in set(infreq)]
        opts[col] = sorted(map(str, keep))
    return opts


def save_bundle(path, pre, mlp, name, cfg, options, extra: dict):
    joblib.dump(dict(preprocessor=pre, model=mlp, name=name, cfg=cfg, options=options,
                     classes=CLASSES, features=FEATURES, **extra), path)
