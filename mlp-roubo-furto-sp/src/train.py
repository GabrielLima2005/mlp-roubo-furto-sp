"""Executa o pipeline completo e gera modelo + métricas + figuras.
Uso:  python -m src.train --data data/raw/ARQUIVO.xlsx [--n-max 20000]
      python -m src.train --demo         # usa dados SINTÉTICOS (só para testar o pipeline)
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sem janela: só salva as figuras

from src import mlp_lib as L
from src.data_prep import build_dataset
from src.synthetic_data import gerar_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", help="arquivo(s) .xlsx/.csv da SSP-SP")
    ap.add_argument("--demo", action="store_true", help="usa dados sintéticos (NÃO são dados reais)")
    ap.add_argument("--n-max", type=int, default=20000)
    a = ap.parse_args()
    if a.demo:
        a.data = [gerar_csv("data/raw/DEMO_sintetico_formato_ssp.csv")]
    if not a.data:
        ap.error("informe --data ARQUIVO ou --demo")

    df, rel = build_dataset(a.data, n_max=a.n_max)
    print(rel)
    Xtr, Xva, Xte, ytr, yva, yte = L.split_data(df)
    pre = L.make_preprocessor().fit(Xtr)                       # fit só no treino
    Ptr, Pva, Pte = pre.transform(Xtr), pre.transform(Xva), pre.transform(Xte)

    res = L.run_experiments(Ptr, ytr, Pva, yva)
    tab = L.comparison_table(res)
    print(tab.round(4).to_string())
    best = L.select_best(tab)                                  # seleção pela VALIDAÇÃO
    m = res[best]["model"]
    test = L.metrics(yte, m.predict(Pte))                      # teste usado UMA vez
    print("Selecionado:", best, "| teste:", {k: round(v, 4) for k, v in test.items()})

    # ablação: mesma arquitetura SEM o indicador hora_informada
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    pre2 = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                                    sparse_output=False), L.CAT)]).fit(Xtr)
    m2, _ = L.train_mlp(res[best]["cfg"], pre2.transform(Xtr), ytr, pre2.transform(Xva), yva)
    ablacao = dict(com=res[best]["val"], sem_hora_informada=L.metrics(yva, m2.predict(pre2.transform(Xva))))

    figs = Path("reports/figs"); figs.mkdir(parents=True, exist_ok=True)
    L.plot_curves(res[best]["hist"], f"Modelo selecionado: {best}", figs / "curvas.png")
    L.plot_confusion(yte, m.predict(Pte), "Matriz de confusão (teste)", figs / "matriz_confusao.png")
    L.plot_comparison(tab, figs / "comparacao.png")

    options = L.form_options(pre, Xtr)
    Path("models").mkdir(exist_ok=True)
    L.save_bundle("models/modelo_mlp.joblib", pre, m, best, res[best]["cfg"], options,
                  dict(synthetic=bool(a.demo), n_features_in=int(Ptr.shape[1])))
    base = float(max(yte.mean(), 1 - yte.mean()))             # acurácia de "sempre a classe majoritária"
    out = dict(synthetic=bool(a.demo), baseline_acc=base, ablacao=ablacao, relatorio_dados=rel, n_treino=len(ytr), n_val=len(yva), n_teste=len(yte),
               n_features_pos_codificacao=int(Ptr.shape[1]), selecionado=best,
               config_selecionada={k: (list(v) if isinstance(v, tuple) else v) for k, v in res[best]["cfg"].items()},
               batch_size=L.BATCH, max_epocas=L.MAX_EPOCHS, paciencia=L.PATIENCE,
               tabela=tab.reset_index().to_dict(orient="records"),
               teste=test, matriz_confusao=__import__("sklearn.metrics", fromlist=["x"]).confusion_matrix(yte, m.predict(Pte)).tolist(),
               historico={k: res[k]["hist"] for k in res},
               configs={k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v["cfg"].items()} for k, v in res.items()})
    Path("models/metrics.json").write_text(json.dumps(out, indent=2, default=float, ensure_ascii=False))
    print("Artefatos salvos em models/ e reports/figs/")


if __name__ == "__main__":
    main()
