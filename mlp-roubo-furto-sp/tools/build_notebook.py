"""Gera notebooks/mlp_ocorrencias_sp.ipynb, executa as células e embute as saídas."""
import base64, contextlib, io, json, os, sys, traceback
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

MD, CODE = "markdown", "code"
cells = []
def md(t): cells.append((MD, t.strip("\n")))
def code(t): cells.append((CODE, t.strip("\n")))

md("""
# Segurança pública inteligente em São Paulo — classificação de ocorrências com MLP
**Global Solution · Aprendizado Profundo com Redes Neurais Convolucionais**  
Equipe: Gabriel Lima da Silva (RM 568436) · Nicolas Araujo de Oliveira (RM 566780)

**Objetivo.** Treinar e comparar MLPs que, dadas as características de uma ocorrência **já registrada (ou hipotética)** na capital paulista,
indicam se ela é mais compatível com **roubo** ou **furto**.

> ⚠️ **O modelo não prevê crimes.** A saída é a probabilidade do *tipo* da ocorrência, dado que ela ocorreu —
> nunca a chance de um crime acontecer em determinado local/horário.

**Sumário:** 1 Fonte e recorte · 2 Exploração · 3 Atributos · 4 Divisão e codificação · 5 Implementação da MLP ·
6 Experimentos · 7 Avaliação e seleção · 8 Teste final · 9 Exportação · 10 Conclusões
""")

md("""
## 1. Fonte de dados e recorte

* **Fonte:** SSP-SP, Portal de Estatísticas → *Consultas* → *Bases de Dados*: <https://www.ssp.sp.gov.br/estatistica/consultas>  
  Base **Celulares Subtraídos 2026** (`CelularesSubtraidos_2026.xlsx`, aba `CELULAR_2026`, 256.076 linhas × 55 colunas) — traz roubos e furtos no mesmo arquivo.
  O `.xlsx` (72 MB) é convertido para CSV com `python tools/xlsx_to_csv.py` e salvo em `data/raw/`.
* **Recorte geográfico:** fatos ocorridos no **município de São Paulo**. Pelo *Dicionário de Dados* da própria base, `CIDADE` = "cidade dos fatos", enquanto
  `NOME_MUNICIPIO` = "município **responsável pelo registro**" (delegacia). Usamos **`CIDADE == S.PAULO`**.
* **Rótulo (y):** a partir da **rubrica**: `Roubo (art. 157)` → 1 · `Furto (art. 155)` / `Furto de coisa comum (art. 156)` → 0. Outras naturezas
  (perda/extravio, estelionato, extorsão, localização de veículo…) e boletins com roubo *e* furto são **descartados**.
* **Duplicidades:** um boletim com vários itens (vários celulares) gera várias linhas → mantemos **uma linha por boletim**. A chave é `NUM_BO + ANO_BO + ID_DELEGACIA`
  (o número do BO se repete entre delegacias).
* **Hora ausente:** `HORA_OCORRENCIA` está vazia em ~41% dos BOs, mas nesses casos `DESCR_PERIODO` traz o período. Descartar essas linhas removeria dezenas de milhares de registros e **enviesaria a
  amostra** (a proporção de roubo é muito diferente entre BOs com e sem hora exata). Por isso o período vem da hora **ou** de `DESCR_PERIODO` e criamos o atributo binário `hora_informada`.
* **Amostragem:** até 20 mil registros, sorteio aleatório com semente fixa (preserva a proporção natural das classes).
""")

code("""
import sys, inspect, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
ROOT = Path("..").resolve(); sys.path.insert(0, str(ROOT))
from src.data_prep import build_dataset
from src import mlp_lib as L
from src.synthetic_data import gerar_csv

# >>> Arquivos reais da SSP-SP em data/raw/ (os DEMO_* são sintéticos e ficam de fora).
# Se só houver o .xlsx, converta antes:  python tools/xlsx_to_csv.py ARQ.xlsx data/raw/ARQ.csv
reais = [p for p in sorted((ROOT/"data/raw").glob("*")) if p.suffix.lower() in (".xlsx",".csv") and not p.name.startswith("DEMO")]
if reais:
    DATA_FILES, SINTETICO = reais, False
else:
    DATA_FILES, SINTETICO = [gerar_csv(ROOT/"data/raw/DEMO_sintetico_formato_ssp.csv")], True
    print("⚠️  ATENÇÃO: nenhum arquivo real encontrado em data/raw/. Usando DADOS SINTÉTICOS DE DEMONSTRAÇÃO.")
    print("    Os números abaixo NÃO representam São Paulo. Baixe a base da SSP-SP e rode novamente.\\n")

df, rel = build_dataset(DATA_FILES, n_max=20000, seed=L.SEED)
print(json.dumps(rel, indent=2, ensure_ascii=False))
""")

md("## 2. Exploração dos dados")
code("""
print(df.head(8).to_string()); print()
print("Dimensão:", df.shape, "| valores ausentes:", int(df.isna().sum().sum()))
print("\\nDistribuição das classes (0=furto, 1=roubo):")
print(df["y"].value_counts().rename({0:"furto",1:"roubo"}).to_frame("n").assign(pct=lambda d: (100*d.n/d.n.sum()).round(1)).to_string())
print("\\nCardinalidade:", {c: df[c].nunique() for c in L.CAT})
""")
code("""
fig, ax = plt.subplots(1, 4, figsize=(19, 3.8))
df["y"].map({0:"furto",1:"roubo"}).value_counts().plot.bar(ax=ax[0], rot=0, color=["#c2410c","#0f766e"]); ax[0].set_title("Classes")
(df.groupby("tipo_local")["y"].agg(["mean","size"]).query("size>=30")["mean"].sort_values()).plot.barh(ax=ax[1]); ax[1].set_title("% de roubo por tipo de local"); ax[1].set_xlim(0,1)
(df.groupby("periodo")["y"].mean().reindex(["Madrugada","Manhã","Tarde","Noite"])).plot.bar(ax=ax[2], rot=0); ax[2].set_title("% de roubo por período"); ax[2].set_ylim(0,1)
df.groupby("hora_informada")["y"].mean().rename({0:"só período",1:"hora exata"}).plot.bar(ax=ax[3], rot=0); ax[3].set_title("% de roubo: BO com hora exata?"); ax[3].set_ylim(0,1)
plt.tight_layout()
""")
md("""
**Leitura da exploração.** As taxas de roubo variam entre tipos de local e períodos do dia — há sinal nos atributos escolhidos.
Se as classes estiverem desbalanceadas, a acurácia sozinha engana; por isso reportamos também **precision, recall e F1** da classe *roubo*
(classe positiva) e usamos divisão **estratificada**.
""")

md("""
## 3. Atributos de entrada (e justificativa)

| Atributo | Tipo | Por quê | Tratamento |
|---|---|---|---|
| `bairro` | categórico (alta cardinalidade: ~1.900 grafias) | local do fato é o contexto geográfico mais direto | one-hot; bairros com < 30 ocorrências no **treino** → categoria "infrequente" |
| `tipo_local` | categórico | via pública, terminal, residência, comércio… alteram a natureza do crime | one-hot |
| `dia_semana` | categórico | rotina semanal (dia útil × fim de semana); derivado de `DATA_OCORRENCIA_BO` | one-hot |
| `periodo` | categórico | madrugada/manhã/tarde/noite, da hora exata ou de `DESCR_PERIODO` | one-hot |
| `hora_informada` | binário 0/1 | o BO traz hora exata ou só o período (furtos tendem a ser percebidos depois) | passa direto (já está em 0/1) |

**Escalonamento:** todas as entradas são one-hot ou binárias (0/1), já na mesma escala; não há variável contínua que exija padronização.
Ponto de atenção: `hora_informada` reflete *como o BO foi registrado*; a ablação na seção 7 mede o quanto o modelo depende dela.

**Excluídos de propósito:** a própria rubrica/natureza (revela o rótulo → *data leakage*), campos de conduta/tipo de crime que a definem, número do BO, vítima/dados pessoais,
`DESCR_CONDUTA`/`DESDOBRAMENTO`/`CIRCUNSTANCIA` (qualificam a natureza criminal), marca/tipo do objeto, latitude/longitude e logradouro (quase identificadores) e o município de registro.
""")

md("## 4. Divisão dos dados e codificação\nDivisão **estratificada 70/15/15** (treino/validação/teste). O pré-processador é ajustado (`fit`) **apenas no treino**.")
code("""
Xtr, Xva, Xte, ytr, yva, yte = L.split_data(df)
pre = L.make_preprocessor().fit(Xtr)
Ptr, Pva, Pte = pre.transform(Xtr), pre.transform(Xva), pre.transform(Xte)
print(f"treino={len(ytr)}  validação={len(yva)}  teste={len(yte)}")
print(f"proporção de roubo -> treino {ytr.mean():.3f} | val {yva.mean():.3f} | teste {yte.mean():.3f}")
print("Entradas da rede após codificação:", Ptr.shape[1], "colunas")
print("Valores distintos nas colunas codificadas:", np.unique(Ptr))
base = max(yte.mean(), 1-yte.mean())
print(f"Baseline (sempre a classe majoritária): acurácia = {base:.3f}  -> a MLP precisa superar isso")
""")

md("""
## 5. Implementação da MLP

**Arquitetura geral.** `Entrada (n atributos codificados) → [Dense + ativação] × k → Dense(1) + sigmoide`.

* **Camadas ocultas densas:** cada neurônio calcula `a = f(W·x + b)`. Empilhar camadas permite combinações não lineares dos atributos (ex.: "via pública *e* noite").
* **Ativação oculta – ReLU** (`max(0,z)`): barata, evita saturação do gradiente. Testamos também **tanh** (config. E).
* **Saída – 1 neurônio + sigmoide:** devolve P(roubo) ∈ (0,1) → classificação binária (limiar 0,5).
* **Perda – entropia cruzada binária (log-loss):** é a perda adequada para probabilidades e é convexa na saída.
* **Otimizador – Adam** (`lr=1e-3`): taxa adaptativa por parâmetro, converge bem sem muito ajuste. **Batch** = 128.
* **Regularização:** penalidade **L2** (`alpha`) + **early stopping** (para quando a perda de validação não melhora por 8 épocas e restaura os melhores pesos).

> Implementação com `sklearn.neural_network.MLPClassifier` treinado **uma época por vez** (`partial_fit`) para registrar as curvas de treino/validação.
> Código abaixo (também em `src/mlp_lib.py`):
""")
code("print(inspect.getsource(L.train_mlp))")

md("""
## 6. Experimentos com arquiteturas
Variamos **profundidade, largura, ativação e regularização** (uma mudança principal por comparação):

| Config. | Camadas ocultas | Ativação | L2 (alpha) | O que testa |
|---|---|---|---|---|
| A | 16 | ReLU | 1e-4 | linha de base pequena (subajuste?) |
| B | 64-32 | ReLU | 1e-4 | mais capacidade |
| C | 128-64-32 | ReLU | 1e-5 | capacidade alta, pouca regularização (risco de overfitting) |
| D | 128-64-32 | ReLU | 1e-1 | mesma rede de C com **L2 forte** — efeito da regularização |
| E | 64-32 | tanh | 1e-4 | mesma rede de B com outra **ativação** |
""")
code("""
results = L.run_experiments(Ptr, ytr, Pva, yva)
tab = L.comparison_table(results)
print(tab.round(4).to_string())
""")

md("## 7. Avaliação comparativa e seleção (somente validação)")
code("L.plot_comparison(tab);")
code("""
fig, axs = plt.subplots(len(results), 2, figsize=(11, 2.9*len(results)), sharex=False)
for r, (name, res) in zip(axs, results.items()):
    h = res["hist"]; ep = np.arange(1, len(h["loss"])+1)
    r[0].plot(ep, h["loss"], label="treino"); r[0].plot(ep, h["val_loss"], label="validação"); r[0].axvline(h["best_epoch"], ls="--", c="gray", lw=1)
    r[0].set_title(f"{name} — perda"); r[0].legend(fontsize=8)
    r[1].plot(ep, h["acc"], label="treino"); r[1].plot(ep, h["val_acc"], label="validação"); r[1].set_title(f"{name} — acurácia")
plt.tight_layout()
""")
code("""
best = L.select_best(tab)
print("Critério: maior F1 na validação; empate técnico (<= 0,5 p.p.) -> menos parâmetros.")
print("Arquitetura selecionada:", best)
print("\\nDiagnóstico automático (treino vs. validação):")
for name, res in results.items():
    gap = res["train"]["accuracy"] - res["val"]["accuracy"]
    h = res["hist"]; ruido = h["val_loss"][-1] - min(h["val_loss"])
    estado = ("possível OVERFITTING" if gap > 0.03 or ruido > 0.02 else
              "possível UNDERFITTING" if res["train"]["accuracy"] < 0.65 else "bom ajuste")
    print(f"  {name:28s} gap acc(treino-val)={gap:+.3f} | perda val subiu {ruido:+.3f} após o mínimo (época {h['best_epoch']}/{h['epochs_run']}) -> {estado}")
""")
md("""
**Como ler o diagnóstico.**
*Overfitting*: acurácia/perda de treino continuam melhorando enquanto a de validação piora (a distância entre as curvas aumenta) — o early stopping interrompe esse ponto e restaura os melhores pesos.
*Underfitting*: treino e validação ruins e próximos. Se as arquiteturas maiores (C, D) **não superam** a pequena (A) de forma relevante, a limitação está nos **atributos** (informação disponível), não na capacidade da rede.
""")

md("""
### Ablação: o quanto o modelo depende de `hora_informada`?
Treinamos a arquitetura selecionada **sem** esse atributo e comparamos na validação.
""")
code("""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
pre2 = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), L.CAT)]).fit(Xtr)
m2, h2 = L.train_mlp(results[best]["cfg"], pre2.transform(Xtr), ytr, pre2.transform(Xva), yva)
abl = L.metrics(yva, m2.predict(pre2.transform(Xva)))
com = results[best]["val"]
print(pd.DataFrame({"com hora_informada": com, "sem hora_informada": abl}).T.round(4).to_string())
print(f"Baseline (classe majoritária) na validação: {max(yva.mean(), 1-yva.mean()):.3f}")
""")
md("## 8. Avaliação final no conjunto de teste (usado uma única vez)")
code("""
from sklearn.metrics import classification_report
m = results[best]["model"]; pred_te = m.predict(Pte)
print("Métricas no TESTE:", {k: round(v,4) for k,v in L.metrics(yte, pred_te).items()}, f"| baseline acc = {base:.4f}")
print(); print(classification_report(yte, pred_te, target_names=L.CLASSES, digits=3))
""")
code("L.plot_curves(results[best][\"hist\"], f\"Modelo selecionado: {best}\"); L.plot_confusion(yte, pred_te, \"Matriz de confusão (teste)\");")

md("## 9. Exportação do modelo para a aplicação web")
code("""
options = L.form_options(pre, Xtr)
(ROOT/"models").mkdir(exist_ok=True)
L.save_bundle(ROOT/"models/modelo_mlp.joblib", pre, m, best, results[best]["cfg"], options, dict(synthetic=SINTETICO, n_features_in=int(Ptr.shape[1])))
print("Salvo em models/modelo_mlp.joblib | opções do formulário:", {k: (len(v) if k!='hora' else v) for k,v in options.items()})
""")

md("""
## 10. Conclusões e limitações
* A arquitetura escolhida e as métricas estão acima; **a escolha foi feita na validação** e o teste foi usado só no final.
* **A acurácia deve ser lida contra o baseline** de sempre prever a classe majoritária (impresso nas seções 4 e 8). Como roubo é a classe minoritária, **recall e F1 de roubo** são as métricas mais informativas.
* Os atributos disponíveis (local, bairro, dia e período) carregam informação limitada sobre a natureza do crime; arquiteturas maiores (C, D) não superaram a pequena de forma relevante, indicando que o limite está na **informação dos atributos**, não na capacidade da rede.
* Com limiar 0,5 o modelo é conservador ao apontar roubo (precision > recall); o limiar pode ser ajustado conforme o custo de cada tipo de erro.
* **Limitações:** (i) boletins refletem só o que foi registrado (subnotificação); (ii) rubrica depende da classificação do registro policial; (iii) possíveis vieses geográficos/sociais nos dados — o modelo **não** deve ser usado para policiamento preditivo nem para julgar pessoas ou locais; (iv) bairros raros são agrupados.
* **Trabalhos futuros:** mais anos de dados, validação temporal (treinar em meses anteriores, testar nos posteriores), calibração de probabilidades, ajuste de limiar conforme o custo de cada tipo de erro.
* **Uso de IA generativa:** assistente de IA apoiou a estruturação do código e da documentação; a equipe revisou, executou e é capaz de explicar cada etapa.
""")

# ---------------- execução e gravação ----------------
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"))
ns = {"__name__": "__main__"}
nb_cells, count = [], 0
for kind, src in cells:
    if kind == MD:
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}); continue
    count += 1; outs = []
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"<cell {count}>", "exec"), ns)
    except Exception:
        buf.write(traceback.format_exc()); print(buf.getvalue()); raise
    if buf.getvalue():
        outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
    for n in plt.get_fignums():
        f = io.BytesIO(); plt.figure(n).savefig(f, format="png", dpi=110, bbox_inches="tight")
        outs.append({"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(f.getvalue()).decode(), "text/plain": ["<Figure>"]}})
    plt.close("all")
    nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": count, "source": src.splitlines(True), "outputs": outs})
nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("mlp_ocorrencias_sp.ipynb", "w"), ensure_ascii=False, indent=1)
print("notebook gerado:", len(nb_cells), "células")
