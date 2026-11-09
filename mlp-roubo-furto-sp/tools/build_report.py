"""Gera docs/relatorio_tecnico.pptx (10 slides) a partir de models/metrics.json e reports/figs/.
Uso: python tools/build_report.py   (rode depois de treinar com os dados reais)"""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "models/metrics.json").read_text())
F = ROOT / "reports/figs"
SYN = M["synthetic"]
NAVY, ORG, TEAL, INK, MUT, LIGHT = (RGBColor(0x14, 0x20, 0x3A), RGBColor(0xC2, 0x41, 0x0C), RGBColor(0x0F, 0x76, 0x6E),
                                    RGBColor(0x1D, 0x24, 0x33), RGBColor(0x5F, 0x6B, 0x7F), RGBColor(0xF5, 0xF6, 0xF8))
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
rel, tab = M["relatorio_dados"], M["tabela"]
sel = next(r for r in tab if r["modelo"] == M["selecionado"])
cfg = M["config_selecionada"]; pct = lambda v: f"{100*v:.1f}%"
dot = lambda n: f"{n:,}".replace(",", ".")


def tb(s, x, y, w, h, text, size=16, bold=False, color=INK):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.word_wrap = True
    for i, ln in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; p.space_after = Pt(6)
    return box


def slide(n, title, dark=False):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = NAVY if dark else RGBColor(255, 255, 255)
    tb(s, .6, .35, 12, .9, title, 30, True, RGBColor(255, 255, 255) if dark else NAVY)
    tb(s, 12.4, 7.0, .8, .3, str(n), 11, color=MUT)
    if SYN:
        r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.6), Inches(7.0), Inches(4.6), Inches(.34))
        r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xFD, 0xE6, 0x8A); r.line.fill.background()
        t = r.text_frame; t.text = "DADOS SINTÉTICOS DE DEMONSTRAÇÃO — trocar pela base real"
        ru = t.paragraphs[0].runs[0]; ru.font.size = Pt(10); ru.font.bold = True; ru.font.color.rgb = RGBColor(0x78, 0x35, 0x0F)
    return s


def card(s, x, y, w, h, head, body, color=TEAL):
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = LIGHT; r.line.color.rgb = RGBColor(0xE2, 0xE6, 0xEE); r.adjustments[0] = .06
    tb(s, x + .2, y + .12, w - .4, .5, head, 16, True, color); tb(s, x + .2, y + .62, w - .4, h - .7, body, 16)


def table(s, rows, x, y, w, h, widths=None, size=14, bold_row=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v); ru = c.text_frame.paragraphs[0].runs[0]
            ru.font.size = Pt(size); ru.font.bold = (i == 0) or (bold_row is not None and r[0] == bold_row)
    if widths:
        for j, wd in enumerate(widths): t.columns[j].width = Inches(wd)


te = M["teste"]
# 1
s = slide(1, "Roubo ou furto? Classificação de ocorrências em São Paulo com MLP", dark=True)
tb(s, .6, 2.0, 12, 1, "Global Solution — Aprendizado Profundo com Redes Neurais Convolucionais", 20, color=RGBColor(0xCB, 0xD5, 0xE6))
tb(s, .6, 3.1, 12, 1.6, ["Integrantes: Gabriel Lima da Silva (RM 568436) · Nicolas Araujo de Oliveira (RM 566780)",
   "Objetivo: construir, comparar e implantar uma MLP que indica se uma ocorrência já registrada é mais compatível com ROUBO ou FURTO."], 18, color=RGBColor(255, 255, 255))
tb(s, .6, 5.3, 12, 1, "O modelo classifica o TIPO de uma ocorrência. Ele não prevê onde ou quando um crime vai acontecer.", 16, True, RGBColor(0xFB, 0x92, 0x3C))
# 2
s = slide(2, "Problema, fonte de dados e recorte")
card(s, .6, 1.4, 6, 2.6, "Fonte e recorte", ["SSP-SP · Portal de Estatísticas → Bases de Dados",
    "https://www.ssp.sp.gov.br/estatistica/consultas", f"Arquivo(s): {', '.join(rel['fonte'])}",
    "Recorte: CIDADE (cidade dos fatos) = S.PAULO — não o município do registro"])
card(s, 6.8, 1.4, 5.9, 2.6, "Registros", [f"Brutos: {dot(rel['registros_brutos'])} → capital: {dot(rel['apos_recorte_capital'])}",
    f"Finais (amostra): {dot(rel['registros_finais'])} · roubo = {pct(rel['proporcao_roubo'])}",
    f"Duplicidades por BO removidas: {dot(rel.get('linhas_duplicadas_por_bo_removidas', 0))}",
    f"Outras naturezas descartadas: {dot(rel['descartados_rubrica_ambigua_ou_outra'])} · BOs sem hora exata: {pct(rel.get('pct_bo_sem_hora_exata', 0))}"], ORG)
card(s, .6, 4.25, 12.1, 2.5, "Atributos de entrada e justificativa", [
    "bairro e tipo de local (contexto do fato) · dia da semana e período (rotina) · hora exata informada? (0/1).",
    "Excluídos: rubrica/natureza (data leakage), nº do BO, dados pessoais, lat/long exatas.",
    "Rótulo: rubrica contém ROUBO → 1; FURTO → 0; ambíguos descartados."])
# 3
s = slide(3, "Preparação dos dados")
card(s, .6, 1.4, 4, 2.4, "Tratamento", ["Hora ausente (~41%) → período vem de DESCR_PERIODO + flag hora_informada", "Texto normalizado (sem acentos)", "Demais ausentes → descartados"])
card(s, 4.8, 1.4, 4, 2.4, "Codificação", ["One-hot: bairro, tipo_local, dia_semana, período", "Categorias com <30 casos no treino → 'infrequente'", f"Entradas da rede: {M['n_features_pos_codificacao']} colunas"], ORG)
card(s, 9.0, 1.4, 3.7, 2.4, "Escalonamento", ["Entradas são one-hot/binárias (0/1)", "Sem variável contínua → padronização dispensada"])
card(s, .6, 4.1, 12.1, 2.4, "Separação (estratificada pela classe)", [
    f"Treino {dot(M['n_treino'])} (70%) · Validação {dot(M['n_val'])} (15%) · Teste {dot(M['n_teste'])} (15%)",
    "Pré-processador ajustado (fit) só no treino → sem vazamento de informação do teste.",
    "Teste usado uma única vez, no modelo já escolhido pela validação."])
# 4
s = slide(4, "Arquitetura da MLP selecionada")
arch = " → ".join([f"Entrada ({M['n_features_pos_codificacao']})"] + [f"Dense {n} ({cfg['activation']})" for n in cfg["hidden"]] + ["Dense 1 (sigmoide)"])
tb(s, .6, 1.4, 12.1, .8, arch, 20, True, NAVY)
card(s, .6, 2.5, 6, 3.9, "Como a rede funciona", ["Cada neurônio: a = f(W·x + b).",
    "Camadas ocultas combinam atributos de forma não linear (ex.: 'via pública' E 'noite').",
    "Saída sigmoide = P(roubo) entre 0 e 1; limiar 0,5.", f"Modelo escolhido: {M['selecionado']} — {dot(sel['parametros'])} pesos."])
card(s, 6.8, 2.5, 5.9, 3.9, "Por que essas escolhas", ["ReLU: simples e evita saturação do gradiente.",
    "Sigmoide na saída: classificação binária com probabilidade.",
    "Parcimônia: entre modelos com F1 equivalente (±0,5 p.p.), escolhe-se o menor."], ORG)
# 5
s = slide(5, "Configuração do treinamento")
card(s, .6, 1.4, 6, 4.9, "Hiperparâmetros", ["Perda: entropia cruzada binária (log-loss)", "Otimizador: Adam",
    f"Taxa de aprendizado: {cfg['lr']}", f"Batch size: {M['batch_size']}",
    f"Épocas: até {M['max_epocas']} com early stopping (paciência {M['paciencia']})", f"Regularização L2 (alpha): {cfg['alpha']}"])
card(s, 6.8, 1.4, 5.9, 4.9, "Early stopping", ["Monitora a perda de VALIDAÇÃO a cada época.",
    "Para quando não melhora por 8 épocas e restaura os melhores pesos.",
    f"Selecionado: melhor época {sel['melhor_epoca']} de {sel['epocas']} executadas."], ORG)
# 6
s = slide(6, "Arquiteturas testadas e justificativas")
why = {"A": "Linha de base pequena: há subajuste?", "B": "Mais capacidade (2 camadas)", "C": "Rede profunda, pouca regularização",
       "D": "Mesma rede de C com L2 forte: efeito da regularização", "E": "Mesma rede de B com tanh: efeito da ativação"}
rows = [["Config.", "Camadas ocultas", "Ativação", "L2", "Por que testar"]]
for k, c in M["configs"].items():
    rows.append([k[0], "-".join(map(str, c["hidden"])), c["activation"], f"{c['alpha']:g}", why[k[0]]])
table(s, rows, .6, 1.5, 12.1, 3.6, [1.1, 2.2, 1.4, 1.1, 6.3])
tb(s, .6, 5.5, 12.1, 1, "Uma mudança principal por comparação (profundidade/largura, regularização, ativação), com dados, divisão e semente fixos.", 15, color=MUT)
# 7
s = slide(7, "Comparação dos modelos (validação) e escolha")
rows = [["Modelo", "Acurácia", "Precision", "Recall", "F1", "Pesos"]] + [
    [r["modelo"], pct(r["acc_val"]), pct(r["precision_val"]), pct(r["recall_val"]), pct(r["f1_val"]), dot(r["parametros"])] for r in tab]
table(s, rows, .6, 1.4, 12.1, 3.3, [4.1, 1.6, 1.6, 1.6, 1.6, 1.6], bold_row=M["selecionado"])
tb(s, .6, 5.0, 12.1, 1.8, [f"Escolhido: {M['selecionado']} (maior F1 na validação; empate técnico → menos pesos).",
   f"Teste final: acurácia {pct(te['accuracy'])} · precision {pct(te['precision'])} · recall {pct(te['recall'])} · F1 {pct(te['f1'])}  (classe positiva = roubo).",
   f"Baseline 'sempre a classe majoritária' (furto): acurácia {pct(M['baseline_acc'])} — a MLP supera em {100*(te['accuracy']-M['baseline_acc']):.1f} p.p."], 16, True, NAVY)
# 8
s = slide(8, "Curvas, matriz de confusão e diagnóstico")
s.shapes.add_picture(str(F / "curvas.png"), Inches(.4), Inches(1.3), width=Inches(8.3))
s.shapes.add_picture(str(F / "matriz_confusao.png"), Inches(8.9), Inches(1.3), width=Inches(4.1))
gap = sel["acc_treino"] - sel["acc_val"]
diag = ("Curvas de treino e validação próximas → sem overfitting relevante; sem sinais de underfitting grave (supera o baseline)."
        if abs(gap) < 0.01 else "Treino melhora e validação estagna/piora → overfitting; early stopping restaura os melhores pesos.")
tb(s, .6, 4.9, 12.1, 2, [f"Acurácia treino−validação: {100*gap:+.1f} p.p.; melhor época = {sel['melhor_epoca']} de {sel['epocas']}.", diag,
   "Redes maiores (C, D) não superaram a pequena: o limite está na informação dos atributos, não na capacidade da rede."], 14)
# 9
s = slide(9, "Aplicação web")
if (F / "app_screenshot.png").exists():
    s.shapes.add_picture(str(F / "app_screenshot.png"), Inches(.6), Inches(1.3), height=Inches(5.6))
card(s, 6.6, 1.4, 6.1, 5.3, "Funcionalidades e integração", [
    "Formulário: bairro, tipo de local, dia da semana, período e 'hora exata informada' — exatamente os atributos do treino.",
    "Flask carrega o pipeline salvo (one-hot + MLP) e chama predict_proba.",
    "Resultado: classe estimada + probabilidades de roubo e furto.",
    "Aviso fixo: não é risco de crime. Valores raros → categoria 'infrequente'.",
    "Endpoints: GET / · POST /api/predict · GET /health.",
    "Link: https://claude.ai/artifact/8WTRgG4cbjwM8Awmh2txn4"])
# 10
s = slide(10, "Conclusões, limitações e uso de IA generativa", dark=True)
tb(s, .6, 1.4, 12.1, 5.4, [
    f"• Melhor modelo: {M['selecionado']} — F1 de teste {pct(te['f1'])}, acurácia {pct(te['accuracy'])}.",
    f"• Baseline (sempre furto) = {pct(M['baseline_acc'])}; a MLP supera, mas o recall de roubo é modesto ({pct(te['recall'])}).",
    "• Redes maiores não ajudaram: os atributos disponíveis limitam o desempenho.",
    f"• Ablação: sem 'hora exata informada' o F1 cai de {pct(M['ablacao']['com']['f1'])} para {pct(M['ablacao']['sem_hora_informada']['f1'])} — o atributo reflete como o BO foi registrado.",
    "• Limitações: subnotificação, rubrica depende do registro, vieses geográficos, bairros raros agrupados, só BOs de 2026.",
    "• Uso ético: não serve para policiamento preditivo nem para julgar pessoas ou locais.",
    "• Próximos passos: mais anos, validação temporal, calibração de probabilidades, ajuste do limiar.",
    "• IA generativa (Claude): apoiou estrutura do código, depuração e documentação; a equipe revisou, executou e sabe explicar cada etapa."],
    18, color=RGBColor(255, 255, 255))
out = ROOT / "docs/relatorio_tecnico.pptx"; prs.save(out); print(out)
