"""GERADOR DE DADOS SINTÉTICOS — apenas para testar o pipeline sem internet.

Cria um CSV com o *formato* dos arquivos da SSP-SP (inclui ruído: outros municípios, rubricas
ambíguas, boletins duplicados, horas ausentes). Os padrões roubo x furto aqui são INVENTADOS:
nenhuma conclusão sobre São Paulo pode ser tirada destes dados.
Para o trabalho final, baixe a base real (ver README) e rode com ela.
"""
import numpy as np
import pandas as pd

BAIRROS = ["SE", "REPUBLICA", "CONSOLACAO", "BELA VISTA", "LIBERDADE", "PINHEIROS", "VILA MADALENA",
           "ITAIM BIBI", "MOEMA", "VILA MARIANA", "SANTANA", "TUCURUVI", "TATUAPE", "MOOCA", "BRAS",
           "PENHA", "ITAQUERA", "SAO MIGUEL PAULISTA", "GUAIANASES", "CAPAO REDONDO", "JARDIM ANGELA",
           "CAMPO LIMPO", "SANTO AMARO", "INTERLAGOS", "GRAJAU", "LAPA", "PERDIZES", "BARRA FUNDA",
           "BOM RETIRO", "SANTA CECILIA", "CAMBUCI", "IPIRANGA", "SAUDE", "JABAQUARA", "BUTANTA",
           "MORUMBI", "FREGUESIA DO O", "BRASILANDIA", "CIDADE TIRADENTES", "SAPOPEMBA",
           "VILA PRUDENTE", "ARICANDUVA", "CARRAO", "VILA MATILDE", "CIDADE ADEMAR", "PARELHEIROS",
           "RARO BAIRRO A", "RARO BAIRRO B"]
TIPOS = {"VIA PUBLICA": 1.5, "TRANSPORTE COLETIVO": 0.0, "RESIDENCIA": -1.0,
         "COMERCIO E SERVICOS": -0.9, "SHOPPING/ESTACIONAMENTO": -0.6, "OUTROS": -0.2}


def gerar_csv(path, n=24000, seed=7):
    rng = np.random.default_rng(seed)
    peso_b = rng.dirichlet(np.ones(len(BAIRROS)) * 0.7)
    peso_b[-2:] = 0.0004
    peso_b /= peso_b.sum()
    efeito_b = dict(zip(BAIRROS, rng.normal(0, 0.45, len(BAIRROS))))

    bairro = rng.choice(BAIRROS, n, p=peso_b)
    tipo = rng.choice(list(TIPOS), n, p=[.50, .12, .12, .13, .05, .08])
    hora = np.clip(rng.normal(14, 6.5, n), 0, 23.99).astype(int)
    datas = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, n), unit="D")
    dow = np.asarray(datas.dayofweek)

    noite = ((hora >= 18) | (hora < 6)).astype(float)
    logit = (0.1 + np.array([TIPOS[t] for t in tipo]) + 0.9 * noite
             + 0.25 * (dow >= 5) + np.array([efeito_b[b] for b in bairro]) + rng.normal(0, 0.8, n))
    roubo = rng.random(n) < 1 / (1 + np.exp(-logit))

    rub = np.where(roubo, "Roubo (art. 157)", "Furto (art. 155)").astype(object)
    rub[rng.random(n) < 0.03] = "Extorsao (art. 158)"                    # outra natureza
    rub[rng.random(n) < 0.02] = "Roubo (art. 157) / Furto (art. 155)"    # ambíguo
    mun = np.where(rng.random(n) < 0.85, "S.PAULO", rng.choice(["GUARULHOS", "OSASCO", "SANTO ANDRE"], n))
    horas = np.array([f"{h:02d}:{rng.integers(0, 60):02d}:00" for h in hora], dtype=object)
    horas[rng.random(n) < 0.02] = ""                                      # hora ausente

    df = pd.DataFrame({
        "NUM_BO": np.arange(1, n + 1), "ANO_BO": 2024, "CIDADE": mun, "BAIRRO": bairro,
        "DATA_OCORRENCIA_BO": datas.strftime("%d/%m/%Y"), "HORA_OCORRENCIA_BO": horas,
        "DESCR_TIPOLOCAL": tipo, "RUBRICA": rub})
    dup = df.sample(frac=0.06, random_state=seed)                         # BO com 2 itens = 2 linhas
    df = pd.concat([df, dup]).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(path, sep=";", index=False, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(gerar_csv("data/raw/DEMO_sintetico_formato_ssp.csv"))
