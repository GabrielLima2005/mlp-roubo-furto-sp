"""Leitura, filtragem, rotulagem e engenharia de atributos dos boletins de ocorrência (SSP-SP).

Pipeline (ver notebook para a justificativa de cada passo):
  1. leitura do arquivo bruto (.xlsx ou .csv) baixado do portal da SSP-SP;
  2. recorte geográfico: fatos ocorridos no MUNICÍPIO de São Paulo (campo do local do fato,
     e não o da delegacia que registrou o boletim);
  3. rótulo: ROUBO = 1, FURTO = 0 (a partir da rubrica); ambíguos/outros são descartados;
  4. duplicidades: um boletim com vários itens (ex.: 2 celulares) vira UMA linha;
  5. atributos: bairro, tipo de local, dia da semana, período e "hora exata informada" (0/1).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
        "sexta-feira", "sábado", "domingo"]
PERIODOS = ["Madrugada", "Manhã", "Tarde", "Noite"]

# Nomes possíveis das colunas nos arquivos da SSP-SP (variam entre bases/anos).
# Confirme no dicionário de dados da base que você baixou.
ALIASES = {
    "num_bo":     ["NUM_BO", "NUMERO_BO", "N_BO"],
    "ano_bo":     ["ANO_BO"],
    "id_delegacia": ["ID_DELEGACIA"],
    # CIDADE = "Cidade dos fatos" (dicionário SSP). NOME_MUNICIPIO é o município RESPONSÁVEL PELO REGISTRO
    # (delegacia) e NÃO deve ser usado para o recorte geográfico.
    "municipio":  ["CIDADE", "MUNICIPIO"],
    "bairro":     ["BAIRRO"],
    "data":       ["DATA_OCORRENCIA_BO", "DATA_OCORRENCIA"],
    "hora":       ["HORA_OCORRENCIA_BO", "HORA_OCORRENCIA"],
    "periodo_desc": ["DESCR_PERIODO"],      # usado quando a hora exata está ausente
    "tipo_local": ["DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL", "TIPO_LOCAL"],
    "rubrica":    ["RUBRICA", "NATUREZA_APURADA"],
}
REQUIRED = ["municipio", "bairro", "data", "hora", "tipo_local", "rubrica"]


def _norm(s) -> str:
    """Maiúsculas, sem acentos e sem espaços nas pontas."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().upper()


def _read_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        # a planilha de dados é a que contém a rubrica; as demais costumam ser dicionário
        for df in sheets.values():
            cols = {_norm(c) for c in df.columns}
            if cols & set(ALIASES["rubrica"]):
                return df
        return max(sheets.values(), key=len)
    wanted = {_norm(o) for opts in ALIASES.values() for o in opts}
    for enc in ("utf-8", "latin-1"):
        for sep in (";", ",", "\t"):
            try:  # lê só as colunas necessárias (bases da SSP têm ~55 colunas)
                df = pd.read_csv(path, sep=sep, encoding=enc, dtype=object, low_memory=False,
                                 usecols=lambda c: _norm(c) in wanted)
                if df.shape[1] > 3:
                    return df
            except Exception:
                continue
    raise ValueError(f"Não consegui ler {path}")


def _pick(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia as colunas reais para nomes padronizados."""
    norm_cols = {_norm(c): c for c in df.columns}
    out, missing = {}, []
    for key, options in ALIASES.items():
        hit = next((norm_cols[o] for o in options if o in norm_cols), None)
        if hit is None:
            if key in REQUIRED:
                missing.append(key)
        else:
            out[key] = df[hit]
    if missing:
        raise KeyError(f"Colunas não encontradas para {missing}. Colunas do arquivo: {list(df.columns)}. "
                       "Ajuste o dicionário ALIASES em src/data_prep.py.")
    return pd.DataFrame(out)


def _parse_hour(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if hasattr(v, "hour"):
        return float(v.hour)
    m = re.match(r"^\s*(\d{1,2})\s*[:hH]", str(v))
    if not m:
        return np.nan
    h = int(m.group(1))
    return float(h) if 0 <= h <= 23 else np.nan


_PERIODO_DESC = {"DE MADRUGADA": "Madrugada", "PELA MANHA": "Manhã", "A TARDE": "Tarde", "A NOITE": "Noite"}


def hora_para_periodo(h: float) -> str:
    h = int(h)
    return "Madrugada" if h < 6 else "Manhã" if h < 12 else "Tarde" if h < 18 else "Noite"


def _label(rubrica: str):
    r = _norm(rubrica)
    roubo, furto = "ROUBO" in r, "FURTO" in r
    if roubo and not furto:
        return 1
    if furto and not roubo:
        return 0
    return np.nan  # ambíguo ou outra natureza (ex.: extorsão, receptação)


def build_dataset(paths, n_max: int | None = 20000, seed: int = 42):
    """Retorna (df_modelo, relatorio). df_modelo tem: bairro, tipo_local, dia_semana, periodo, hora, y."""
    paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    raw = pd.concat([_pick(_read_one(p)) for p in paths], ignore_index=True)
    rel = {"fonte": [p.name for p in paths], "registros_brutos": len(raw)}

    # 2) recorte geográfico (município do fato)
    mun = raw["municipio"].map(_norm)
    raw = raw[mun.isin({"SAO PAULO", "S.PAULO", "S PAULO", "S. PAULO", "SAO PAULO (CAPITAL)"})].copy()
    rel["apos_recorte_capital"] = len(raw)

    # 3) rótulo
    raw["y"] = raw["rubrica"].map(_label)
    rel["descartados_rubrica_ambigua_ou_outra"] = int(raw["y"].isna().sum())
    raw = raw.dropna(subset=["y"])

    # 4) duplicidades por boletim
    if "num_bo" in raw:
        key = raw["num_bo"].astype(str) + "/" + (raw["ano_bo"].astype(str) if "ano_bo" in raw else "") \
              + "/" + (raw["id_delegacia"].astype(str) if "id_delegacia" in raw else "")
        conflito = raw.groupby(key)["y"].transform("nunique") > 1       # BO com roubo E furto
        rel["linhas_de_bo_com_roubo_e_furto_removidas"] = int(conflito.sum())
        raw = raw[~conflito]
        key = key[raw.index]
        antes = len(raw)
        raw = raw.loc[~key.duplicated()]
        rel["linhas_duplicadas_por_bo_removidas"] = antes - len(raw)

    # 5) atributos
    out = pd.DataFrame({
        "bairro": raw["bairro"].map(lambda v: _norm(v) if pd.notna(v) else np.nan),
        "tipo_local": raw["tipo_local"].map(lambda v: _norm(v) if pd.notna(v) else np.nan),
        "y": raw["y"].astype(int),
    })
    hora = raw["hora"].map(_parse_hour)
    out["hora_informada"] = hora.notna().astype(int)          # 1 = BO traz hora exata; 0 = só o período
    # período: da hora exata; se ausente, do campo DESCR_PERIODO ("Em hora incerta" -> descartado)
    per_desc = (raw["periodo_desc"].map(lambda v: _PERIODO_DESC.get(_norm(v)) if pd.notna(v) else None)
                if "periodo_desc" in raw else pd.Series(None, index=raw.index))
    out["periodo"] = [hora_para_periodo(h) if pd.notna(h) else d for h, d in zip(hora, per_desc)]
    dt = pd.to_datetime(raw["data"], errors="coerce", dayfirst=True)
    out["dia_semana"] = dt.dt.dayofweek.map(lambda i: DIAS[int(i)] if pd.notna(i) else np.nan)
    out = out.replace({"": np.nan, "NAO INFORMADO": np.nan, "NULL": np.nan})
    antes = len(out)
    out = out.dropna()
    rel["descartados_por_valor_ausente"] = antes - len(out)
    rel["pct_bo_sem_hora_exata"] = round(float(1 - out["hora_informada"].mean()), 4)

    if n_max and len(out) > n_max:
        out = out.sample(n=n_max, random_state=seed)
    out = out.reset_index(drop=True)[["bairro", "tipo_local", "dia_semana", "periodo", "hora_informada", "y"]]
    rel["registros_finais"] = len(out)
    rel["proporcao_roubo"] = round(float(out["y"].mean()), 4)
    return out, rel
