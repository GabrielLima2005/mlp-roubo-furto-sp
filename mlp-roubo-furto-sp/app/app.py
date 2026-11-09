"""Aplicação web (Flask) que usa a MLP treinada para classificar uma ocorrência como ROUBO ou FURTO.
Execução local:  python app/app.py     (abre em http://127.0.0.1:5000)
Produção:        gunicorn app.app:app
"""
from pathlib import Path
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_prep import DIAS, PERIODOS  # noqa: E402

app = Flask(__name__)
BUNDLE = joblib.load(ROOT / "models" / "modelo_mlp.joblib")
PRE, MLP, OPTS = BUNDLE["preprocessor"], BUNDLE["model"], BUNDLE["options"]
OUTRO = "OUTRO / NÃO LISTADO"


def _prever(bairro: str, tipo_local: str, dia_semana: str, periodo: str, hora_informada) -> dict:
    # valores fora da lista (ou "outro") vão para a categoria 'infrequente' aprendida no treino
    bairro = bairro if bairro in OPTS["bairro"] else "__RARO__"
    tipo_local = tipo_local if tipo_local in OPTS["tipo_local"] else "__RARO__"
    if dia_semana not in DIAS:
        raise ValueError("dia_semana inválido")
    if periodo not in PERIODOS:
        raise ValueError("periodo inválido")
    flag = 1 if str(hora_informada).lower() in ("1", "true", "on", "sim") else 0
    row = pd.DataFrame([{"bairro": bairro, "tipo_local": tipo_local, "dia_semana": dia_semana,
                         "periodo": periodo, "hora_informada": flag}])
    p_roubo = float(MLP.predict_proba(PRE.transform(row))[0, 1])
    return {"classe": "roubo" if p_roubo >= 0.5 else "furto", "prob_roubo": p_roubo, "prob_furto": 1 - p_roubo}


@app.get("/")
def index():
    return render_template("index.html", opts=OPTS, dias=DIAS, periodos=PERIODOS, outro=OUTRO, modelo=BUNDLE["name"],
                           synthetic=BUNDLE.get("synthetic", False))


@app.post("/api/predict")
def api_predict():
    d = request.get_json(silent=True) or request.form
    try:
        return jsonify(_prever(d.get("bairro", ""), d.get("tipo_local", ""), d.get("dia_semana", ""),
                               d.get("periodo", ""), d.get("hora_informada", 0)))
    except (ValueError, TypeError) as e:
        return jsonify({"erro": str(e)}), 400


@app.get("/health")
def health():
    return {"status": "ok", "modelo": BUNDLE["name"]}


if __name__ == "__main__":
    app.run(debug=True)
