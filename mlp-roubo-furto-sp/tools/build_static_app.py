"""Exporta a MLP treinada para uma página HTML autônoma (inferência em JavaScript, sem servidor).
Uso: python tools/build_static_app.py  ->  app/app_estatica.html"""
import json, sys
from pathlib import Path
import joblib
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.data_prep import DIAS, PERIODOS
B = joblib.load(ROOT / "models/modelo_mlp.joblib")
pre, m = B["preprocessor"], B["model"]
enc = pre.named_transformers_["cat"]
cats = {}
for col, c, inf in zip(["bairro", "tipo_local", "dia_semana", "periodo"], enc.categories_, enc.infrequent_categories_):
    inf = set(map(str, inf)) if inf is not None else set()
    freq = [str(x) for x in c if str(x) not in inf]
    cats[col] = {"freq": freq, "infreq": 1 if inf else 0}      # coluna 'infrequente' vem depois das frequentes
assert len(m.coefs_) == 2
model = {"cats": cats, "W1": m.coefs_[0].tolist(), "b1": m.intercepts_[0].tolist(),
         "W2": [w[0] for w in m.coefs_[1].tolist()], "b2": float(m.intercepts_[1][0]),
         "act": m.activation, "nome": B["name"], "synthetic": bool(B.get("synthetic", False))}
html = (ROOT / "tools/static_app_template.html").read_text(encoding="utf-8") \
    .replace("__MODEL__", json.dumps(model, ensure_ascii=False)) \
    .replace("__DIAS__", json.dumps(DIAS, ensure_ascii=False)).replace("__PERIODOS__", json.dumps(PERIODOS, ensure_ascii=False))
(ROOT / "app/app_estatica.html").write_text(html, encoding="utf-8")
print("ok", len(html) // 1024, "KB; n_entradas =", len(model["W1"]))
