"""Converte o .xlsx grande da SSP-SP (aba de dados) em CSV, em streaming (pandas.read_excel é lento/pesado).
Uso: python tools/xlsx_to_csv.py ARQUIVO.xlsx data/raw/ARQUIVO.csv"""
import csv, datetime as dt, sys
import openpyxl

src, dst = sys.argv[1], sys.argv[2]
wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
ws = next(w for w in wb.worksheets if "RUBRICA" in {str(c.value).upper() for c in next(w.iter_rows(max_row=1))})
fmt = lambda v: "" if v is None else v.strftime("%d/%m/%Y") if isinstance(v, dt.datetime) else v.strftime("%H:%M:%S") if isinstance(v, dt.time) else v
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        w.writerow([fmt(v) for v in r])
print("linhas:", i + 1, "->", dst)
