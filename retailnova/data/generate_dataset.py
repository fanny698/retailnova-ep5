"""
Generador del dataset sintético RetailNova.

Replica fielmente los datos descritos en el enunciado del caso (Caso 1. RetailNova):
- ~185.000 clientes registrados
- 36 meses de historial
- 8% de clientes sin información demográfica completa
- Participación desigual entre regiones (RM sobrerrepresentada)
- Variables: id, edad, comuna, región, fecha de registro, canal de compra,
  cantidad de compras, frecuencia de compra, monto promedio, monto total,
  categoría de producto, uso de promociones, devoluciones, reclamos,
  interacciones con campañas, fecha de última compra.
- Variable objetivo (abandono) con tasa global ~24%, calculada con un criterio
  de abandono diferenciado por categoría (1.3 del WBS: "Criterio de abandono
  por categoría"): Hogar = 6 meses de inactividad, Tecnología/Electrodomésticos
  = 9 meses (compra menos frecuente por ser de mayor durabilidad).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

N = 185_000
HOY = datetime(2026, 10, 1)
INICIO_HISTORIAL = HOY - timedelta(days=36 * 30)

# ---------------------------------------------------------------
# 1. Identificador y región / comuna (participación desigual)
# ---------------------------------------------------------------
cliente_id = [f"CUST-{i:06d}" for i in range(1, N + 1)]

REGIONES = ["Metropolitana", "Valparaíso", "Biobío"]
REGION_P = [0.68, 0.19, 0.13]  # RM sobrerrepresentada, como indica el caso
region = rng.choice(REGIONES, size=N, p=REGION_P)

COMUNAS = {
    "Metropolitana": ["Santiago", "Maipú", "Puente Alto", "La Florida", "Las Condes", "Ñuñoa"],
    "Valparaíso": ["Valparaíso", "Viña del Mar", "Quilpué", "San Antonio"],
    "Biobío": ["Concepción", "Talcahuano", "Los Ángeles", "Chillán"],
}
comuna = np.array([rng.choice(COMUNAS[r]) for r in region])

# ---------------------------------------------------------------
# 2. Edad y canal de compra
# ---------------------------------------------------------------
edad = rng.normal(42, 14, size=N).round().astype(int)
edad = np.clip(edad, 18, 90)

CANALES = ["Tienda física", "E-commerce"]
canal_compra = rng.choice(CANALES, size=N, p=[0.58, 0.42])

# ---------------------------------------------------------------
# 3. Categoría principal de producto
# ---------------------------------------------------------------
CATEGORIAS = ["Hogar", "Tecnología", "Electrodomésticos"]
categoria_principal = rng.choice(CATEGORIAS, size=N, p=[0.46, 0.29, 0.25])

# ---------------------------------------------------------------
# 4. Fecha de registro (dentro de los últimos 36 meses)
# ---------------------------------------------------------------
dias_desde_inicio = rng.integers(0, 36 * 30, size=N)
fecha_registro = np.array([INICIO_HISTORIAL + timedelta(days=int(d)) for d in dias_desde_inicio])

# ---------------------------------------------------------------
# 5. Comportamiento de compra (RFM) -- base de la propensión de abandono
# ---------------------------------------------------------------
# frecuencia_compra_dias: cada cuántos días compra en promedio (menor = más frecuente)
frecuencia_compra_dias = rng.gamma(shape=2.2, scale=35, size=N) + 10
frecuencia_compra_dias = np.clip(frecuencia_compra_dias, 10, 400)

# cantidad de compras en el historial, relacionada con la antigüedad y la frecuencia
antiguedad_dias = np.array([(HOY - fr).days for fr in fecha_registro], dtype=float)
cantidad_compras = np.maximum(1.0, (antiguedad_dias / frecuencia_compra_dias) * rng.uniform(0.6, 1.0, size=N))
cantidad_compras = np.round(cantidad_compras).astype(int)

monto_promedio = rng.lognormal(mean=10.3, sigma=0.45, size=N).round(-2)  # CLP
monto_promedio = np.clip(monto_promedio, 8_000, 450_000)
monto_total_acumulado = (monto_promedio * cantidad_compras * rng.uniform(0.9, 1.1, size=N)).round(-2)

uso_promociones = rng.poisson(lam=1.6, size=N)
devoluciones = rng.poisson(lam=0.35, size=N)
reclamos = rng.poisson(lam=0.22, size=N)
interacciones_campanas = rng.poisson(lam=2.3, size=N)

# días desde la última compra (correlacionado inversamente con frecuencia/uso)
recencia_base = rng.exponential(scale=frecuencia_compra_dias * 1.1, size=N)
dias_desde_ultima_compra = np.clip(recencia_base, 1, 700).round().astype(int)
dias_desde_ultima_compra = np.minimum(dias_desde_ultima_compra, antiguedad_dias.astype(int))
fecha_ultima_compra = np.array([HOY - timedelta(days=int(d)) for d in dias_desde_ultima_compra])

# ---------------------------------------------------------------
# 6. Variable objetivo: abandono (criterio diferenciado por categoría + propensión)
# ---------------------------------------------------------------
umbral_dias = np.where(categoria_principal == "Hogar", 180, 270)  # 6 vs 9 meses

# Propensión latente de abandono (logística), influida por RFM + servicio al cliente
z_sin_intercepto = (
    0.60 * (dias_desde_ultima_compra - umbral_dias) / 30         # recencia vs. umbral de categoría
    + 0.20 * (frecuencia_compra_dias - 60) / 30                  # menor frecuencia -> más riesgo
    - 0.09 * np.log1p(cantidad_compras)                          # más compras históricas -> más fidelidad
    - 0.000006 * (monto_promedio - 60000)                        # ticket más alto -> leve protección
    + 0.22 * reclamos
    + 0.18 * devoluciones
    - 0.12 * uso_promociones
    - 0.10 * interacciones_campanas
    + rng.normal(0, 0.55, size=N)                                # ruido individual
)

# Calibración del intercepto para alcanzar la tasa de abandono observada por
# RetailNova en el caso (~24% de clientes frecuentes sin recompra en 6 meses)
TASA_OBJETIVO = 0.24
lo, hi = -10.0, 10.0
for _ in range(60):
    mid = (lo + hi) / 2
    tasa = (1 / (1 + np.exp(-(z_sin_intercepto + mid)))).mean()
    if tasa > TASA_OBJETIVO:
        hi = mid
    else:
        lo = mid
intercepto = (lo + hi) / 2

z = z_sin_intercepto + intercepto
prob_abandono = 1 / (1 + np.exp(-z))
abandono = (rng.uniform(0, 1, size=N) < prob_abandono).astype(int)

# ---------------------------------------------------------------
# 7. 8% de clientes sin información demográfica completa (edad + comuna)
# ---------------------------------------------------------------
idx_incompletos = rng.choice(N, size=int(N * 0.08), replace=False)
edad = edad.astype(float)
comuna = comuna.astype(object)
edad[idx_incompletos] = np.nan
comuna[idx_incompletos] = None

# ---------------------------------------------------------------
# 8. Ensamblar DataFrame final
# ---------------------------------------------------------------
df = pd.DataFrame({
    "cliente_id": cliente_id,
    "edad": edad,
    "comuna": comuna,
    "region": region,
    "fecha_registro": fecha_registro,
    "canal_compra": canal_compra,
    "cantidad_compras": cantidad_compras,
    "frecuencia_compra_dias": frecuencia_compra_dias.round(1),
    "monto_promedio": monto_promedio,
    "monto_total_acumulado": monto_total_acumulado,
    "categoria_principal": categoria_principal,
    "uso_promociones": uso_promociones,
    "devoluciones": devoluciones,
    "reclamos": reclamos,
    "interacciones_campanas": interacciones_campanas,
    "fecha_ultima_compra": fecha_ultima_compra,
    "dias_desde_ultima_compra": dias_desde_ultima_compra,
    "abandono": abandono,
})

# ---------------------------------------------------------------
# 9. Guardar como Excel (datos + diccionario), con fechas como texto
#    AAAA-MM-DD y montos formateados en pesos chilenos (CLP)
# ---------------------------------------------------------------
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.cell import WriteOnlyCell

df_out = df.copy()
df_out["fecha_registro"] = df_out["fecha_registro"].dt.strftime("%Y-%m-%d")
df_out["fecha_ultima_compra"] = df_out["fecha_ultima_compra"].dt.strftime("%Y-%m-%d")

wb = Workbook(write_only=True)
NAVY = "1E2761"
header_font = Font(name="Arial", bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor=NAVY)
body_font = Font(name="Arial", size=10)

def header_cell(ws, text):
    c = WriteOnlyCell(ws, value=text)
    c.font = header_font
    c.fill = header_fill
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    return c

# Hoja 1: diccionario de datos
ws_dict = wb.create_sheet("Diccionario de datos")
ws_dict.append([header_cell(ws_dict, h) for h in ["Columna", "Descripción", "Unidad / tipo", "Ejemplo"]])
dictionary = [
    ("cliente_id", "Identificador único del cliente.", "Texto (CUST-XXXXXX)", "CUST-000001"),
    ("edad", "Edad del cliente. Vacío cuando el dato demográfico no está disponible (8% de los casos).", "Años", "53"),
    ("comuna", "Comuna de residencia del cliente. Vacío cuando falta el dato demográfico.", "Texto", "San Antonio"),
    ("region", "Región de residencia del cliente.", "Texto (Metropolitana / Valparaíso / Biobío)", "Valparaíso"),
    ("fecha_registro", "Fecha en que el cliente se registró por primera vez.", "Fecha (AAAA-MM-DD)", "2026-06-09"),
    ("canal_compra", "Canal principal de compra del cliente.", "Texto (Tienda física / E-commerce)", "Tienda física"),
    ("cantidad_compras", "Número total de compras históricas registradas.", "Número entero", "2"),
    ("frecuencia_compra_dias", "Cada cuántos días compra el cliente en promedio. NO es un porcentaje: es un intervalo de tiempo. Un valor más bajo significa compras más seguidas (cliente más fiel); un valor más alto significa compras más espaciadas (mayor riesgo de abandono).", "Días", "40.4"),
    ("monto_promedio", "Monto promedio gastado por el cliente en cada compra.", "Pesos chilenos (CLP)", "$54.100"),
    ("monto_total_acumulado", "Monto total acumulado gastado por el cliente en todo su historial.", "Pesos chilenos (CLP)", "$101.300"),
    ("categoria_principal", "Categoría de producto que más compra el cliente.", "Texto (Hogar / Tecnología / Electrodomésticos)", "Hogar"),
    ("uso_promociones", "Número de veces que el cliente ha usado una promoción.", "Número entero", "3"),
    ("devoluciones", "Número de devoluciones realizadas por el cliente.", "Número entero", "0"),
    ("reclamos", "Número de reclamos registrados por el cliente.", "Número entero", "0"),
    ("interacciones_campanas", "Número de interacciones del cliente con campañas de marketing.", "Número entero", "2"),
    ("fecha_ultima_compra", "Fecha de la última compra registrada del cliente.", "Fecha (AAAA-MM-DD)", "2026-09-10"),
    ("dias_desde_ultima_compra", "Días transcurridos desde la última compra hasta la fecha de corte del análisis (01-oct-2026).", "Días", "114"),
    ("abandono", "Variable objetivo del modelo: indica si el cliente se considera en abandono (1) o activo (0), según el umbral diferenciado por categoría (Hogar: 6 meses sin comprar · Tecnología/Electrodomésticos: 9 meses sin comprar).", "Binario (0 = activo, 1 = abandono)", "1"),
]
for row in dictionary:
    cells = []
    for val in row:
        c = WriteOnlyCell(ws_dict, value=val)
        c.font = body_font
        c.alignment = Alignment(wrap_text=True, vertical="top")
        cells.append(c)
    ws_dict.append(cells)
ws_dict.column_dimensions["A"].width = 24
ws_dict.column_dimensions["B"].width = 70
ws_dict.column_dimensions["C"].width = 28
ws_dict.column_dimensions["D"].width = 16

# Hoja 2: datos
ws = wb.create_sheet("Clientes RetailNova")
headers = list(df_out.columns)
ws.append([header_cell(ws, h) for h in headers])
money_idx = {i for i, h in enumerate(headers) if h in {"monto_promedio", "monto_total_acumulado"}}
for row in df_out.itertuples(index=False):
    cells = []
    for i, val in enumerate(row):
        c = WriteOnlyCell(ws, value=val)
        c.font = body_font
        if i in money_idx:
            c.number_format = '"$"#,##0'
        cells.append(c)
    ws.append(cells)
widths = {
    "cliente_id": 13, "edad": 7, "comuna": 14, "region": 15, "fecha_registro": 14,
    "canal_compra": 13, "cantidad_compras": 15, "frecuencia_compra_dias": 18,
    "monto_promedio": 15, "monto_total_acumulado": 18, "categoria_principal": 18,
    "uso_promociones": 14, "devoluciones": 12, "reclamos": 10,
    "interacciones_campanas": 20, "fecha_ultima_compra": 16,
    "dias_desde_ultima_compra": 20, "abandono": 10,
}
for i, h in enumerate(headers, start=1):
    ws.column_dimensions[get_column_letter(i)].width = widths.get(h, 13)
ws.freeze_panes = "A2"

out_path = "/home/claude/retailnova/data/retailnova_clientes.xlsx"
wb.save(out_path)

print("Filas:", len(df))
print("Tasa de abandono global: {:.1%}".format(df["abandono"].mean()))
print("\n% faltante edad:", df["edad"].isna().mean())
print("% faltante comuna:", df["comuna"].isna().mean())
print("\nDistribución por región:")
print(df["region"].value_counts(normalize=True).round(3))
print("\nTasa de abandono por región:")
print(df.groupby("region")["abandono"].mean().round(3))
print("\nTasa de abandono por categoría:")
print(df.groupby("categoria_principal")["abandono"].mean().round(3))
print("\nGuardado en:", out_path)
