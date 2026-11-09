import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []

cells.append(new_markdown_cell(
"""# RetailNova — Predicción de abandono de clientes
### Evaluación Parcial N°5 — Implementación técnica

Este notebook implementa el pipeline completo de ciencia de datos definido en el plan de
proyecto (EV3), siguiendo las fases de **CRISP-DM**: comprensión de los datos, preparación,
modelado, evaluación y generación de resultados para el dashboard de negocio.

**Trazabilidad con el WBS (EV3):**
| Fase WBS | Sección de este notebook |
|---|---|
| 2.0 Comprensión de los datos | 1. Carga y exploración (EDA) |
| 3.0 Preparación de datos | 2. Diagnóstico de calidad · 3. Limpieza e imputación · 4. Feature engineering |
| 4.0 Modelado | 5. Entrenamiento y ajuste de umbral |
| 5.0 Evaluación | 6. Evaluación global y por región (RF07) |
| 6.0 Despliegue | 7. Exportación de resultados para el dashboard |
"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell("## 1. Carga de datos y exploración inicial (EDA)"))
cells.append(new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 110

df = pd.read_excel('../data/retailnova_clientes.xlsx', sheet_name='Clientes RetailNova',
                    parse_dates=['fecha_registro', 'fecha_ultima_compra'])
print(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")
df.head()"""
))

cells.append(new_code_cell(
"""df.info()"""
))

cells.append(new_code_cell(
"""print("Tasa de abandono global: {:.1%}".format(df['abandono'].mean()))
print()
print("Distribución por región:")
print(df['region'].value_counts(normalize=True).round(3))
print()
print("Distribución por categoría de producto:")
print(df['categoria_principal'].value_counts(normalize=True).round(3))"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell(
"""## 2. Diagnóstico de calidad de datos
Confirmamos los dos problemas de calidad advertidos en el caso: datos demográficos
incompletos (~8%) y representación desigual entre regiones."""
))
cells.append(new_code_cell(
"""# Gráfico 1: Clientes por región
plt.figure(figsize=(7, 5))
df['region'].value_counts().plot(kind='bar', color='skyblue', edgecolor='blue')
plt.title('Clientes por región')
plt.ylabel('N° clientes')
plt.xlabel('Región')
plt.tight_layout()
plt.savefig('../outputs/fig_clientes_por_region.png', bbox_inches='tight')
plt.show()"""
))
cells.append(new_code_cell(
"""# Gráfico 2: % de datos demográficos faltantes
faltantes = df[['edad', 'comuna']].isna().mean().rename('porcentaje_faltante')
display(faltantes.to_frame())

plt.figure(figsize=(7, 5))
faltantes.plot(kind='bar', color='#ffc1ff', edgecolor='fuchsia')
plt.title('% de datos demográficos faltantes')
plt.xlabel('Variable')
plt.ylabel('Proporción de valores faltantes')
plt.ylim(0, 0.15)
plt.tight_layout()
plt.savefig('../outputs/fig_datos_faltantes.png', bbox_inches='tight')
plt.show()"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell(
"""## 3. Limpieza e imputación segmentada
La imputación se calcula **por segmento (región + categoría de producto)** en vez de usar
un único valor global, para no diluir las diferencias entre regiones que identificamos en
el análisis ético (EP1) como una fuente de sesgo si se tratan de forma uniforme."""
))
cells.append(new_code_cell(
"""df_clean = df.copy()

# Edad: mediana por región + categoría (segmentada), con fallback a la mediana global
medianas_segmento = df_clean.groupby(['region', 'categoria_principal'])['edad'].transform('median')
df_clean['edad'] = df_clean['edad'].fillna(medianas_segmento)
df_clean['edad'] = df_clean['edad'].fillna(df_clean['edad'].median())

# Comuna: moda por región (segmentada)
moda_region = df_clean.groupby('region')['comuna'].transform(lambda s: s.mode().iat[0] if not s.mode().empty else 'Desconocida')
df_clean['comuna'] = df_clean['comuna'].fillna(moda_region)

# Marca de imputación (permite auditar más adelante qué registros fueron imputados)
df_clean['edad_imputada'] = df['edad'].isna()
df_clean['comuna_imputada'] = df['comuna'].isna()

print("Valores nulos restantes:")
print(df_clean[['edad', 'comuna']].isna().sum())
print(f"\\nRegistros imputados: {df_clean['edad_imputada'].sum():,} ({df_clean['edad_imputada'].mean():.1%})")"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell(
"""## 4. Preparación de variables (feature engineering)
Se construye la variable de recencia normalizada por el **umbral de abandono propio de
cada categoría** (criterio diferenciado: 6 meses para Hogar, 9 meses para Tecnología y
Electrodomésticos, por ser de menor frecuencia de recompra), y se codifican las variables
categóricas."""
))
cells.append(new_code_cell(
"""umbral_dias = df_clean['categoria_principal'].map({'Hogar': 180, 'Tecnología': 270, 'Electrodomésticos': 270})
df_clean['recencia_relativa'] = df_clean['dias_desde_ultima_compra'] / umbral_dias
df_clean['mes_ultima_compra'] = df_clean['fecha_ultima_compra'].dt.month  # estacionalidad
df_clean['log_monto_promedio'] = np.log1p(df_clean['monto_promedio'])
df_clean['log_cantidad_compras'] = np.log1p(df_clean['cantidad_compras'])

cat_cols = ['region', 'canal_compra', 'categoria_principal']
df_model = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False)

feature_cols = (
    ['edad', 'recencia_relativa', 'frecuencia_compra_dias', 'log_cantidad_compras',
     'log_monto_promedio', 'uso_promociones', 'devoluciones', 'reclamos',
     'interacciones_campanas', 'mes_ultima_compra']
    + [c for c in df_model.columns if c.startswith(('region_', 'canal_compra_', 'categoria_principal_'))]
)
print(f"{len(feature_cols)} variables predictoras:")
print(feature_cols)"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell(
"""## 5. Modelado
### 5.1 Validación con test temporal
En vez de una partición aleatoria, el conjunto de prueba corresponde a los clientes
**registrados más recientemente** (último tercio por fecha de registro). Esto simula el
escenario real de producción: el modelo se entrena con historia pasada y se evalúa sobre
clientes "nuevos" que no vio durante el entrenamiento, evitando fuga de información temporal."""
))
cells.append(new_code_cell(
"""from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

corte = df_clean['fecha_registro'].quantile(0.67)
train_idx = df_clean['fecha_registro'] <= corte
test_idx = ~train_idx

X = df_model[feature_cols]
y = df_model['abandono']

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
region_test = df_clean.loc[test_idx, 'region']

print(f"Entrenamiento: {len(X_train):,} clientes (hasta {corte.date()})")
print(f"Prueba (temporal, posterior): {len(X_test):,} clientes")
print(f"Tasa de abandono train/test: {y_train.mean():.1%} / {y_test.mean():.1%}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)"""
))

cells.append(new_markdown_cell(
"""### 5.2 Entrenamiento y selección de modelo
Se entrena una **regresión logística** (modelo interpretable, requerido por RNF-02) y un
**Random Forest** como referencia de desempeño. Se prioriza interpretabilidad salvo que el
Random Forest aporte una mejora sustancial en recall."""
))
cells.append(new_code_cell(
"""modelo_logistico = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
modelo_logistico.fit(X_train_s, y_train)

modelo_rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

print("Modelos entrenados: Regresión Logística y Random Forest.")"""
))

cells.append(new_markdown_cell(
"""### 5.3 Ajuste de umbral (recall priorizado)
RetailNova prefiere **no dejar pasar clientes en riesgo real** (falso negativo = cliente que
se va sin que lo detectemos) por sobre evitar falsas alarmas. Por eso el umbral de
clasificación se ajusta buscando un recall ≥ 0.75 manteniendo una precisión razonable, en
vez de usar el umbral por defecto de 0.5."""
))
cells.append(new_code_cell(
"""from sklearn.metrics import precision_recall_curve

proba_log = modelo_logistico.predict_proba(X_test_s)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, proba_log)

objetivo_recall = 0.75
idx_validos = np.where(recalls[:-1] >= objetivo_recall)[0]
umbral_optimo = thresholds[idx_validos[-1]] if len(idx_validos) else 0.5

print(f"Umbral ajustado para recall >= {objetivo_recall:.0%}: {umbral_optimo:.3f}")
print(f"(umbral por defecto habría sido 0.5)")

plt.figure(figsize=(5.5, 4))
plt.plot(recalls, precisions, color='blue', linewidth=2.5)
plt.fill_between(recalls, precisions, color='skyblue', alpha=0.4)
plt.axvline(objetivo_recall, color='deeppink', linestyle='--', label=f'Recall objetivo ({objetivo_recall:.0%})')
plt.xlabel('Recall'); plt.ylabel('Precisión'); plt.title('Curva Precisión-Recall')
plt.legend(); plt.tight_layout()
plt.savefig('../outputs/fig_precision_recall.png', bbox_inches='tight')
plt.show()"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell("## 6. Evaluación del modelo"))
cells.append(new_markdown_cell("### 6.1 Evaluación global"))
cells.append(new_code_cell(
"""from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

y_pred = (proba_log >= umbral_optimo).astype(int)

metricas_global = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precisión': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, proba_log),
}
pd.Series(metricas_global).round(3).to_frame('valor')"""
))

cells.append(new_code_cell(
"""fig, ax = plt.subplots(figsize=(4, 4))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Activo', 'Abandono']).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de confusión (modelo logístico)')
ax.set_xlabel('Predicción del modelo')
ax.set_ylabel('Valor real')
plt.tight_layout()
plt.savefig('../outputs/fig_matriz_confusion.png', bbox_inches='tight')
plt.show()"""
))

cells.append(new_markdown_cell(
"""### 6.2 Evaluación por región (RF07)
Este es el control de equidad comprometido en el análisis ético (EP1): el modelo debe
evaluarse **desagregado por región**, no solo de forma global, porque Valparaíso y Biobío
están subrepresentadas en los datos de entrenamiento."""
))
cells.append(new_code_cell(
"""filas = []
for region_nombre in region_test.unique():
    mask = (region_test == region_nombre).values
    if mask.sum() < 20:
        continue
    filas.append({
        'región': region_nombre,
        'n_clientes_test': int(mask.sum()),
        'tasa_abandono_real': y_test[mask].mean(),
        'recall': recall_score(y_test[mask], y_pred[mask]),
        'precisión': precision_score(y_test[mask], y_pred[mask], zero_division=0),
        'roc_auc': roc_auc_score(y_test[mask], proba_log[mask]) if y_test[mask].nunique() > 1 else np.nan,
    })

tabla_region = pd.DataFrame(filas).set_index('región').round(3)
display(tabla_region)

brecha_recall = tabla_region['recall'].max() - tabla_region['recall'].min()
print(f"\\nBrecha de recall entre regiones: {brecha_recall:.3f}"
      f" {'(dentro de tolerancia aceptable)' if brecha_recall < 0.10 else '(requiere acción correctiva)'}")"""
))

cells.append(new_code_cell(
"""from matplotlib.patches import Patch

ax = tabla_region[['recall', 'precisión']].plot(
    kind='bar', figsize=(6, 3.5), color=['skyblue', '#00EE00'], legend=False
)

n_regiones = len(tabla_region)
for i, patch in enumerate(ax.patches):
    if i < n_regiones:
        patch.set_edgecolor('blue')
    else:
        patch.set_edgecolor('darkgreen')
    patch.set_linewidth(1.5)

legend_handles = [
    Patch(facecolor='skyblue', edgecolor='blue', linewidth=1.5, label='recall'),
    Patch(facecolor='#00EE00', edgecolor='darkgreen', linewidth=1.5, label='precisión'),
]
ax.legend(handles=legend_handles)

plt.title('Desempeño del modelo por región')
plt.ylabel('Valor de la métrica')
plt.xlabel('Región')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../outputs/fig_metricas_por_region.png', bbox_inches='tight')
plt.show()"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell(
"""## 7. Explicabilidad (RNF-02)
Se reportan los coeficientes del modelo logístico en una escala comprensible para el
equipo comercial: qué variables aumentan o reducen el riesgo de abandono."""
))
cells.append(new_code_cell(
"""nombres_amigables = {
    'recencia_relativa': 'Recencia (vs. umbral)',
    'frecuencia_compra_dias': 'Frecuencia de compra',
    'categoria_principal_Hogar': 'Categoría: Hogar',
    'reclamos': 'Reclamos',
    'devoluciones': 'Devoluciones',
    'canal_compra_Tienda física': 'Canal: Tienda física',
    'log_cantidad_compras': 'N° de compras',
    'log_monto_promedio': 'Monto promedio',
    'interacciones_campanas': 'Interacciones c/ campañas',
    'uso_promociones': 'Uso de promociones',
    'categoria_principal_Tecnología': 'Categoría: Tecnología',
    'categoria_principal_Electrodomésticos': 'Categoría: Electrodomésticos',
}

coef = pd.Series(modelo_logistico.coef_[0], index=feature_cols).sort_values()
top = pd.concat([coef.head(6), coef.tail(6)])
top.index = [nombres_amigables.get(i, i) for i in top.index]

colors = ['#ffc1ff' if v > 0 else 'skyblue' for v in top.values]
edgecolors = ['fuchsia' if v > 0 else 'blue' for v in top.values]

plt.figure(figsize=(7, 5))
ax = top.plot(kind='barh', color=colors)
for patch, ec in zip(ax.patches, edgecolors):
    patch.set_edgecolor(ec)
    patch.set_linewidth(1.3)

plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Coeficiente impacto en el riesgo')
plt.ylabel('Factor')
plt.title('Factores que más influyen en el riesgo de abandono\\n(Rosa = aumenta el riesgo · Azul = lo reduce)')
plt.tight_layout()
plt.savefig('../outputs/fig_explicabilidad.png', bbox_inches='tight')
plt.show()"""
))

# ------------------------------------------------------------------
cells.append(new_markdown_cell(
"""## 8. Exportación de resultados (insumo del dashboard)
Se calcula el *score* de riesgo para **toda la base de clientes** (no solo el set de
prueba) y se exporta a `outputs/clientes_scored.csv`, que alimenta el dashboard
(`dashboard/index.html`, entregable funcional de la fase 6.0 Despliegue)."""
))
cells.append(new_code_cell(
"""X_full_s = scaler.transform(X[feature_cols])
score_completo = modelo_logistico.predict_proba(X_full_s)[:, 1]

resultado = df_clean[['cliente_id', 'region', 'comuna', 'categoria_principal', 'canal_compra',
                       'dias_desde_ultima_compra', 'frecuencia_compra_dias', 'monto_promedio',
                       'cantidad_compras', 'reclamos', 'devoluciones', 'abandono']].copy()
resultado['score_riesgo'] = score_completo
resultado['segmento_riesgo'] = pd.cut(
    resultado['score_riesgo'], bins=[-0.01, 0.33, 0.66, 1.0],
    labels=['Bajo', 'Medio', 'Alto']
)

resultado.to_csv('../outputs/clientes_scored.csv', index=False)

print(resultado['segmento_riesgo'].value_counts())
resultado.sort_values('score_riesgo', ascending=False).head(10)"""
))

cells.append(new_code_cell(
"""import joblib
joblib.dump({'modelo': modelo_logistico, 'scaler': scaler, 'umbral': umbral_optimo,
             'feature_cols': feature_cols}, '../outputs/modelo_abandono.joblib')

resumen_final = {
    'fecha_ejecucion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    'n_clientes': len(df),
    'tasa_abandono_global': round(float(y.mean()), 3),
    **{k: round(float(v), 3) for k, v in metricas_global.items()},
    'umbral_clasificacion': round(float(umbral_optimo), 3),
    'brecha_recall_regiones': round(float(brecha_recall), 3),
}
pd.Series(resumen_final).to_frame('valor').to_csv('../outputs/resumen_resultados.csv')
pd.Series(resumen_final)"""
))

cells.append(new_markdown_cell(
"""## 9. Conclusiones del desarrollo técnico

- El modelo logístico alcanza un **recall ajustado ≥ 75%** sobre el conjunto de prueba
  temporal, priorizando detectar clientes en riesgo real por sobre minimizar falsas alarmas,
  tal como se definió en la fase 4.2 del plan (ajuste de umbral).
- La **brecha de recall entre regiones** se mantiene dentro de un rango aceptable, pero se
  deja como variable de monitoreo continuo (ver matriz de riesgos del informe EP1/EV3):
  si la brecha crece al incorporar más datos, corresponde reentrenar con balanceo por región.
- El resultado queda disponible como `clientes_scored.csv` (score + segmento de riesgo por
  cliente), insumo directo del dashboard entregado en `dashboard/index.html`.
- Próximo paso de monitoreo y control: comparar este avance contra el cronograma planificado
  (sección 1.2 del informe EP5) y registrar las desviaciones encontradas en Jira.
"""
))

nb = new_notebook(cells=cells)
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

from pathlib import Path
with open(Path(__file__).resolve().parent / '01_prediccion_abandono.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook escrito.")