# RetailNova — Predicción de abandono de clientes

Proyecto de la asignatura **Gestión de Proyectos de Datos (SCY1102)**, Evaluación Parcial N°5,
caso *Caso 1. RetailNova*. Equipo: Stephanie Chamorro, María Calfileo, Rocío Cruces.

Implementa el pipeline completo de ciencia de datos (metodología CRISP-DM) para predecir el
abandono de clientes de una cadena retail chilena, desde la generación/comprensión de los datos
hasta un dashboard funcional para el equipo comercial.

## Estructura del repositorio

```
retailnova/
├── data/
│   ├── generate_dataset.py      # Generador del dataset sintético (185k clientes)
│   └── retailnova_clientes.xlsx # Dataset (usado por el notebook), con hoja de diccionario de datos y montos en formato CLP
├── notebooks/
│   ├── build_notebook.py        # Script que construye el notebook (nbformat)
│   └── 01_prediccion_abandono.ipynb  # Notebook ejecutado: EDA, limpieza, modelado, evaluación
├── dashboard/
│   ├── index.html               # Dashboard funcional (HTML + Chart.js, autocontenido)
│   ├── data.json                # Datos agregados que consume el dashboard
│   └── chart.umd.min.js         # Librería de gráficos (local, sin dependencias externas)
└── outputs/
    ├── modelo_abandono.joblib   # Modelo entrenado (regresión logística + scaler + umbral)
    ├── clientes_scored.csv      # Score de riesgo para los 185.000 clientes
    ├── resumen_resultados.csv   # Métricas finales del modelo
    └── fig_*.png                # Gráficos de diagnóstico y evaluación
```

## Cómo reproducir

### 0. Crear y activar un entorno virtual (recomendado)

Evita conflictos con otras instalaciones de Python. Desde la carpeta raíz del repositorio
(`retailnova_repo` o como se llame tu carpeta descargada):

```bash
# Windows (PowerShell / CMD)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Verifica que quedó activado: el inicio de la línea del terminal debe mostrar `(venv)`.
En VS Code, además selecciona ese entorno como kernel del notebook (ícono arriba a la
derecha del notebook → "Select Kernel" → elegir el intérprete dentro de `venv`).

Si al ejecutar una celda sale el error `requiere el paquete ipykernel`, instálalo dentro
del entorno activado:

```bash
pip install ipykernel
python -m ipykernel install --user --name python3
```

### 1. Instalar dependencias y generar el dataset (crea `data/retailnova_clientes.xlsx`)

```bash
pip install pandas numpy openpyxl scikit-learn joblib nbformat nbclient matplotlib
python data/generate_dataset.py
```

### 2. Construir y ejecutar el notebook

```bash
python notebooks/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/01_prediccion_abandono.ipynb --inplace
```

Alternativa si `jupyter nbconvert` no está disponible en tu entorno: abre
`notebooks/01_prediccion_abandono.ipynb` en VS Code (con el kernel de `venv` seleccionado)
y usa el botón **"Ejecutar todo"**.

### 3. Ver el dashboard

```bash
cd dashboard
python -m http.server 8000
# abrir http://localhost:8000 en el navegador (no http://[::]:8000, esa dirección solo
# indica que el servidor escucha en todas las interfaces de red)
```

En Windows, si `python3` no es reconocido, usa `python` (sin el 3). Para detener el
servidor, `Ctrl + C` en la terminal.

## Resultados principales

| Métrica | Valor |
|---|---|
| Accuracy | 0.792 |
| Precisión | 0.489 |
| Recall (ajustado) | 0.750 |
| F1-score | 0.592 |
| ROC-AUC | 0.856 |
| Brecha de recall entre regiones | 0.015 |

## Gestión del proyecto

El seguimiento planificado vs. ejecutado, desviaciones y acciones correctivas se documenta en el
tablero Jira del proyecto: https://retailnova.atlassian.net/jira/software/projects/SCRUM/boards/1
(tareas SCRUM-15 a SCRUM-20).