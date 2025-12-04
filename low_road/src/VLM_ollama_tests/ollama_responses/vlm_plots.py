import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# --- 1. DATI FITTIZI ---
data = {
    'Model_Name': ['CLIP_Base', 'ViT-L/14', 'FLAVA', 'CoCa'],
    'VLM_Accuracy': [10.5, 40.2, 55.0, 78.5],
    'Inference_Time_s': [150, 180, 250, 300],
    'Parameters_B': [1, 3, 8, 15],
    'Architecture_Family': ['Qwen', 'Qwen', 'Qwen', 'Gemma']
}

df = pd.DataFrame(data)

# --- 2. PREPARAZIONE DEL GRAFICO ---
scale_factor = 1000 # Ho ridotto leggermente dato che aggiungiamo un offset

# CORREZIONE QUI: Aggiungiamo un valore base (es. +0.5 o +1) al risultato del log
# In questo modo: log10(1) = 0 -> diventa 0.5 -> dimensione visibile
df['Bubble_Size'] = (np.log10(df['Parameters_B']) + 0.5) * scale_factor

colors = pd.factorize(df['Architecture_Family'])[0]
# Nota: get_cmap è deprecato nelle nuove versioni, meglio usare 'matplotlib.colormaps' o così:
cmap = plt.cm.Spectral 

# --- 3. CREAZIONE DEL PLOT ---
plt.figure(figsize=(14, 8))

scatter = plt.scatter(
    df['Inference_Time_s'],
    df['VLM_Accuracy'],
    s=df['Bubble_Size'],
    c=colors,
    cmap=cmap,
    alpha=0.6,
    edgecolors='w',
    linewidth=0.5
)

# --- 4. ETICHETTE E ANNOTAZIONI ---
for i in range(len(df)):
    plt.annotate(
        df['Model_Name'][i],
        (df['Inference_Time_s'][i] + 5, df['VLM_Accuracy'][i]),
        fontsize=9,
        alpha=0.9
    )

# --- LEGENDA DIMENSIONE CORRETTA PER L'OFFSET ---
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4) # num=4 perché hai pochi dati

# Estrazione numeri
try:
    size_values_raw = [float(re.findall(r"[\d.]+", l)[0]) for l in labels]
except IndexError:
    size_values_raw = [float(l.replace('$', '').replace('\\mathdefault{', '').replace('}', '')) for l in labels]

# Conversione Inversa corretta tenendo conto dell'offset aggiunto prima (+0.5)
# Formula inversa: (Size / scale) - 0.5 = log10(Param)  =>  Param = 10^((Size/scale) - 0.5)
param_values = [f"{10**((s/scale_factor) - 0.5):.1f} B" for s in size_values_raw]

# plt.legend(handles, param_values, title="Parameters (B)", loc="lower right", frameon=True)

plt.title('VLMs performance (Accuracy vs. Inference)', fontsize=16)
plt.xlabel('Inference time (s)', fontsize=12)
plt.ylabel('VLM Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()