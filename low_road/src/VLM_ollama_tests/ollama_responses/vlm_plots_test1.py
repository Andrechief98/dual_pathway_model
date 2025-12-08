import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import json
import os
import matplotlib.patches as mpatches

# --- IL TUO CODICE DI ESTRAZIONE DATI INIZIALE RIMANE INVARIATO ---
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_name = "test1.json"
full_json_path = os.path.join(script_dir, json_file_name)

# Il tuo codice originale per caricare i dati:
with open(full_json_path, "r") as f:
    responses = json.load(f)

models_list = list(responses.keys())
inference_times_list = []
parameters_list = []
for model in models_list:
    sum_inference_time = 0
    for trial in list(responses[model].keys()):
        sum_inference_time += responses[model][trial]["inference_time"]
    inference_times_list.append(round(sum_inference_time/len(list(responses[model].keys())),4))
    params_numb = re.search(r'(\d+\.?\d*)b', model)
    if params_numb:
        parameters_list.append(float(params_numb.group(1)))
    else:
        parameters_list.append(np.nan) # Gestione errore se il nome non combacia col regex


data = {
    'Model_Name': models_list,
    'VLM_Accuracy': [10.5, 40.2, 55.0, 78.5, 60, 80][:len(models_list)], # Tronca se i dati JSON sono meno di 6
    'Inference_Time_s': inference_times_list,
    'Parameters_B': parameters_list,
    'Architecture': ['Qwen', 'Qwen', 'Qwen', 'Gemma', 'Gemma', 'llama'][:len(models_list)]
}

print(models_list)
print(len(models_list))

print(inference_times_list)
print(len(inference_times_list))

print(parameters_list)
print(len(parameters_list))


df = pd.DataFrame(data)

# --- 2. PREPARAZIONE DEL GRAFICO ---
scale_factor = 1000
df['Bubble_Size'] = (np.log10(df['Parameters_B']) + 0.5) * scale_factor

# Creazione della mappatura colori per le famiglie
cmap = plt.cm.Spectral
# Mappa le categorie (stringhe) a valori numerici per il plotter
category_mapping, unique_categories = pd.factorize(df['Architecture'])


# --- 3. CREAZIONE DEL PLOT ---
plt.figure(figsize=(14, 8))

scatter = plt.scatter(
    df['Inference_Time_s'],
    df['VLM_Accuracy'],
    s=df['Bubble_Size'],
    c=category_mapping, # Usa i valori numerici per i colori
    cmap=cmap,
    alpha=0.6,
    edgecolors='w',
    linewidth=0.5
)

# --- 4. ETICHETTE E ANNOTAZIONI (MODIFICATE) ---

# Aggiusta il posizionamento del testo sopra la bolla
y_offset_text = 2 # Offset verticale per posizionare il nome sopra la bolla

for i in range(len(df)):
    plt.annotate(
        df['Model_Name'][i],
        # Posiziona il testo esattamente sopra il centro X della bolla, con un piccolo offset Y
        (df['Inference_Time_s'][i], df['VLM_Accuracy'][i] + y_offset_text),
        fontsize=9,
        alpha=0.9,
        ha='center',  # Allinea orizzontalmente al centro
        va='bottom'   # Allinea verticalmente al fondo (così inizia sopra il punto)
    )

# --- GESTIONE LEGEND DELLA DIMENSIONE ---
handles_size, labels_size = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
try:
    size_values_raw = [float(re.findall(r"[\d.]+", l)[0]) for l in labels_size]
except IndexError:
    size_values_raw = [float(l.replace('$', '').replace('\\mathdefault{', '').replace('}', '')) for l in labels_size]

param_values = [f"{10**((s/scale_factor) - 0.5):.1f} B" for s in size_values_raw]


# --- GESTIONE LEGENDA DEI COLORI (NUOVA SEZIONE) ---
# Creiamo manualmente gli oggetti legenda per i colori basati sulle categorie uniche
# Creiamo un mapper per i colori
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
norm = Normalize(vmin=min(category_mapping), vmax=max(category_mapping))
mapper = ScalarMappable(norm=norm, cmap=cmap)

# Creazione dei patch colorati
legend_patches = []
for i, category in enumerate(unique_categories):
    # Usa il colore mappato per quella categoria specifica
    color = mapper.to_rgba(i) 
    legend_patches.append(mpatches.Patch(color=color, label=category, alpha=0.6))

# Aggiungi la legenda delle famiglie di architetture, posizionata in alto a destra
plt.legend(handles=legend_patches, title="Architecture", loc="lower right", frameon=True)



# --- 5. FINITURE DEL PLOT ---
plt.title('VLMs performance (Accuracy vs. Inference)', fontsize=16)
plt.xlabel('Inference time (s)', fontsize=12)
plt.ylabel('VLM Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
