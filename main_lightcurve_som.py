import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path #per creare cartella
import csv
import json
import random
from pdb import set_trace as stop
import math
import pandas as pd
import torch
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from kohonen_som_gpu import Som



df = pd.read_csv('lightcurves/dataset_lc_test_2.csv')
decam_df = pd.read_csv('dataset_decam_lc_30points.csv')
train_df, early_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(early_df, test_size=0.5, random_state=42, shuffle=True)

#produce array di train
Ncolstrain = len(train_df.columns) - 1
feature_cols = train_df.columns[:Ncolstrain]
data_train = np.zeros((len(train_df), Ncolstrain))
target_train = train_df['63']
for i, col in enumerate(feature_cols):
    data_train[:, i] = train_df[col].values

#produce array di test
Ncolstest = len(test_df.columns) - 1
feature_cols = test_df.columns[:Ncolstest]
data_test = np.zeros((len(test_df), Ncolstest))
target_test = test_df['63']
for i, col in enumerate(feature_cols):
    data_test[:, i] = test_df[col].values


#produce array di val
Ncolsval = len(val_df.columns) - 1
feature_cols = val_df.columns[:Ncolsval]
data_val= np.zeros((len(val_df), Ncolsval))
target_val = val_df['63']
for i, col in enumerate(feature_cols):
    data_val[:, i] = val_df[col].values


#produce array di early
Ncolsearly = len(early_df.columns) - 1
feature_cols = early_df.columns[:Ncolsearly]
data_early= np.zeros((len(early_df), Ncolsearly))
target_early = early_df['63']
for i, col in enumerate(feature_cols):
    data_early[:, i] = early_df[col].values


#produce array di decam
Ncolsdecam = len(decam_df.columns) - 1
feature_cols = decam_df.columns[:Ncolsdecam]
data_decam= np.zeros((len(decam_df), Ncolsdecam))
for i, col in enumerate(feature_cols):
    data_decam[:, i] = decam_df[col].values

# Parametri di training
learning_rate = 0.5
sigma = 4
nearby_neurons_spread_function = "gaussian"
train_len = len(train_df)
neurons_train = int(5*np.sqrt(train_len))
n_neurons = 25  #int(np.sqrt(neurons_train)),
num_epochs = 300
early_stopping = True   # oppure False


som = Som(
    x=n_neurons,
    y=n_neurons,
    input_len=Ncolstrain,
    learning_rate=learning_rate,
    sigma=sigma,
    decay_function='exponential_decay', #'exponential_decay'
    nearby_neurons_spread_function=nearby_neurons_spread_function,
)

trained = True #False se devi allenare di nuovo
if trained == True:

    som.load_weights(
        filename='neuronmap_test_2.txt',
        statusfile='status_test_2.txt'
    )
else:

    som.random_weights_from_data_init(data_train)

    start = timer()

    if early_stopping:
        epochs, errors, movements, error_test = som.train_loop(
            data_train,
            num_iteration=num_epochs,
            use_early_stopping=True,
            test_data= data_test,
            Nsubdata= 10,
        )

    if not early_stopping:
        epochs, errors, movements, error_test= som.train_loop(
            data_train,
            num_iteration=num_epochs,
            use_early_stopping=False,
            Nsubdata= 10
        )

    end = timer()
    training_time = end - start

    som.write_weights(
        som.get_weights(),
        filename='neuronmap_test_2.txt',
        statusfile='status_test_2.txt'
    )
    print(f"Training Time : {training_time:.2f} s")
    print(f"Number of epochs: {len(epochs)}")
    print(f"Final error: {errors[-1]}")
    print(f"Final movements: {movements[-1]}")
    
    plt.figure()
    plt.plot(epochs, errors)
    plt.plot(epochs, error_test)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error during training")
    plt.grid(True)
    plt.savefig("training_error.png")
    #plt.show()
    plt.close()


    plt.figure()
    plt.plot(epochs, movements)
    plt.xlabel("Epochs")
    plt.ylabel("Number of movements")
    plt.title("Movements during training")
    plt.grid(True)
    plt.savefig("training_movements.png")
    plt.close()

#sull'altra definizione era una lista labels, ma dovrebbe essere un dizionario
def plot_umat_map_target(self,
                         n_neurons: int,
                         m_neurons: int,
                         data: torch.tensor,
                         target,
                         labels: dict,
                         markers: list,
                         colors,
                         output: str,
                         mask_flag: bool = False,
                         mm: list = [],
                         mode: str = 'sum') -> None:

    class_ids = list(labels.keys())
    class_names = list(labels.values())
    target_values = target.to_numpy() if hasattr(target, 'to_numpy') else np.asarray(target)

    plt.figure(figsize=(n_neurons,m_neurons))

    dataset_copy = data.copy()
    mask_copy = mm.copy()

    u_matrix = self.umatrix_map(scale=mode)

    plt.pcolor(u_matrix.T, cmap='bone_r')

    colorbar = plt.colorbar()
    colorbar.ax.tick_params(labelsize=30)

    # Disegno dei marker sulle BMU
    if mask_flag:

        for sample_idx, sample in enumerate(dataset_copy):

            sample_mask = mask_copy[sample_idx]

            bmu = self.winning_neuron(
                sample,
                mask_flag=True,
                mm=sample_mask
            )

            plt.plot(
                bmu[0] + 0.5,
                bmu[1] + 0.5,
                markers[target_values[sample_idx] - 1],
                markerfacecolor='None',
                markeredgecolor=colors[target_values[sample_idx] - 1],
                markersize=8,
                markeredgewidth=1
            )

        bmu_x = []
        bmu_y = []

        for sample_idx, sample in enumerate(dataset_copy):

            sample_mask = mask_copy[sample_idx]

            bmu = self.winning_neuron(
                sample,
                mask_flag=True,
                mm=sample_mask
            )

            bmu_x.append(bmu[0])
            bmu_y.append(bmu[1])

        bmu_x = np.array(bmu_x)
        bmu_y = np.array(bmu_y)

        for class_id in np.unique(target_values):

            class_mask = target_values == class_id

            plt.scatter(
                bmu_x[class_mask] + 0.5 +
                (np.random.rand(np.sum(class_mask)) - 0.5) * 0.8,

                bmu_y[class_mask] + 0.5 +
                (np.random.rand(np.sum(class_mask)) - 0.5) * 0.8,

                s=50,
                c=colors[class_id - 1],
                label=labels[class_id]
            )

    else:

        for sample_idx, sample in enumerate(dataset_copy):

            try:
                bmu = self.winning_neuron(
                    torch.tensor(sample, device='cuda')
                )
            except:
                bmu = self.winning_neuron(
                    torch.tensor(sample, device='cpu')
                )

            label_index = class_names.index(target_values[sample_idx])

            plt.plot(
                bmu[0] + 0.5,
                bmu[1] + 0.5,
                markers[label_index],
                markerfacecolor='None',
                markeredgecolor=colors[label_index],
                markersize=8,
                markeredgewidth=1
            )

        bmu_x = []
        bmu_y = []

        for sample in dataset_copy:

            try:
                bmu = self.winning_neuron(
                    torch.tensor(sample, device='cuda')
                )
            except:
                bmu = self.winning_neuron(
                    torch.tensor(sample, device='cpu')
                )

            bmu_x.append(bmu[0])
            bmu_y.append(bmu[1])

        bmu_x = np.array(bmu_x)
        bmu_y = np.array(bmu_y)

        for class_id in np.unique(target_values):

            class_mask = target_values == class_id

            label_index = class_names.index(class_id)

            plt.scatter(
                bmu_x[class_mask] + 0.5 +
                (np.random.rand(np.sum(class_mask)) - 0.5) * 0.8,

                bmu_y[class_mask] + 0.5 +
                (np.random.rand(np.sum(class_mask)) - 0.5) * 0.8,

                s=50,
                c=colors[label_index],
                label=class_id
            )

    plt.legend(
        loc='upper right',
        prop={'size': 30}
    )

    x_tick_positions = (
        np.arange(0, n_neurons, n_neurons // 5) + 0.5
    )

    x_tick_labels = (
        np.arange(0, n_neurons, n_neurons // 5) + 1
    )

    y_tick_positions = (
        np.arange(0, m_neurons, m_neurons // 5) + 0.5
    )

    y_tick_labels = (
        np.arange(0, m_neurons, m_neurons // 5) + 1
    )

    plt.xticks(
        x_tick_positions,
        x_tick_labels,
        fontsize=30
    )

    plt.yticks(
        y_tick_positions,
        y_tick_labels,
        fontsize=30
    )

    if not mask_flag:

        for row in range(1, n_neurons):
            plt.axhline(
                row,
                color='white',
                linewidth=0.5
            )

        for col in range(1, m_neurons):
            plt.axvline(
                col,
                color='white',
                linewidth=0.5
            )

    plt.savefig(output)
    plt.close()

label_names = {'RRab' :'RRab' , 'RRc': 'RRc' , 'Cep': 'Cep', 'T2Cep': 'T2Cep', 'EA' : 'EA', 'EB': 'EB', 'EW' : 'EW', 'LENS': 'LENS'}
markers= ['o', 's', 'd', 'v', 'p', 'h', '+', 'x']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

plot_umat_map_target(som, n_neurons = n_neurons, m_neurons = n_neurons,data = data_train, target = target_train, labels = label_names, markers= markers, colors = colors, output = "umat_train.png")

plot_umat_map_target(som, n_neurons = n_neurons, m_neurons = n_neurons,data = data_early, target = target_early, labels = label_names, markers= markers, colors = colors, output = "umat_early.png")


def make_probability_map(self, n_neurons, m_neurons, data, target, labels):
        
        
        # crea un dizionario che associ ad ogni coordinata i,j quanti e quali oggetti ci cadono
        data_exploded = self.explode_data(data, target, labels)

        label_list = [label for label in labels.values()]
        # Si crea il dizionario che sfrutta i label come keys, ogni label e' associato ad una mappa
        ProbabilityMapDict = dict.fromkeys(label_list)

        # Si crea una mappa di prob per ogni label
        for label in label_list:
            #Si crea una mappa di prob inizializzata a zero con la dimensione dei neuroni
            ProbabilityMap_label = np.zeros((n_neurons, m_neurons))
            # per ogni neurone (se sono presenti oggetti) si prende il numero e tipo di oggetti
            for i in range(0, n_neurons):
                for j in range(0, m_neurons):
                    coord = tuple([i, j])
                    if coord in data_exploded.keys():
                        # la funzione misura quanti oggetti del tipo label sono presenti
                        # nel neurone di coordinate coord
                        num_elements = self.get_data_exploded(data_exploded, coord, label)
                        # con l'argomento total si ottengono il numero total di oggetti
                        # nel neurone di coordinate coord 
                        tot_elements = self.get_data_exploded(data_exploded, coord, 'Total')
                        
                        ProbabilityMap_label[i, j] = num_elements / float(tot_elements) * 100
                        max_val = np.max(ProbabilityMap_label)

                        if max_val > 0:
                            ProbabilityMap_label = ProbabilityMap_label / max_val * 100
            ProbabilityMapDict[label] = ProbabilityMap_label
        return ProbabilityMapDict

probability_maps_train = make_probability_map(som, n_neurons, n_neurons, data_train, target_train, label_names)
print(probability_maps_train.keys())

for label, prob_map in probability_maps_train.items():
    plt.figure(figsize=(8,6))
    plt.imshow(prob_map.T, origin="lower", cmap="viridis")
    plt.colorbar(label="Probability (%)")
    plt.title(f"Probability map: {label}")
    plt.savefig(f"probability_map_train_{label}_test_2.png")
    plt.close()


def classify_with_probability_maps(self, data, probability_maps, label_names, apply_pth=False, pth=70):
    """
    Per ogni campione in `data` trova il winning neuron e restituisce
    la classe con probabilità massima tra tutte le mappe.
 
    Parametri
    ----------
    som              : istanza di Som già allenata
    data             : array numpy (N, n_features)
    probability_maps : dict  { label_name: np.ndarray (n_neurons, m_neurons) }
 
    Ritorna
    -------
    predictions : list di label (stessa lunghezza di data)
    """
    label_list = list(label_names.keys())
    predictions = []
    probs_list = []

        
    for isample, sample in enumerate(data):
        try:
            bmu = self.winning_neuron(torch.tensor(sample, device='cuda'))
        except Exception:
            bmu = self.winning_neuron(torch.tensor(sample, device='cpu'))
 
        i, j = int(bmu[0]), int(bmu[1])
 
        # probabilita' di quel neurone per ogni classe
        probs = {label: probability_maps[label][i, j] for label in label_list}
        # classe con prob massima
        if not apply_pth:
            predicted_label = max(probs, key=probs.get)
            predictions.append(predicted_label)
            prob_max = np.max(list(probs.values()))
            probs_list.append(prob_max)
        else:
            predicted_label = max(probs, key=probs.get)
            prob_max = np.max(list(probs.values()))
            if prob_max > pth:
                predictions.append(predicted_label)
                probs_list.append(prob_max)
 
    return predictions, probs_list



early_predictions, early_probs = classify_with_probability_maps(som, data_early, probability_maps_train, label_names)
#passiamo alla rete i dati 20% di early stopping e classifichiamo in base a winning neuron
#di questi dati teniamo una classificazione legata alla mappa di prob che e' stata ottenuta dall'allenamento
#da winning neuron prendo le coordinate, vedo per ogni mappa in quelle coordinate qual'e' la prob e prendo
#la piu' alta tra le mappe, questo mi da la classificazione 
#allo stesso tempo pero' per questi oggetti conosciamo la classe perche' teniamo il target
#allora confrontiamo la classificazione della rete con quella del target
#facendo cio' possiamo calcolare la completezza e la purezza
# Dopo cio' fare anche per le singole classi
decam_predictions, decam_probs = classify_with_probability_maps(som,data_decam, probability_maps_train, label_names, apply_pth = False ,pth=50)

#ups e' folding prodotto dalla rete neurale upsilon
#m2 e m3 sono i due metodi di folding,
#ls e' lombe scargle
"""
def compute_completeness_purity(y_true, y_pred, labels, probs, prb_thr):
    
    Calcola completezza e purezza per ogni classe.
 
    Definizioni
    -----------
    Completezza (recall)  per classe C:
        = TP_C / (TP_C + FN_C)
        = (oggetti di classe C classificati correttamente) /
          (tutti gli oggetti veri di classe C)
 
    Purezza (precision)  per classe C:
        = TP_C / (TP_C + FP_C)
        = (oggetti di classe C classificati correttamente) /
          (tutti gli oggetti classificati come classe C)
 
    Parametri
    ----------
    y_true : array-like di label veri
    y_pred : array-like di label predetti
    labels : lista ordinata di tutti i label
 
    Ritorna
    -------
    results : dict  { label: {'completezza': float, 'purezza': float,
                               'TP': int, 'FP': int, 'FN': int} }
    

    y_true = np.asarray(y_true)[np.array(probs) > prb_thr]
    y_pred = np.asarray(y_pred)[np.array(probs) > prb_thr]
 
    results = {}
 
    for cls in labels:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
 
        completezza = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        purezza     = TP / (TP + FP) if (TP + FP) > 0 else 0.0
 
        results[cls] = {
            'completezza': completezza,
            'purezza':     purezza,
            'TP': int(TP),
            'FP': int(FP),
            'FN': int(FN),
        }
 
    return results
"""

def compute_completeness_purity(y_true, y_pred, labels, probs, prb_thr):
    """
    Calcola completezza e purezza per ogni classe, ed in aggiunta la
    completezza/purezza micro-average sull'intero dataset passato
    alla funzione.
 
    Definizioni
    -----------
    Completezza (recall)  per classe C:
        = TP_C / (TP_C + FN_C)
        = (oggetti di classe C classificati correttamente) /
          (tutti gli oggetti veri di classe C)
 
    Purezza (precision)  per classe C:
        = TP_C / (TP_C + FP_C)
        = (oggetti di classe C classificati correttamente) /
          (tutti gli oggetti classificati come classe C)

    Completezza/Purezza globali (micro-average)
    ---------------------------------------------
    Si sommano TP, FP, FN su tutte le classi e si calcola un'unica
    completezza/purezza globale:
        completezza_micro = TP_tot / (TP_tot + FN_tot)
        purezza_micro     = TP_tot / (TP_tot + FP_tot)
    Equivale all'accuratezza complessiva quando ogni oggetto riceve
    sempre una predizione tra le classi note (in quel caso FP totali
    = FN totali, percio' completezza_micro = purezza_micro).
 
    Parametri
    ----------
    y_true : array-like di label veri
    y_pred : array-like di label predetti
    labels : lista ordinata di tutti i label
    probs  : array-like delle probabilita' (confidenze) associate
             a ciascuna predizione
    prb_thr: soglia di probabilita' sotto la quale un oggetto viene
             escluso dal calcolo
 
    Ritorna
    -------
    results : dict  { label: {'completezza': float, 'purezza': float,
                               'TP': int, 'FP': int, 'FN': int} }
               con una chiave aggiuntiva:
               'tot' : {'completezza': float, 'purezza': float,
                          'TP': int, 'FP': int, 'FN': int}
    """

    y_true = np.asarray(y_true)[np.array(probs) > prb_thr]
    y_pred = np.asarray(y_pred)[np.array(probs) > prb_thr]
 
    results = {}

    TP_tot = 0
    FP_tot = 0
    FN_tot = 0
 
    for cls in labels:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
 
        completezza = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        purezza     = TP / (TP + FP) if (TP + FP) > 0 else 0.0
 
        results[cls] = {
            'completezza': completezza,
            'purezza':     purezza,
            'TP': int(TP),
            'FP': int(FP),
            'FN': int(FN),
        }

        TP_tot += TP
        FP_tot += FP
        FN_tot += FN

    # --- Micro-average: somma di TP/FP/FN su tutte le classi ---
    completezza_tot = TP_tot / (TP_tot + FN_tot) if (TP_tot + FN_tot) > 0 else 0.0
    purezza_tot      = TP_tot / (TP_tot + FP_tot) if (TP_tot + FP_tot) > 0 else 0.0

    results['tot'] = {
        'completezza': completezza_tot,
        'purezza':     purezza_tot,
        'TP': int(TP_tot),
        'FP': int(FP_tot),
        'FN': int(FN_tot),
    }
    
    return results
#completezza e purezza totale
#fare anche per train

#verificare se il periodo della simulazione e di decam sono differenti
#poi verificare curve di luce
#plottare distribuzione dei periodi per simulazione e per decam
results = compute_completeness_purity(target_early, early_predictions, label_names, early_probs, 70)
for cl in label_names: results[cl]


# ============================================================
# PLOT DI 5 RRc E 5 CEPHEIDI DAI DATI SIMULATI
# ============================================================

N_MAG_POINTS = 30
rng = np.random.default_rng(18)

for target_class in ["RRc", "Cep"]:

    indices = np.where(target_train.values == target_class)[0]

    if len(indices) == 0:
        print(f"Nessun oggetto {target_class} trovato nei dati simulati")
        continue

    n_examples = min(5, len(indices))
    selected = rng.choice(indices, size=n_examples, replace=False)

    fig, axes = plt.subplots(
        n_examples,
        1,
        figsize=(10, 2.5 * n_examples)
    )

    if n_examples == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected):

        phase = data_train[idx][:N_MAG_POINTS]
        mag = data_train[idx][N_MAG_POINTS:N_MAG_POINTS + 30]
        ax.plot(
            phase,
            mag,
            marker='.',
            linewidth=1
        )

        ax.set_title(
            f"{target_class} - simulazione #{idx}",
            fontsize=10
        )

        ax.set_xlabel("Phase")
        ax.set_ylabel("Mag")
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{target_class}_simulated_examples.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.close()


# ============================================================
# PLOT DI 5 RRc E 5 CEPHEIDI DA DECam
# LE CLASSI SONO QUELLE PREDTTE DALLE PROBABILITY MAPS
# ============================================================

decam_predictions = np.array(decam_predictions)
"""
for target_class in ["RRc", "Cep", "EB", "EA", "EW"]:

    indices = np.where(decam_predictions == target_class)[0]

    if len(indices) == 0:
        print(f"Nessun oggetto DECam classificato come {target_class}")
        continue

    n_examples = min(5, len(indices))
    selected = rng.choice(indices, size=n_examples, replace=False)

    fig, axes = plt.subplots(
        n_examples,
        1,
        figsize=(10, 2.5 * n_examples)
    )

    if n_examples == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected):
        
        phase = data_train[idx][:N_MAG_POINTS]
        mag = data_train[idx][N_MAG_POINTS:N_MAG_POINTS + 30]

        ax.plot(
            phase,
            mag,
            marker='.',
            linewidth=1
        )
        ax.set_title(
            f"{target_class} - DECam #{decam_probs[idx]}",
            fontsize=10
        )

        ax.set_xlabel("Point")
        ax.set_ylabel("Mag")
        ax.grid(alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        f"{target_class}_decam_examples.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
"""

# ============================================================
# PLOT DI RRab, RRc, CEP, EA, EB, EW DA DECam
# LE CLASSI SONO QUELLE PREDETTE DALLE PROBABILITY MAPS
# ============================================================

decam_predictions = np.array(decam_predictions)

for target_class in ["RRab", "RRc", "Cep", "EA", "EB", "EW"]:

    indices = np.where(decam_predictions == target_class)[0]

    if len(indices) == 0:
        print(f"Nessun oggetto DECam classificato come {target_class}")
        continue

    n_examples = min(5, len(indices))
    selected = rng.choice(indices, size=n_examples, replace=False)

    fig, axes = plt.subplots(
        n_examples,
        1,
        figsize=(10, 2.5 * n_examples)
    )

    if n_examples == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected):

        phase = data_decam[idx][:N_MAG_POINTS]
        mag = data_decam[idx][N_MAG_POINTS:N_MAG_POINTS + 30]

        ax.scatter(
            np.concatenate([phase,phase+1]),
            np.concatenate([mag, mag]),
            s=20,
            marker='o'
        )
        ax.set_title(
            f"{target_class} - DECam - Prob {decam_probs[idx]}",
            fontsize=10
        )

        ax.set_xlabel("Phases")
        ax.set_ylabel("Mag")
        ax.grid(alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        f"{target_class}_decam_examples.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

stop()
