'''

Questo script Python genera curve di luce sintetiche per diverse classi di stelle variabili (RRL, Cep, EB). 
Implementa modelli analitici ed espansioni in serie di Fourier per simulare curve di luce con campionamento non uniforme e rumore fotometrico gaussiano.


1. Funzioni di base
- generate_observation_times(n_points, duration_days)
Funzione: Genera la baseline temporale (epoca delle osservazioni).
Metodo: Campionamento da una distribuzione uniforme [0,duration_days], seguito da ordinamento crescente per simulare osservazioni sequenziali non equispaziate.
- fourier_lightcurve(phases, amplitudes, phases_offsets, target_amplitude)
Funzione: Sintetizza il profilo della curva di luce.
Metodo: Espansione in serie di Fourier troncata. Prende in input i coefficienti di ampiezza e fase. Include un algoritmo di normalizzazione che scala la curva 
risultante per farle corrispondere l'esatta target_amplitude richiesta.


2. Funzioni simulazione lc per varibili pulsanti
- simulate_rr_lyrae_ab(n_points)
Profilo: Fortemente asimmetrico (rapida ascesa, lento declino).
Parametri: Modello ad alta frequenza spaziale (6 armoniche). Periodo: 0.4−1.0 giorni.
- simulate_rr_lyrae_c(n_points)
Profilo: Pseudo-sinusoidale.
Parametri: Bassa frequenza spaziale (3 armoniche). Periodo: 0.2−0.45 giorni.
- simulate_cepheid_classical(n_points)
Profilo: Asimmetrico classico (instabilità di supergigante).
Parametri: 4 armoniche. Periodo: 5.0−15.0 giorni.
- simulate_cepheid_type_II(n_points)
Profilo: Variazioni con morfologia specifica per Cefeidi di Popolazione II.
Parametri: 4 armoniche. Periodo: 10.0−25.0 giorni.

3. Funzioni simulazione lc per EB
- simulate_binary_detached(n_points)
Configurazione: Sistema binario staccato (no scambio materia).
Modello: Somma della magnitudine di base con due profili di attenuazione esponenziale (gaussiane) per minimi primari e secondari. Larghezza delle eclissi ridotta.
- simulate_binary_semidetached(n_points)
Configurazione: Lobo di Roche riempito da una componente (gigante + stella piccola).
Modello: Analogo al sistema detached, ma con un parametro eclipse_width maggiorato per simulare transizioni fotometriche meno nette.
- simulate_binary_contact(n_points)
Configurazione: Inviluppo comune (mutuo scambio di materia).
Modello: Variazione continua modellata tramite una funzione coseno al quadrato per le distorsioni ellissoidali.


4. Output: Ogni funzione restituisce una tupla di 4 elementi: phase, mag, err, stats.
- phase: Fase orbitale/pulsazionale calcolata sul periodo ($[0, 1)$).
- mag: Magnitudine apparente simulata con rumore gaussiano per simulare l'errore strumentale.
- err: Incertezza fotometrica stimata per ogni punto (in questa simulazione, un array costante pari a mag_err).
- stats: Contiene i parametri utilizzati dalla simulazione e generati casualmente (es. Periodo P, Ampiezza Amp, Magnitudine baseline Mag, Shift di fase Phi0).
'''


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path #per creare cartella
import csv
import json
import random
from pdb import set_trace as stop

# ==========================================
# FUNZIONI 
# ==========================================

def generate_observation_times(n_points, duration_days):
    t = np.random.uniform(0, duration_days, n_points)
    return np.sort(t)

def fourier_lightcurve(phases, amplitudes, phases_offsets, target_amplitude):
    curve = np.zeros_like(phases)
    for i, (A, Phi) in enumerate(zip(amplitudes, phases_offsets)):
        n = i + 1 
        curve += A * np.cos(2 * np.pi * n * phases + Phi)
    
    curve -= np.mean(curve)
    current_amp = np.max(curve) - np.min(curve)
    
    if current_amp > 0:
        curve = curve * (target_amplitude / current_amp)
    return curve

def save_lightcurve_txt(filename, phase, mag, err):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True) #crea directory e se ci sono cartelle annidate crea quelle mancanti
    data = np.column_stack((phase, mag, err))
    np.savetxt(
        filename,
        data,
        fmt="%.6f",#ovvero si intende numero in formato decimale con 6 cifre dopo la virgola
        header= "phase mag err",
        comments= ""
    )

def plot_lightcurve(lc_name, phase ,mag):
    plt.scatter(np.concatenate[phase, phase + 1], np.concatenate[mag, mag])
    plt.title(lc_name)
    plt.gca().invert_yaxis()
    plt.show()


# ==========================================
# SIMULAZIONE LC VARIABILI PULSANTI (RRLyrae, Cefeidi, Altre ...TBD)
# ==========================================

def simulate_rr_lyrae_ab(n_points):
    period = np.random.uniform(0.2, 1.2)#0.4, 1
    baseline_mag = np.random.uniform(14.0, 18.0)
    amplitude = np.random.uniform(0.5, 1.5)
    mag_err = np.random.uniform(0.02, 0.05)
    phase_shift = np.random.uniform(0.0, 1.0) 
    
    A_coeffs = [1.0, 0.45, 0.30, 0.18, 0.10, 0.05]
    Phi_coeffs = [0.0, 3.9, 1.8, 6.0, 3.8, 1.5]
    
    t = generate_observation_times(n_points, duration_days=15)
    phase = (t % period) / period # fase
    shifted_phase = (phase + phase_shift) % 1.0 # shifto la fase di phase_shift
    
    clean_mag = baseline_mag + fourier_lightcurve(shifted_phase, A_coeffs, Phi_coeffs, amplitude)
    stats = {'P': period, 'Amp': amplitude, 'Mag': baseline_mag, 'Phi0': phase_shift}
    
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats

def simulate_rr_lyrae_c(n_points):
    period = np.random.uniform(0.2, 0.45)
    baseline_mag = np.random.uniform(16.0, 22.0)
    amplitude = np.random.uniform(0.2, 0.6)
    mag_err = np.random.uniform(0.02, 0.05)
    phase_shift = np.random.uniform(0.0, 1.0)
    
    A_coeffs = [1.0, 0.15, 0.05]
    Phi_coeffs = [0.0, 4.7, 3.5]
    
    t = generate_observation_times(n_points, duration_days=5)
    phase = (t % period) / period
    shifted_phase = (phase + phase_shift) % 1.0
    
    clean_mag = baseline_mag + fourier_lightcurve(shifted_phase, A_coeffs, Phi_coeffs, amplitude)
    stats = {'P': period, 'Amp': amplitude, 'Mag': baseline_mag, 'Phi0': phase_shift}
    
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats

def simulate_cepheid_classical(n_points):
    period = np.random.uniform(0.5, 15.0) #5,15
    baseline_mag = np.random.uniform(16.0, 22.0)
    amplitude = np.random.uniform(0.5, 1.5)
    mag_err = np.random.uniform(0.01, 0.04)
    phase_shift = np.random.uniform(0.0, 1.0)
    
    A_coeffs = [1.0, 0.40, 0.18, 0.08]
    Phi_coeffs = [0.0, 4.8, 3.3, 1.6]
    
    t = generate_observation_times(n_points, duration_days=period * 5)
    phase = (t % period) / period
    shifted_phase = (phase + phase_shift) % 1.0
    
    clean_mag = baseline_mag + fourier_lightcurve(shifted_phase, A_coeffs, Phi_coeffs, amplitude)
    stats = {'P': period, 'Amp': amplitude, 'Mag': baseline_mag, 'Phi0': phase_shift}
    
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats

def simulate_cepheid_type_II(n_points):
    period = np.random.uniform(10.0, 25.0)
    baseline_mag = np.random.uniform(16.0, 22.0)
    amplitude = np.random.uniform(0.4, 1.2)
    mag_err = np.random.uniform(0.02, 0.05)
    phase_shift = np.random.uniform(0.0, 1.0)
    
    A_coeffs = [1.0, 0.45, 0.25, 0.10]
    Phi_coeffs = [0.0, 5.5, 4.5, 3.0]
    
    t = generate_observation_times(n_points, duration_days=period * 4)
    phase = (t % period) / period
    shifted_phase = (phase + phase_shift) % 1.0
    
    clean_mag = baseline_mag + fourier_lightcurve(shifted_phase, A_coeffs, Phi_coeffs, amplitude)
    stats = {'P': period, 'Amp': amplitude, 'Mag': baseline_mag, 'Phi0': phase_shift}
    
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats

# ==========================================
# SIMULAZIONE LC BINARIE AD ECLISSE (staccate, semidetached, contact)
# ==========================================

def simulate_binary_detached(n_points):
    period = np.random.uniform(1.0, 10.0)
    baseline_mag = np.random.uniform(16.0, 22.0)
    primary_depth = np.random.uniform(0.8, 2.5)
    secondary_depth = np.random.uniform(0.1, primary_depth * 0.8)
    eclipse_width = np.random.uniform(0.02, 0.04)
    mag_err = np.random.uniform(0.015, 0.04)
    phase_shift = np.random.uniform(0.0, 1.0)
    
    t = generate_observation_times(n_points, duration_days=period * 10)
    phase = (t % period) / period
    shifted_phase = (phase + phase_shift) % 1.0
    
    dist_to_primary = np.minimum(shifted_phase, 1 - shifted_phase)
    dist_to_secondary = np.abs(shifted_phase - 0.5)
    
    clean_mag = baseline_mag \
                + primary_depth * np.exp(-0.5 * (dist_to_primary / eclipse_width)**2) \
                + secondary_depth * np.exp(-0.5 * (dist_to_secondary / eclipse_width)**2)
                
    stats = {'P': period, 'Dip1': primary_depth, 'Mag': baseline_mag, 'Phi0': phase_shift}
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats

def simulate_binary_semidetached(n_points):
    period = np.random.uniform(0.5, 5.0)
    baseline_mag = np.random.uniform(16.0, 22.0)
    primary_depth = np.random.uniform(0.8, 1.5)
    secondary_depth = np.random.uniform(0.1, primary_depth * 0.8)
    eclipse_width = np.random.uniform(0.08, 0.1)
    mag_err = np.random.uniform(0.015, 0.04)
    phase_shift = np.random.uniform(0.0, 1.0)
    
    t = generate_observation_times(n_points, duration_days=period * 10)
    phase = (t % period) / period
    shifted_phase = (phase + phase_shift) % 1.0
    
    dist_to_primary = np.minimum(shifted_phase, 1 - shifted_phase)
    dist_to_secondary = np.abs(shifted_phase - 0.5)
    
    clean_mag = baseline_mag \
                + primary_depth * np.exp(-0.5 * (dist_to_primary / eclipse_width)**2) \
                + secondary_depth * np.exp(-0.5 * (dist_to_secondary / eclipse_width)**2)
                
    stats = {'P': period, 'Dip1': primary_depth, 'Mag': baseline_mag, 'Phi0': phase_shift}
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats


def simulate_binary_contact(n_points):
    period = np.random.uniform(0.5, 5.0)
    baseline_mag = np.random.uniform(16.0, 22.0)
    primary_depth = np.random.uniform(0.8, 1.5)
    secondary_depth = np.random.uniform(primary_depth * 0.85, primary_depth * 0.95)
    eclipse_width = np.random.uniform(0.08, 0.1)
    mag_err = np.random.uniform(0.015, 0.04)
    phase_shift = np.random.uniform(0.0, 1.0)

    t = generate_observation_times(n_points, duration_days=period * 10)
    phase = (t % period) / period
    shifted_phase = (phase + phase_shift) % 1.0

    dist_to_primary = np.minimum(shifted_phase, 1 - shifted_phase)
    dist_to_secondary = np.abs(shifted_phase - 0.5)

    clean_mag = baseline_mag \
                + primary_depth * np.exp(-0.5 * (dist_to_primary / eclipse_width) ** 2) \
                + secondary_depth * np.exp(-0.5 * (dist_to_secondary / eclipse_width) ** 2)

    stats = {'P': period, 'Dip1': primary_depth, 'Mag': baseline_mag, 'Phi0': phase_shift}
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats


def simulate_binary_contact2(n_points):
    period = np.random.uniform(0.2, 0.8)
    baseline_mag = np.random.uniform(16.0, 22.0)
    amplitude = np.random.uniform(0.2, 0.8)
    mag_err = np.random.uniform(0.015, 0.04)
    phase_shift = np.random.uniform(0.0, 1.0)
    
    t = generate_observation_times(n_points, duration_days=period * 15)
    phase = (t % period) / period
    shifted_phase = (phase + phase_shift) % 1.0
    
    oconnell_effect = np.random.uniform(0.01, 0.08)
    clean_mag = baseline_mag + amplitude * (np.cos(2 * np.pi * shifted_phase)**2) + oconnell_effect * np.cos(2 * np.pi * shifted_phase)
    
    stats = {'P': period, 'Amp': amplitude, 'Mag': baseline_mag, 'Phi0': phase_shift}
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats

"""
def simulate_microlensing(n_points):
    # Curva di Paczynski per una sorgente puntiforme.
    # u e' espresso in unita' dell'angolo di Einstein theta_E.
    t_E = np.random.uniform(10.0, 100.0)       # tempo di Einstein [giorni]
    t0 = np.random.uniform(0.4, 0.6) * 6 * t_E # epoca del massimo allineamento
    u0 = np.random.uniform(0.01, 1.0)          # minimo parametro d'impatto
    baseline_mag = np.random.uniform(16.0, 22.0)
    mag_err = np.random.uniform(0.01, 0.04)

    t = generate_observation_times(n_points, duration_days=6 * t_E)
    phase = t / (6 * t_E)

    u = np.sqrt(u0**2 + ((t - t0) / t_E)**2)
    magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

    clean_mag = baseline_mag - 2.5 * np.log10(magnification)

    stats = {
        'P': t_E,
        'Amp': np.max(clean_mag) - np.min(clean_mag),
        'Mag': baseline_mag,
        'T0': t0,
        'u0': u0,
        'tE': t_E,
        'Amax': np.max(magnification)
    }
"""


def simulate_microlensing(n_points):
    """
    Simula una curva di luce di microlensing con curva di Paczynski.

    La curva contiene:
    - una parte iniziale quasi piatta alla magnitudine di baseline
    - una fase centrale di amplificazione
    - una parte finale di nuovo quasi piatta alla baseline

    Output compatibile con le altre funzioni:
        phase, mag, err, stats
    """

    # Parametri fisici/fotometrici dell'evento
    t_E = np.random.uniform(10.0, 100.0)      # tempo di Einstein [giorni]
    baseline_mag = np.random.uniform(16.0, 22.0)
    u0 = np.random.uniform(0.4, 0.9)         # parametro d'impatto minimo
    mag_err = np.random.uniform(0.01, 0.04)

    # Baseline temporale lunga: da -5 t_E a +5 t_E
    # In questo modo inizio e fine mostrano bene la magnitudine costante.
    tau_min = -6.0
    tau_max = 6.0

    # Campionamento diviso in tre zone:
    # ali iniziali, regione centrale amplificata, ali finali.
    n_left = n_points // 3
    n_center = n_points // 3
    n_right = n_points - n_left - n_center

    tau_left = np.random.uniform(tau_min, -2.0, n_left)
    tau_center = np.random.uniform(-2.0, 2.0, n_center)
    tau_right = np.random.uniform(2.0, tau_max, n_right)

    tau = np.sort(np.concatenate([tau_left, tau_center, tau_right]))

    # Tempo fisico: tau = (t - t0) / t_E
    t0 = 5.0 * t_E
    t = t0 + tau * t_E

    # Asse normalizzato 0-1, utile per salvarlo come "phase"
    phase = (tau - tau_min) / (tau_max - tau_min)

    # Separazione angolare lente-sorgente in unita' di theta_E
    u = np.sqrt(u0**2 + tau**2)

    # Magnification factor di Paczynski, sorgente puntiforme
    magnification = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))

    # Conversione da amplificazione di flusso a magnitudine
    clean_mag = baseline_mag - 2.5 * np.log10(magnification)

    mag = clean_mag + np.random.normal(0, mag_err, n_points)
    err = np.full(n_points, mag_err)

    stats = {
        'P': t_E,  # per compatibilita' con il dataset esistente
        'tE': t_E,
        'T0': t0,
        'u0': u0,
        'Mag': baseline_mag,
        'Amp': np.max(clean_mag) - np.min(clean_mag),
        'Amax': np.max(magnification)
    }

    return phase, mag, err, stats
"""
    return phase, clean_mag + np.random.normal(0, mag_err, n_points), np.full(n_points, mag_err), stats
    #vedere curva di luce da letteratura, guardare magnification factor,
    #costruire curva di luce che faccia variare u in funzione del tempo facendo variare angolo tra sorgente e lente
    #plottare direttamente il grafico del magnification factor, u è espresso in angolo di Einstein 
    #stiamo assumendo sorgente puntiforme
    #curva Paczynksi
    #riscrivere u(t) in termini del tempo di einstein 
    #simulare variando tempo di einstein tra un range 
"""





################################################
# MAIN CODE
################################################

N_INPUT_POINTS = 30



# Lista di tutte le classi da plottare: (nome file, titolo, funzione di simulazione)
lc_classes = [
    ("RR_lyrae_AB.png",       "RR Lyrae AB",        simulate_rr_lyrae_ab),
    ("RR_lyrae_C.png",        "RR Lyrae C and δ Scuti",         simulate_rr_lyrae_c),
    ("Classical_Cepheids.png","Classical Cepheid",   simulate_cepheid_classical),
    ("Cepheids_TypeII.png",   "Type II Cepheid",    simulate_cepheid_type_II),
    ("Binary_contact.png",    "Contact Eclipsing Binary", simulate_binary_contact),
    ("Binary_detached.png",   "Detached Eclipsing Binary",   simulate_binary_detached),
    ("Binary_semidetached.png","Semidetached Eclipsing Binary", simulate_binary_semidetached),
]
 
# Cartella di output per le immagini
out_dir = Path("lightcurves")
out_dir.mkdir(parents=True, exist_ok=True)
 
# Figura combinata: griglia 4 righe x 2 colonne (una cella per ciascuna classe)
N_ROWS, N_COLS = 4, 2
fig, axes = plt.subplots(
    nrows=N_ROWS,
    ncols=N_COLS,
    figsize=(6 * N_COLS, 3 * N_ROWS),
    squeeze=False
)
 
for idx, (filename, title, sim_func) in enumerate(lc_classes):
    phase, mag, err, stats = sim_func(N_INPUT_POINTS)
 
    # --- Salvataggio della singola immagine (come nello script originale) ---
    plt.figure()
    plt.scatter(np.concatenate([phase, phase + 1]), np.concatenate([mag, mag]))
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig(out_dir / filename)
    plt.close()
 
    # --- Aggiunta della stessa curva alla cella corrispondente della griglia ---
    row, col = divmod(idx, N_COLS)
    ax = axes[row][col]
    ax.scatter(np.concatenate([phase, phase + 1]), np.concatenate([mag, mag]), s=10)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mag")

phase, mag, err, stats = simulate_microlensing(N_INPUT_POINTS)
ax = axes[3][1]
ax.scatter(phase, mag, s=10)
ax.set_title("Microlensing")
ax.invert_yaxis()
ax.set_xlabel("Phase")
ax.set_ylabel("Mag")

"""
# Nasconde eventuali ce"lle vuote (se il numero di classi non riempie la griglia)
for idx in range(len(lc_classes), N_ROWS * N_COLS):
    row, col = divmod(idx, N_COLS)
    axes[row][col].axis("off")
"""
fig.tight_layout()
fig.savefig(out_dir / "all_lightcurves_combined.png", dpi=150)
plt.close(fig)
 
print("Immagini singole e figura combinata salvate in", out_dir.resolve())
stop()


"""
phase, mag, err, stats = simulate_rr_lyrae_ab(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("RR Lyrae AB")
plt.gca().invert_yaxis()
plt.savefig("RR_lyrae_AB.png")
plt.close()


phase, mag, err, stats = simulate_rr_lyrae_c(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("RR Lyrae C")
plt.gca().invert_yaxis()
plt.savefig("RR_lyrae_C.png")
plt.close()


phase, mag, err, stats = simulate_cepheid_classical(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Classical Cepheid")
plt.gca().invert_yaxis()
plt.savefig("Classical_Cepheids.png")
plt.close()

phase, mag, err, stats = simulate_cepheid_type_II(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Type II Cepheid")
plt.gca().invert_yaxis()
plt.savefig("Cepheids_TypeII.png")
plt.close()


phase, mag, err, stats = simulate_binary_contact(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Eclipsing Binary Contact")
plt.gca().invert_yaxis()
plt.savefig("Binary_contact.png")
plt.close()

phase, mag, err, stats = simulate_binary_detached(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Eclipsing Binary Detached")
plt.gca().invert_yaxis()
plt.savefig("Binary_detached.png")
plt.close()

phase, mag, err, stats = simulate_binary_semidetached(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Eclipsing Binary Semidetached")
plt.gca().invert_yaxis()
plt.savefig("Binary_semidetached.png")
plt.close()


phase, mag, err, stats = simulate_microlensing(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Microlensing")
plt.gca().invert_yaxis()
plt.savefig("microlensing.png")
plt.close()
stop()

"""
"""
phase, mag, err, stats = simulate_rr_lyrae_ab(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("RR Lyrae AB")
plt.gca().invert_yaxis()
plt.show()
phase, mag, err, stats = simulate_rr_lyrae_c(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("RR Lyrae C")
plt.gca().invert_yaxis()
plt.show()

phase, mag, err, stats = simulate_cepheid_classical(N_INPUT_POINTS)
plt.scatter(phase, mag)
plt.title("Cefeide Classica")
plt.gca().invert_yaxis()
plt.savefig("cefeide1.png")
plt.close()

plt.plot(mag[np.argsort(phase)], '.')
plt.title("Cefeide Classica")
plt.gca().invert_yaxis()
plt.savefig("cefeide2.png")
plt.close()


phase, mag, err, stats = simulate_cepheid_type_II(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Cefeide Tipo II")
plt.gca().invert_yaxis()f
plt.show()
phase, mag, err, stats = simulate_binary_detached(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Binaria staccata")
plt.gca().invert_yaxis()
plt.show()
phase, mag, err, stats = simulate_binary_semidetached(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Binaria Semistaccata")
plt.gca().invert_yaxis()
plt.show()
phase, mag, err, stats = simulate_binary_contact(N_INPUT_POINTS)
plt.scatter(np.concatenate([phase,phase + 1]), np.concatenate([mag,mag]))
plt.title("Binaria a contatto")
plt.gca().invert_yaxis()
plt.show()
"""
# Lista unica di esempio (una riga per curva di luce)
# Ogni elemento: [mag, P, A, tipo_variabile]
dataset_lc = []
#fare 10000 per classe 
for i in range(1000):
    phase, mag, err, stats = simulate_rr_lyrae_ab(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_rr_lyrae_ab" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) +','+'RRab'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Dip1")), "RRab"])
    dataset_lc.append(line)

    phase, mag, err, stats = simulate_rr_lyrae_c(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_rr_lyrae_c" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) + ',' + 'RRc'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Amp")), "RRc"])
    dataset_lc.append(line)

    phase, mag, err, stats = simulate_cepheid_classical(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_cepheid_classical" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) + ',' + 'Cep'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Amp")), "ClasCep"])
    dataset_lc.append(line)

    phase, mag, err, stats = simulate_cepheid_type_II(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_cepheid_type_II" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) + ',' + 'T2Cep'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Amp")), "T2Cep"])
    dataset_lc.append(line)

    phase, mag, err, stats = simulate_binary_detached(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_binary_detached" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) + ',' + 'EA'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Dip1")), "EA"])
    dataset_lc.append(line)

    phase, mag, err, stats = simulate_binary_semidetached(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_binary_semidetached" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) + ',' + 'EB'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Dip1")), "EB"])
    dataset_lc.append(line)

    phase, mag, err, stats = simulate_binary_contact(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_binary_contact" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) + ',' + 'EW'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Dip1")), "EW"])
    dataset_lc.append(line)

    phase, mag, err, stats = simulate_microlensing(N_INPUT_POINTS)
    mag = mag[np.argsort(phase)]
    phase = phase[np.argsort(phase)]
    err = err[np.argsort(phase)]
    save_lightcurve_txt("lightcurves/lc_microlensing" + str(i) + ".txt", phase, mag, err)
    line = ''
    for m in phase: line = line + str(m) + ','
    for m in mag: line = line + str(m) + ','
    line = line + str(stats["P"]) + ',' + str(stats.get("Amp", stats.get("Dip1"))) + ',' + 'LENS'
    #dataset_lc.append([','.join([str(i) for i in mag]), stats["P"], stats.get("Amp", stats.get("Dip1")), "EW"])
    dataset_lc.append(line)


random.shuffle(dataset_lc)

file = open('lightcurves/dataset_lc_test_2.csv', 'w')

nn = np.arange(1,64)
ll = ''
for l in nn: 
    ll = ll + str(l) + ','
ll = ll[:-1] + '\n'
file.write(ll)

#file.write("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33" + '\n')
for line in dataset_lc:
    file.write(line+'\n')

file.close()

stop()




