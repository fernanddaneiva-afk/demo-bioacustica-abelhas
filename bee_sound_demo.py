"""Demo de classificação de sons de abelhas com SINAIS SINTÉTICOS.

Simula dois comportamentos (voo e vibração/sonicação em flor) como tons com
harmônicos e ruído, extrai features espectrais e treina um classificador.
Não usa dados reais nem código de nenhum projeto de pesquisa.
"""
import numpy as np
from scipy.signal import spectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

SR = 8000
DUR = 1.0
rng = np.random.default_rng(0)


def synth(f0, harmonics, noise):
    t = np.arange(int(SR * DUR)) / SR
    f = f0 * (1 + 0.03 * rng.standard_normal())
    sig = sum(np.sin(2 * np.pi * f * (k + 1) * t) / (k + 1) for k in range(harmonics))
    return sig + noise * rng.standard_normal(t.size)


def features(sig, n_bands=16):
    f, _, S = spectrogram(sig, fs=SR, nperseg=512)
    P = S.mean(axis=1)
    bands = np.array_split(P, n_bands)
    band_e = np.log10(np.array([b.sum() for b in bands]) + 1e-12)
    centroid = (f * P).sum() / P.sum()
    return np.r_[band_e, centroid]


def dataset(n=200):
    X, y = [], []
    for _ in range(n):
        X.append(features(synth(200, 4, 0.5))); y.append(0)   # voo
        X.append(features(synth(320, 6, 0.5))); y.append(1)   # sonicação
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = dataset()
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    acc = cross_val_score(clf, X, y, cv=5)
    print(f"Acurácia 5-fold (dados sintéticos): {acc.mean():.3f} ± {acc.std():.3f}")
