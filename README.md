# Simulateur Échographique B-Mode (Python)

Ce projet est un simulateur léger d'imagerie ultrasonore (échographie) codé en Python. Il génère des données brutes RF (Radio-Frequency) et reconstitue des images B-mode via un beamforming DAS (Delay-and-Sum) classique.

Le simulateur modélise une émission en **Onde Plane (Plane Wave)** et inclut la gestion du *speckle* et de cibles ponctuelles (points brillants).

## 📋 Fonctionnalités
- **Simulation Physique :** Modélisation de la réponse impulsionnelle, délais de vol, et atténuation géométrique.
- **Beamforming :** Reconstruction d'image par méthode "Delay-and-Sum" (DAS).
- **Correction d'Artefacts :** Gestion des effets de bord de la transformée de Hilbert (Zero-padding) et respect du critère de Nyquist spatial.
- **Export de Données :** Sauvegarde des données brutes (RF) et traitées (Enveloppe, B-mode) au format `.h5` (HDF5).
- **Visualisation :** Génération automatique des images `.png`.

## ⚙️ Installation

Assurez-vous d'avoir Python installé (3.8+ recommandé). Installez les dépendances nécessaires :

```bash
pip install numpy scipy matplotlib h5py
