# Template RunPod — miniDS

## 1. Construire et publier l'image

```bash
docker build -f docker/Dockerfile -t ghcr.io/<utilisateur>/minids:latest .
docker push ghcr.io/<utilisateur>/minids:latest
```

La compilation des noyaux CUDA de gsplat prend 10–20 min. `TORCH_CUDA_ARCH_LIST`
est déjà réglé pour A100 / RTX 30xx / L4 / RTX 4090 / H100, donc l'image
fonctionne quel que soit le GPU loué.

## 2. Réglages du template (Pods → Templates → New Template)

| Champ | Valeur |
|---|---|
| Container Image | `ghcr.io/<utilisateur>/minids:latest` |
| Container Disk | **60 Go** |
| Volume Disk | **50 Go**, monté sur `/workspace` |
| **Expose HTTP Ports** | **`8000`** ← indispensable |
| Expose TCP Ports | *(vide)* |
| Container Start Command | *(vide — l'ENTRYPOINT s'en charge)* |

### Variables d'environnement

| Nom | Obligatoire | Rôle |
|---|---|---|
| `HF_TOKEN` | oui | Téléchargement des poids VGGT-Ω (**accès restreint**, à demander avant) et SAM 3 |
| `MINIDS_CKPT` | oui | `facebook/VGGT-Omega:vggt_omega_1b_512.pt`, ou un chemin local sur le volume |
| `MINIDS_TOKEN` | recommandé | Secret d'accès à l'API. Si absent, il est **généré et affiché dans les logs du pod** |
| `MINIDS_FRAMES` | non | Nombre d'images par défaut (120) |
| `MINIDS_IMAGE_RESOLUTION` | non | Résolution VGGT-Ω (512) |
| `MINIDS_LOG_LEVEL` | non | `info` par défaut |

> L'URL du proxy est publique : **ne jamais laisser `MINIDS_TOKEN` vide sans lire
> le token généré dans les logs.** Sans token configuré, l'API répond 503 à tout.

## 3. Choix du GPU

| GPU | VRAM | Verdict |
|---|---|---|
| **RTX 4090** | 24 Go | **Recommandé.** Le 2DGS est limité par le calcul, pas la mémoire |
| L4 | 24 Go | Fonctionne, ~2,5× plus lent sur l'étape de raffinement |
| RTX A5000 / 3090 | 24 Go | Équivalent L4 |
| A100 40 Go | 40 Go | Utile seulement au-delà de ~250 images |

Consommation VRAM de VGGT-Ω (table du dépôt officiel, entrées 624×416) :
100 images ≈ 13,4 Go · 200 ≈ 20,8 Go · 300 ≈ 28,3 Go. Avec le défaut de
**120 images**, on reste sous 16 Go, donc 24 Go suffisent largement.

## 4. Premier démarrage

1. Lancer le pod, ouvrir ses **logs** (pas de terminal nécessaire).
2. Y lire `MINIDS_TOKEN` s'il a été généré.
3. Récupérer l'ID du pod, l'URL est `https://<POD_ID>-8000.proxy.runpod.net`.
4. Vérifier depuis Windows :

```powershell
$env:MINIDS_URL="https://<POD_ID>-8000.proxy.runpod.net"
$env:MINIDS_TOKEN="<token>"
python client/minids.py health
```

Réponse attendue : `cuda_available: true`, le nom du GPU et la VRAM libre.

## 5. Dépannage

| Symptôme | Cause | Correction |
|---|---|---|
| **502 Bad Gateway** | serveur pas encore prêt | attendre ~60 s (chargement des poids), puis réessayer |
| 502 persistant | port non exposé | ajouter `8000` dans **Expose HTTP Ports** du template |
| **503 MINIDS_TOKEN non configuré** | variable absente | définir `MINIDS_TOKEN`, ou lire celui des logs |
| **401** | token client ≠ token pod | réaligner `MINIDS_TOKEN` des deux côtés |
| Job `failed` sur `vggt` | `MINIDS_CKPT` absent ou accès HF refusé | vérifier l'approbation Hugging Face et `HF_TOKEN` |
| Coupure à ~100 s | requête longue | c'est la limite Cloudflare ; le client la contourne déjà, ne pas appeler l'API à la main avec `curl` sur de gros fichiers |
| `CUDA out of memory` sur `vggt` | trop d'images | baisser `--frames` (100 tient sur 24 Go avec marge) |

## 6. Coût indicatif

RTX 4090 à ~0,69 $/h · scan de 120 images ≈ 15–20 min → **≈ 0,20 $ par objet**.
Penser à **arrêter le pod** entre deux scans : le volume `/workspace` conserve le
cache des poids, le redémarrage suivant est rapide.
