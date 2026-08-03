# miniDS

Vidéo prise au téléphone → objet 3D `.glb`, via un pod RunPod.
VGGT-Ω pour les poses et la profondeur, raffinement 2D Gaussian Splatting,
fusion TSDF, texture bakée. **Aucun terminal web sur le pod** : tout passe par
une CLI locale.

```powershell
python client/minids.py run mon-objet.mp4 --prompt "the sneaker"
# → out/<job>/mesh.glb
```

---

## Le TSDF est-il dépassé ?

C'était la question posée. Réponse courte : **non, mais il ne suffit pas.**

Le TSDF reste l'étape d'extraction du SOTA 2026 — [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
(SIGGRAPH'24), GS2Mesh et G3Splat (déc. 2025) finissent tous par une fusion TSDF.
Ce qui a changé en dix ans, ce n'est pas l'algorithme de fusion : c'est **la
qualité des profondeurs qu'on lui donne**. Le TSDF moyenne ce qu'il reçoit ; il
n'invente rien et ne débruite presque rien.

D'où la conception retenue : on ne remplace pas le TSDF, **on améliore son
entrée** en insérant une passe 2DGS entre VGGT-Ω et la fusion.

| Backend | Temps (4090) | Qualité objet | Statut |
|---|---|---|---|
| TSDF sur profondeur VGGT-Ω brute | ~10 s | bruitée, surfaces ondulées | `--backends tsdf` |
| Poisson screened | ~20 s | lisse, mais arrondit les arêtes et gonfle les zones peu vues | `--backends poisson` |
| NKSR (CVPR'23) | ~1 min | bon sur nuages bruités | **écarté** : dépendances figées (torch/cu118, fvdb) qui cassent l'image |
| **2DGS → TSDF des profondeurs rendues** | ~12 min | arêtes nettes, bruit fortement réduit | **défaut** |
| GOF (marching tetrahedra) | ~15 min | comparable, sans réglage de `depth_trunc` | évolution possible |

Implémentation via **`gsplat`** (PyPI, Apache-2.0, expose `rasterization_2dgs`
avec rendu profondeur + normale) plutôt que le dépôt de recherche INRIA : pas de
rasterizer à compiler à la main, licence utilisable.

Ce classement vient de la littérature. Pour le vérifier **sur tes données** :

```bash
python bench/compare_meshes.py --raw out/<job>/vggt_raw.npz \
    --reference out/<job>/mesh.glb --backends tsdf,poisson --out bench_out
```

Sortie : `bench_out/report.md` — temps, triangles, étanchéité, distance de
Chamfer en % de la diagonale de l'objet, plus un `.glb` par backend.

---

## Le pipeline

| # | Étape | Ce qui se passe |
|---|---|---|
| 1 | `frames` | ffmpeg : 3× plus d'images que nécessaire, mesure du flou (`blurdetect`), on garde la plus nette de chaque intervalle. Tonemap HDR→SDR si le téléphone a filmé en HDR10+ |
| 2 | `vggt` | VGGT-Ω-1B-512, une passe sur toute la séquence → poses, profondeur, confiance |
| 3 | `segment` | SAM 3 sur prompt texte, repli géométrique (plan RANSAC + DBSCAN) si indisponible |
| 4 | `colmap` | export `sparse/0/*.txt` + nuage d'initialisation, réutilisable par n'importe quel outil de splatting |
| 5 | `refine` | 2DGS masqué, ancré au départ sur la profondeur VGGT-Ω, puis rendu des profondeurs médianes |
| 6 | `mesh` | fusion TSDF CUDA (`VoxelBlockGrid`) → marching cubes |
| 7 | `cleanup` | plus grande composante, retrait du plan de support, Taubin, décimation, bouchage des trous |
| 8 | `texture` | dépliage UV xatlas + rétro-projection depuis la meilleure vue (frontalité × netteté), visibilité par lancer de rayons |
| 9 | `export` | `mesh.glb` (glTF 2.0, texture PNG embarquée) |

**La segmentation passe après VGGT-Ω**, et non avant : le repli géométrique a
besoin de la profondeur, et travailler sur les images pré-traitées par le modèle
garantit un alignement pixel à pixel avec les intrinsèques.

L'échelle prédite par VGGT-Ω est **relative**. La scène est donc normalisée
(médiane + percentile 95) pour que la taille de voxel ait un sens stable d'un
scan à l'autre. Pour un GLB à la bonne taille : `--ref-size 0.28` (plus grande
dimension réelle, en mètres).

---

## Transport : pourquoi ce n'est pas un simple POST

Le proxy RunPod passe par Cloudflare, qui **coupe toute connexion à 100 s**. Une
inférence de 15 minutes ne peut donc pas être une requête HTTP synchrone, et un
upload de vidéo 4K non plus.

| Étape | Appel | Pourquoi ça passe |
|---|---|---|
| upload | `POST /jobs` puis `PUT /jobs/{id}/chunks/{n}` (8 Mo) | chaque chunk très en-deçà de 100 s, et repris individuellement |
| lancement | `POST /jobs/{id}/start` | retour immédiat, traitement en tâche de fond |
| suivi | `GET /jobs/{id}` toutes les 5 s | JSON minuscule |
| récupération | `GET /jobs/{id}/artifacts/{nom}` avec `Range:` | reprise depuis le `.part`, vérification sha256 |

L'extraction des images se fait **en local** : uploader 120 JPEG au lieu d'une
vidéo 4K fait passer le transfert de ~500 Mo à ~20 Mo. `--send-video` force
l'ancien comportement si tu préfères.

Artefacts disponibles : `mesh.glb`, `vggt_raw.npz` (sortie brute), `report.json`,
`preview.png`, et `mesh_<backend>.glb` si plusieurs backends ont été demandés.

---

## Installation

### Sur le pod

Voir [runpod/template.md](runpod/template.md) — image Docker, port HTTP 8000,
variables d'environnement, choix du GPU, dépannage.

**Prérequis bloquant** : les poids VGGT-Ω sont sous accès restreint sur Hugging
Face. Demander l'approbation, puis renseigner `HF_TOKEN` et `MINIDS_CKPT`.

### Sur Windows

Rien à installer côté Python — le client n'utilise que la bibliothèque standard.
Seul ffmpeg est requis :

```powershell
winget install Gyan.FFmpeg
$env:MINIDS_URL="https://<POD_ID>-8000.proxy.runpod.net"
$env:MINIDS_TOKEN="<token du pod>"
```

---

## Commandes

```powershell
python client/minids.py health                      # état du pod, GPU, VRAM
python client/minids.py run video.mp4 --prompt "the mug"
python client/minids.py run video.mp4 --also-raw    # récupère aussi vggt_raw.npz
python client/minids.py submit video.mp4            # lance sans attendre
python client/minids.py status <job> --watch        # reprend le suivi
python client/minids.py fetch <job> --all-artifacts # re-télécharge (reprise incluse)
python client/minids.py remesh vggt_raw.npz --backend poisson   # local, sans pod
```

Options utiles de `run` :

| Option | Effet |
|---|---|
| `--frames 120` | images envoyées au modèle (100 pour rester large sur 24 Go) |
| `--no-refine` | saute le 2DGS : ~1 min au lieu de ~15, qualité en retrait |
| `--backends tsdf2dgs,tsdf,poisson` | produit plusieurs maillages pour comparer |
| `--ref-size 0.28` | met le GLB à l'échelle métrique réelle |
| `--texture vertex` | couleurs par sommet au lieu d'une texture 2k |
| `--segmentation geometric` | force le repli sans SAM 3 |
| `--no-watertight` | n'essaie pas de boucher les trous |

Le `remesh` local ne nécessite que `pip install numpy open3d` et tourne sur CPU —
c'est là tout l'intérêt de rapatrier `vggt_raw.npz` : rejouer le maillage sans
relouer un GPU.

---

## Interface graphique

```powershell
pip install PyQt6 tqdm
python -m gui
```

Même client, même transport, même archive d'images : un scan lancé depuis
l'interface est comparable à un scan lancé en ligne de commande — c'est pourquoi
la préparation de la charge utile vit dans `client/payload.py`, partagée par les
deux.

| Onglet | Ce qu'on y fait |
|---|---|
| Connexion | URL et jeton du pod, test de `/health` : GPU, VRAM libre, CUDA, et repérage du mode factice. Le jeton n'est écrit sur disque que si la case est cochée |
| Scan | Tous les paramètres de `run`, avec trois préréglages (validation ~2 min, qualité ~15 min, comparatif de backends), puis suivi en direct : barre du pipeline, barre de transfert, ETA, débit, frise du temps par étape et journal du pod |
| Résultats | Aperçu rendu, triangles, étanchéité, couverture de texture, gaussiennes, débits montant et descendant, liste des artefacts, ouverture du GLB dans Open3D |
| Historique | Tous les scans passés avec leurs paramètres et leurs mesures, médianes de durée et de triangles, temps moyen par étape |

**Reprendre un scan en cours.** Le pod persiste l'état de ses jobs : fermer la
fenêtre — ou la perdre — n'interrompt pas le scan. L'identifiant du job est
mémorisé et repropose à la réouverture ; « Jobs du pod… » liste ceux que le pod
connaît, y compris ceux lancés depuis la CLI. « Rejoindre » reprend le suivi
puis télécharge les artefacts. Sur un scan de 15 minutes de GPU facturé, c'est
la différence entre un incident et une contrariété.

Les réglages et l'historique sont rangés dans `~/.minids/`, hors du dépôt.

---

## Conseils de prise de vue

Ce sont eux qui font la différence, bien plus que les hyperparamètres :

- tour complet **lent** (~40 s), en gardant l'objet centré ;
- surface de pose contrastée, pas un plan blanc uniforme ;
- lumière diffuse, éviter les reflets spéculaires qui se déplacent avec la caméra ;
- **ne pas zoomer** pendant la prise (les intrinsèques sont supposées fixes) ;
- deux hauteurs de vue si possible (à hauteur d'objet, puis en plongée à 45°) ;
- les objets transparents, très brillants ou noirs mats restent difficiles — c'est
  une limite de la stéréo photométrique, pas de ce pipeline.

---

## Développement

```powershell
# Open3D n'a pas de wheel Python 3.13 : le venv de test est en 3.12.
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv312\Scripts\python.exe -m pytest tests/        # 70 tests
```

En Python 3.13, les deux fichiers de test qui dépendent d'Open3D sont ignorés
automatiquement (59 tests restants). Les 18 tests d'interface le sont aussi si
PyQt6 est absent — ils tournent sans écran, via `QT_QPA_PLATFORM=offscreen`.

Le mode `MINIDS_FAKE_GPU=1` rejoue les 10 étapes, la progression et les
artefacts sans GPU ni modèle — il sert à valider tout le transport avant de
payer une minute de pod :

```powershell
$env:MINIDS_FAKE_GPU="1"; $env:MINIDS_TOKEN="demo"
.\.venv\Scripts\python.exe -m uvicorn server.app:app --port 8123
# dans un autre terminal
$env:MINIDS_URL="http://127.0.0.1:8123"; $env:MINIDS_TOKEN="demo"
python client/minids.py run video.mp4
```

### Ce qui est vérifié, et ce qui ne peut pas l'être ici

**70 tests passent sur cette machine.** Les plus utiles reposent sur une vérité
terrain synthétique : un objet connu (sphère + boîte), des caméras en orbite, des
profondeurs rendues exactement — ce qui permet de mesurer l'erreur du *code*,
sans modèle. Sont couverts ainsi la fusion TSDF (Chamfer < 1 % de la diagonale),
Poisson, le nettoyage (un îlot parasite ajouté doit disparaître), le bake de
texture (l'objet est peint par une fonction `f(position)` connue, et l'on vérifie
que la texture obtenue redonne bien `f` au bon endroit), et le re-maillage local
depuis `vggt_raw.npz` jusqu'au GLB à l'échelle métrique.

Le reste : géométrie caméra, conformité du GLB (alignement des chunks, conversion
d'axes, PNG relu octet à octet, validation par un parseur glTF tiers), sélection
d'images par netteté sur une vraie vidéo ffmpeg, et le transport complet contre
un vrai serveur uvicorn — reprise d'upload, reprise de téléchargement, `.part`
périmé, sha256 corrompu, annulation.

L'interface est testée de la même façon : la vraie fenêtre est construite en
`offscreen`, remplie, et lance un scan complet contre un serveur uvicorn en mode
factice. C'est la chaîne signaux/threads qui est vérifiée — l'endroit où une
interface Qt casse réellement — ainsi que la reprise d'un job soumis en dehors de
la fenêtre.

> Ces tests ont déjà payé : ils ont révélé que `create_rays_pinhole` d'Open3D
> renvoie des directions **non normalisées** (composante z = 1), donc que `t_hit`
> est une profondeur *z* et non une distance euclidienne. Le test de visibilité du
> bake comparait les deux — soit ~20 % d'erreur en périphérie d'image, et des trous
> dans chaque texture. Corrigé.

En revanche, **les étapes GPU n'ont pas pu être exécutées ici** : VGGT-Ω, SAM 3 et
gsplat exigent CUDA et des poids sous accès restreint. Les points à surveiller au
premier run réel :

- `server/pipeline/vggt.py` — les formes de tenseurs renvoyées par le modèle sont
  normalisées par `_as_sequence`, qui lève une erreur explicite avec la forme
  observée si l'API du dépôt a bougé ;
- `server/pipeline/segment.py` — l'appel SAM 3 retombe automatiquement sur la
  segmentation géométrique en cas d'écart d'API, sans faire échouer le job ;
- `server/pipeline/refine_2dgs.py` — `_unpack_render` tolère les variantes de
  signature de `rasterization_2dgs`.

Aucun de ces trois points ne fait perdre un scan : le repli est prévu, et
`vggt_raw.npz` est écrit **avant** les étapes lourdes.

---

## Licences

Attention avant tout usage commercial : VGGT-Ω et SAM 3 ont leurs propres
licences Meta (non permissives par défaut), gsplat est en Apache-2.0, Open3D en
MIT. Le code de ce dépôt ne modifie pas ces conditions.
