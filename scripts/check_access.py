"""Vérifie les accès Hugging Face et affiche la valeur exacte de MINIDS_CKPT.

À lancer *avant* de louer un pod : une erreur de token ou un accès non approuvé
ne se verrait sinon qu'après le démarrage du conteneur, GPU facturé.

Stdlib uniquement — aucune installation requise.

    python scripts/check_access.py            # lit $env:HF_TOKEN
    python scripts/check_access.py --token hf_xxx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api"

# Un seul dépôt héberge les deux variantes ; les autres noms sont des replis
# au cas où la publication changerait.
VGGT_CANDIDATES = [
    "facebook/VGGT-Omega",
    "facebook/VGGT-Omega-1B-512",
    "facebook/vggt-omega",
]
WEIGHT_SUFFIXES = (".pt", ".safetensors", ".bin")
# Le pipeline tourne en 512 par défaut : c'est cette variante qu'il faut choisir.
# Sans préférence explicite, on prendrait la 256-text, qui dégraderait la
# profondeur sans le dire.
PREFERRED_RESOLUTION = "512"


def request(path: str, token: str | None) -> tuple[int, dict | None]:
    headers = {"User-Agent": "minids-check"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{API}{path}", headers=headers)  # noqa: S310 - API HTTPS constante
    try:
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310 - requête HTTPS construite ci-dessus
            try:
                payload = json.loads(response.read())
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                print(f"réponse JSON invalide : {exc}", file=sys.stderr)
                return response.status, None
            if not isinstance(payload, dict):
                print("réponse JSON inattendue", file=sys.stderr)
                return response.status, None
            return response.status, payload
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"réseau indisponible : {exc}", file=sys.stderr)
        return 0, None


def check_identity(token: str) -> bool:
    status, payload = request("/whoami-v2", token)
    if status != 200 or payload is None:
        print(f"✗ token refusé par Hugging Face (HTTP {status})")
        print("  → https://huggingface.co/settings/tokens — créer un token de type 'Read'")
        return False
    auth = payload.get("auth") if isinstance(payload.get("auth"), dict) else {}
    access = auth.get("accessToken") if isinstance(auth.get("accessToken"), dict) else {}
    scopes = access.get("role", "?")
    print(f"✓ token valide — utilisateur « {payload.get('name')} », rôle « {scopes} »")
    return True


def check_repo(repo: str, token: str) -> list[str] | None:
    """Retourne la liste des fichiers de poids, ou None si l'accès est refusé."""
    repo = repo.strip()
    if not repo:
        print("? dépôt vide ignoré")
        return None
    encoded_repo = urllib.parse.quote(repo, safe="/")
    status, payload = request(f"/models/{encoded_repo}", token)
    if status == 200 and payload is not None:
        siblings = payload.get("siblings") if isinstance(payload.get("siblings"), list) else []
        files = [
            item["rfilename"] for item in siblings if isinstance(item, dict) and isinstance(item.get("rfilename"), str)
        ]
        weights = sorted(file for file in files if file.lower().endswith(WEIGHT_SUFFIXES))
        gated = payload.get("gated")
        print(f"✓ {repo} — accès OK{' (dépôt restreint, approuvé)' if gated else ''}")
        for name in weights:
            print(f"    poids : {name}")
        if not weights:
            print("    (aucun fichier de poids listé — vérifier manuellement l'onglet Files)")
        return weights
    if status in (401, 403):
        print(f"✗ {repo} — accès refusé (HTTP {status}) : demande non approuvée, ou token sans droit sur ce dépôt")
    elif status == 404:
        print(f"· {repo} — inexistant")
    else:
        print(f"? {repo} — HTTP {status}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--repo", action="append", default=[], help="dépôt supplémentaire à tester")
    args = parser.parse_args()

    args.token = args.token.strip()
    if not args.token:
        print("HF_TOKEN absent. Le définir, ou passer --token hf_xxx", file=sys.stderr)
        return 2
    if not check_identity(args.token):
        return 1

    print("\n— VGGT-Ω —")
    resolved: tuple[str, list[str]] | None = None
    repositories = list(dict.fromkeys(VGGT_CANDIDATES + [repo.strip() for repo in args.repo if repo.strip()]))
    for repo in repositories:
        weights = check_repo(repo, args.token)
        if weights is not None and resolved is None:
            resolved = (repo, weights)

    print("\n— SAM 3 (segmentation par prompt texte, optionnel) —")
    sam3_ok = check_repo("facebook/sam3", args.token) is not None
    if not sam3_ok:
        print("  → sans SAM 3, le pipeline bascule sur la segmentation géométrique (aucun blocage)")
        print("  → pour l'activer : https://huggingface.co/facebook/sam3 puis « Request access »")

    print("\n" + "=" * 62)
    if resolved is None:
        print("Aucun dépôt VGGT-Ω accessible.")
        print("Ouvrir le lien « request access to the checkpoints » du README de")
        print("https://github.com/facebookresearch/vggt-omega, puis relancer avec :")
        print("    python scripts/check_access.py --repo <identifiant/exact>")
        return 1

    repo, weights = resolved
    chosen = pick_checkpoint(weights)
    value = f"{repo}:{chosen}" if chosen else repo
    print("À reporter dans les variables d'environnement du template RunPod :")
    print(f"    MINIDS_CKPT = {value}")
    print(f"    HF_TOKEN    = {_mask_token(args.token)}")
    if chosen and PREFERRED_RESOLUTION not in chosen:
        print()
        print(f"  ⚠ aucun poids en {PREFERRED_RESOLUTION} px trouvé : ajouter")
        print(f"    MINIDS_IMAGE_RESOLUTION = 256   (doit correspondre à « {chosen} »)")
    print("=" * 62)
    return 0


def pick_checkpoint(weights: list[str]) -> str | None:
    """Choisit la variante 512 quand elle existe, sinon le premier poids."""
    if not weights:
        return None
    ordered = sorted(str(name) for name in weights)
    preferred = [name for name in ordered if PREFERRED_RESOLUTION in name]
    return preferred[0] if preferred else ordered[0]


def _mask_token(token: str) -> str:
    """Affiche juste assez du jeton pour l'identifier, jamais sa valeur complète."""
    if len(token) <= 7:
        return f"{token[:2]}…"
    return f"{token[:3]}…{token[-4:]}"


if __name__ == "__main__":
    raise SystemExit(main())
