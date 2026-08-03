# Copier en `minids.env.ps1` (ignoré par git), remplir, puis charger avec :
#     . .\minids.env.ps1
#
# Ne jamais committer le fichier rempli : il contient deux secrets.

# --- Hugging Face : uniquement pour construire/lancer le pod ---
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Valeur exacte donnée par `python scripts/check_access.py`
$env:MINIDS_CKPT = "facebook/VGGT-Omega:vggt_omega_1b_512.pt"

# --- Client : ce qui sert au quotidien ---
# URL du proxy RunPod, visible dans la fiche du pod (bouton Connect)
$env:MINIDS_URL = "https://XXXXXXXXXXXX-8000.proxy.runpod.net"

# Doit être identique à la variable MINIDS_TOKEN du template RunPod.
# En générer un avec :
#     python -c "import secrets; print(secrets.token_urlsafe(24))"
$env:MINIDS_TOKEN = "a-remplacer"

Write-Host "miniDS : $env:MINIDS_URL" -ForegroundColor Green
