# Downloads the Large Movie Review Dataset (IMDB, Maas et al. 2011) into data/aclImdb/ (not committed).
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$root/data" | Out-Null
Invoke-WebRequest -Uri "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" -OutFile "$root/data/imdb.tgz"
tar xzf "$root/data/imdb.tgz" -C "$root/data"
Remove-Item "$root/data/imdb.tgz"
Write-Host "Saved data/aclImdb/"
