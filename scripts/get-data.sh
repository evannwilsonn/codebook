#!/usr/bin/env sh
# Downloads the Large Movie Review Dataset (IMDB, Maas et al. 2011) into data/aclImdb/ (not committed).
set -e
cd "$(dirname "$0")/.."
mkdir -p data
curl -fsSL -o data/imdb.tgz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzf data/imdb.tgz -C data
rm data/imdb.tgz
echo "Saved data/aclImdb/ ($(ls data/aclImdb/test/pos | wc -l) positive test reviews)"
