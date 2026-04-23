# FAQ

Perguntas e respostas operacionais recorrentes do projeto.

## Como usar `rsync` para trazer arquivos da `workstation02` para o repositório local mantendo a estrutura de pastas?

Se a pasta local de destino ainda não existir:

```bash
mkdir -p ~/ailab/nerdd/artifacts/benchmarks/metadata_location_literal_top20_v1
```

Para copiar arquivos específicos diretamente para a pasta correspondente no repositório local:

```bash
rsync -av \
  workstation02:~/ailab/nerdd/artifacts/benchmarks/metadata_location_literal_top20_v1/approved_source_ids.txt \
  workstation02:~/ailab/nerdd/artifacts/benchmarks/metadata_location_literal_top20_v1/top100_review_index.tsv \
  ~/ailab/nerdd/artifacts/benchmarks/metadata_location_literal_top20_v1/
```

Para preservar a estrutura relativa a partir da raiz `~/ailab/nerdd/`, use `--relative`:

```bash
cd ~/ailab/nerdd

rsync -av --relative \
  workstation02:~/ailab/nerdd/./artifacts/benchmarks/metadata_location_literal_top20_v1/approved_source_ids.txt \
  workstation02:~/ailab/nerdd/./artifacts/benchmarks/metadata_location_literal_top20_v1/top100_review_index.tsv \
  .
```
