# Tools

Inventário das ferramentas utilitárias em `src/tools`.

Objetivo deste documento:

- centralizar o propósito de cada script
- reduzir descoberta por inspeção manual de código
- deixar claro quando uma ferramenta é exploratória, operacional ou de exportação

Regra prática:

- se o script altera corpus ou gera artefatos permanentes, rode sempre com caminhos explícitos
- quando houver dúvida sobre formato de entrada, assuma JSONL e confirme no `--help`

## Visão Geral

| Script | Categoria | Entrada típica | Reexecução | Uso principal |
| --- | --- | --- | --- | --- |
| `src/tools/build_annotation_editor.py` | anotação | JSON, JSONL | sobrescreve saída | gerar um editor HTML para revisão manual de spans |
| `src/tools/build_calibration_dataset.py` | calibração | JSON, JSONL | sobrescreve saída | montar dataset de calibração a partir de previsões do modelo |
| `src/tools/clean_generic_spans.py` | limpeza | JSON, JSONL | cuidado com `--inplace` | remover spans genéricos por banlist |
| `src/tools/convert_sanity_jsonl_to_bio_csv.py` | conversão | JSONL | sobrescreve saída | converter JSONL de sanidade para CSV BIO |
| `src/tools/count_dataset_entities.py` | inspeção | JSON, JSONL | seguro | contar spans e distribuição de labels em um corpus |
| `src/tools/evaluate_chunk_quality.py` | auditoria | artefatos locais | sobrescreve saída | avaliar qualidade de um chunk a partir dos artefatos do ciclo |
| `src/tools/export_dissertation_tables.py` | exportação | artefatos locais | sobrescreve saída | wrapper para exportar tabelas de dissertação |
| `src/tools/export_thesis_tables.py` | exportação | artefatos locais | sobrescreve saída | consolidar artefatos em CSV/Markdown para escrita |
| `src/tools/inspect_dense_tips.py` | auditoria | JSON, JSONL | sobrescreve saída | filtrar e visualizar tips com muitas entidades |
| `src/tools/list_distinct_labels.py` | inspeção | JSON, JSONL | seguro | listar labels distintas encontradas em um corpus |
| `src/tools/profile_pseudolabelling_inference.py` | profiling | JSONL | sobrescreve saída opcional | medir custo de inferência do pipeline de pseudolabel |
| `src/tools/render_ner_html.py` | visualização | JSON, JSONL | sobrescreve saída | renderizar corpus anotado em HTML |
| `src/tools/replace_label_in_jsonl.py` | edição | JSON, JSONL | cuidado com `--inplace` | renomear labels em um corpus JSON/JSONL |
| `src/tools/run_remaining_chunk_probes.py` | operação | chunks JSONL | parcialmente idempotente | rodar probes restantes de chunks 50k com configuração fixa |
| `src/tools/sample_large_corpus.py` | amostragem | JSON, JSONL | sobrescreve saída | gerar amostras reproduzíveis de corpus grande |
| `src/tools/split_dataset_for_calibration.py` | calibração | JSON array | sobrescreve saída | separar train/calibration com controle de perfil de labels |
| `src/tools/split_large_corpus_into_chunks.py` | particionamento | JSON, JSONL | sobrescreve saída | dividir corpus grande em chunks fixos |
| `src/tools/summarize_context_boost_audit.py` | auditoria | JSONL | sobrescreve saída | resumir artefatos de auditoria do context boost |

## Convenções Rápidas

### Entrada típica

- `JSON, JSONL`: o script aceita ambos
- `JSONL`: espera linhas JSON independentes
- `JSON array`: espera um arquivo JSON contendo lista na raiz
- `artefatos locais`: opera sobre resultados já gerados em `artifacts/`

### Reexecução

- `seguro`: leitura ou relatório; não altera insumos
- `sobrescreve saída`: pode ser rerodado se o caminho de saída estiver correto
- `cuidado com --inplace`: pode modificar o arquivo de entrada
- `parcialmente idempotente`: tenta pular trabalho já concluído, mas ainda gera novos artefatos auxiliares

## Anotação E Visualização

### `src/tools/build_annotation_editor.py`

Gera um HTML interativo para revisar e editar spans de NER.

Use quando:

- você precisa inspecionar anotações manualmente
- quer distribuir uma visão navegável do corpus para revisão humana

Entradas principais:

- `--input`
- `--output`
- `--title`

Saída:

- arquivo HTML editável no navegador

### `src/tools/render_ner_html.py`

Renderiza um corpus anotado em HTML para visualização, sem foco em edição.

Use quando:

- a necessidade é leitura e auditoria visual
- você quer compartilhar exemplos anotados rapidamente

Entradas principais:

- `--input`
- `--output`
- `--title`

Saída:

- relatório HTML estático

## Calibração

### `src/tools/build_calibration_dataset.py`

Executa inferência do modelo e produz dados para calibrar scores.

Use quando:

- você quer ajustar ou reconstruir o calibrador de confiança
- precisa de previsões alinhadas com o loader atual de inferência

Pontos relevantes:

- usa `src/gliner_loader.py`
- aceita `--map-location`
- emite progresso durante execução

Entradas principais:

- `--model-path`
- `--input`
- `--output-csv` ou artefato equivalente definido no script

### `src/tools/split_dataset_for_calibration.py`

Separa um dataset em subconjuntos para calibração, preservando perfil de labels.

Use quando:

- você precisa montar split específico para calibrador
- quer separar subconjuntos sem depender do split principal de treino

Entradas principais:

- dataset em JSON array
- parâmetros de seed, proporção e campo de label

Saídas:

- arquivos JSON com subconjuntos separados

## Limpeza E Edição De Corpus

### `src/tools/clean_generic_spans.py`

Remove spans genéricos com base em uma banlist por label.

Use quando:

- o corpus contém spans pouco informativos como `local`, `casa`, `morador`
- você quer um passo simples de higienização antes de treino ou auditoria

Entradas principais:

- `--input`
- `--output`
- opções de banlist

### `src/tools/replace_label_in_jsonl.py`

Renomeia labels em arquivos JSON ou JSONL.

Use quando:

- houve mudança de nomenclatura de label
- você precisa uniformizar datasets antigos e novos

Entradas principais:

- `--input`
- `--output`
- label origem e label destino

### `src/tools/convert_sanity_jsonl_to_bio_csv.py`

Converte um JSONL simples de sanidade com spans em CSV BIO.

Use quando:

- você quer validar alinhamento token/spans
- precisa exportar um conjunto pequeno para inspeção em formato BIO

Dependência relevante:

- `nltk`

## Inspeção E Profiling

### `src/tools/count_dataset_entities.py`

Conta spans e distribuição de labels em um corpus.

Use quando:

- você precisa de estatísticas rápidas de volume
- quer comparar corpora antes de treino ou limpeza

### `src/tools/evaluate_chunk_quality.py`

Resume um ou mais runs de chunk usando os artefatos já produzidos pelo ciclo.

Use quando:

- você quer investigar por que um chunk foi bom ou ruim
- precisa comparar `kept_count`, deltas, boosts e redundância de textos
- quer gerar um CSV consolidado por chunk

Métricas incluídas:

- `kept_count` e `kept_rate`
- delta micro e macro
- delta por label
- `boosted_records` e `boosted_entities_total`
- média de entidades por relato kept
- média de entidades fortes e fracas por relato
- taxa de texto duplicado nos kepts
- flags simples como `high_kept_count`, `duplicate_texts`, `no_context_boost`

### `src/tools/inspect_dense_tips.py`

Seleciona tips com muitas entidades e exporta uma visão legível para inspeção.

Use quando:

- você quer investigar outliers com densidade alta de spans
- precisa abrir rapidamente os tips kept mais carregados
- quer exportar um subconjunto para HTML e revisão manual

Saídas possíveis:

- JSONL filtrado
- HTML para leitura
- summary JSON com contagens agregadas

### `src/tools/list_distinct_labels.py`

Lista labels distintas encontradas em `entities`, `ner` ou `spans`.

Use quando:

- você suspeita de labels fora do conjunto esperado
- quer validar consistência entre datasets

### `src/tools/profile_pseudolabelling_inference.py`

Faz profiling de inferência do pipeline de pseudolabeling.

Use quando:

- quer medir throughput
- precisa comparar CPU vs CUDA
- está ajustando `batch-size`, `max-tokens` ou `model-max-length`

Pontos relevantes:

- usa o loader compartilhado
- aceita `--map-location`
- ideal para benchmark rápido antes de um run grande

## Amostragem E Particionamento

### `src/tools/sample_large_corpus.py`

Gera amostras reproduzíveis de um corpus grande.

Use quando:

- você quer criar probes como `10k`
- precisa repetir um experimento com a mesma seed

### `src/tools/split_large_corpus_into_chunks.py`

Divide um corpus grande em chunks JSONL de tamanho fixo.

Use quando:

- você quer rodar pseudolabeling iterativo por partes
- precisa controlar custo por lote

Entradas principais:

- `--input`
- `--output-dir`
- `--chunk-size`
- `--chunk-prefix`

Saída opcional:

- `--summary-json`

## Exportação De Resultados

### `src/tools/export_thesis_tables.py`

Consolida artefatos de treino base, avaliação e pseudolabeling em tabelas CSV/Markdown.

Use quando:

- você precisa atualizar tabelas de dissertação
- quer um snapshot consistente dos artefatos experimentais

Saídas típicas:

- `results_master.csv`
- `tables/table_baselines.csv`
- `tables/table_pseudolabel_probes.csv`
- `tables/table_runtime.csv`

### `src/tools/export_dissertation_tables.py`

Wrapper fino para `src/tools/export_thesis_tables.py`.

Use quando:

- você quer um entrypoint semanticamente alinhado com “dissertation”

## Pseudolabeling Operacional

### `src/tools/summarize_context_boost_audit.py`

Resume o `03_context_boost_details.jsonl` em:

- summary JSON
- CSV com uma linha por entidade boostada

Use quando:

- você precisa auditar o `context boost`
- quer comparar volume e perfil de boosts entre runs

Entradas principais:

- `--details-jsonl`
- `--summary-json`
- `--rows-csv`
- `--top-n`

### `src/tools/run_remaining_chunk_probes.py`

Automatiza execução de probes em chunks 50k restantes para uma configuração fixa.

Use quando:

- você já definiu threshold e versão
- quer deixar vários chunks rodando sem supervisionar manualmente
- precisa pular automaticamente chunks já concluídos

Comportamento atual:

- por padrão roda do `chunk 03` ao `chunk 08`
- usa `t037`
- cria auditoria por chunk
- escreve um CSV incremental com status e deltas

Entradas principais:

- `--chunks-dir`
- `--start-chunk`
- `--end-chunk`
- `--threshold`
- `--version`
- `--summary-csv`

## Convenções Recomendadas

### Nome de artefatos

Para runs operacionais grandes, use nomes previsíveis:

- `multi_with_negatives_chunk02_50k_t037_cuda_v12`
- `context_boost_audit_v12_chunk02`

Isso evita ambiguidade entre:

- threshold
- chunk
- versão do experimento

### Antes de rodar scripts pesados

Confirme:

- caminho do corpus
- caminho do modelo base
- caminho do calibrador
- `map_location`
- diretório de saída

### Depois de rodar scripts pesados

Guarde pelo menos:

- artefato principal do run
- log
- summary JSON
- comparação base vs refit

## Próximos Ajustes Possíveis

Este documento pode evoluir com:

- exemplos mínimos por script
- tabela de dependências externas
- coluna indicando se o script lê JSON, JSONL ou ambos
- coluna indicando se o script é seguro para reexecução idempotente

## Comandos Exemplos

Os exemplos abaixo priorizam os scripts mais operacionais do repositório.

### `build_calibration_dataset.py`

```bash
cd src
python3 tools/build_calibration_dataset.py \
  --model-path ./artifacts/base_model_training/experiments/multi_lr_wd_grid_bs16_with_negatives/best_overall_gliner_model \
  --input ../data/dd_corpus_small_test_final.json \
  --output-csv ./artifacts/calibration/multi_with_negatives/calibration_dataset.csv \
  --output-predictions-jsonl ./artifacts/calibration/multi_with_negatives/calibration_predictions.jsonl \
  --labels Person,Location,Organization \
  --batch-size 8 \
  --max-tokens 512 \
  --threshold 0.0 \
  --map-location cuda
```

### `profile_pseudolabelling_inference.py`

```bash
cd src
python3 tools/profile_pseudolabelling_inference.py \
  --model-path ./artifacts/base_model_training/experiments/multi_lr_wd_grid_bs16_with_negatives/best_overall_gliner_model \
  --input-jsonl ../data/dd_corpus_large_sample_10k.jsonl \
  --labels Person,Location,Organization \
  --text-fields assunto,relato,bairroLocal,logradouroLocal,cidadeLocal,pontodeReferenciaLocal \
  --batch-size 16 \
  --max-tokens 512 \
  --score-threshold 0.0 \
  --limit 100 \
  --map-location cuda
```

### `split_large_corpus_into_chunks.py`

```bash
cd src
python3 tools/split_large_corpus_into_chunks.py \
  --input ../data/dd_corpus_large.json \
  --output-dir ../data/dd_corpus_large_chunks_50k \
  --chunk-size 50000 \
  --chunk-prefix dd_corpus_large_chunk \
  --summary-json ../data/dd_corpus_large_chunks_50k_summary.json
```

### `summarize_context_boost_audit.py`

```bash
cd src
python3 tools/summarize_context_boost_audit.py \
  --details-jsonl ./artifacts/pseudolabelling/multi_with_negatives_chunk02_50k_t037_cuda_v12/03_context_boost_details.jsonl \
  --summary-json ./artifacts/pseudolabelling/context_boost_audit_v12_chunk02/summary.json \
  --rows-csv ./artifacts/pseudolabelling/context_boost_audit_v12_chunk02/boosted_entities.csv \
  --top-n 30
```

### `run_remaining_chunk_probes.py`

```bash
cd src
python3 tools/run_remaining_chunk_probes.py \
  --chunks-dir ../data/dd_corpus_large_chunks_50k \
  --start-chunk 3 \
  --end-chunk 8 \
  --threshold 0.37 \
  --version v13 \
  --run-root ./artifacts/pseudolabelling \
  --summary-csv ./artifacts/pseudolabelling/chunk_probe_status_t037_v13.csv
```

### `evaluate_chunk_quality.py`

```bash
cd src
python3 tools/evaluate_chunk_quality.py \
  --run-glob './artifacts/pseudolabelling/multi_with_negatives_chunk*_50k_t037_cuda_v*' \
  --output-csv ./artifacts/pseudolabelling/chunk_quality_t037.csv \
  --output-json ./artifacts/pseudolabelling/chunk_quality_t037.json
```

### `inspect_dense_tips.py`

```bash
cd src
python3 tools/inspect_dense_tips.py \
  --input ./artifacts/pseudolabelling/multi_with_negatives_chunk04_50k_t037_cuda_v13/05_split/kept.jsonl \
  --min-entities 30 \
  --output-jsonl ./artifacts/pseudolabelling/chunk04_dense_tips.jsonl \
  --output-html ./artifacts/pseudolabelling/chunk04_dense_tips.html \
  --summary-json ./artifacts/pseudolabelling/chunk04_dense_tips_summary.json \
  --title "Chunk 04 Dense Tips"
```
