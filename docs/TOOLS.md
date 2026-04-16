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
| `src/tools/audit_calibration_by_label.py` | auditoria | CSV de calibração | sobrescreve saída | auditar scores brutos vs calibrados por label e por validade |
| `src/tools/audit_refit_regressions.py` | auditoria | gold + predictions.jsonl | sobrescreve saída | auditar regressões entre baseline e refit, com wins/losses/ties e confusões de label |
| `src/tools/build_calibration_dataset.py` | calibração | JSON, JSONL | sobrescreve saída | montar dataset de calibração a partir de previsões do modelo |
| `src/tools/build_train_annotation_prompt_probe.py` | auditoria | audits + lote fonte | sobrescreve saída | montar um probe pequeno e diagnóstico para testar prompts de adjudicação voltados a treino |
| `src/tools/manage_codex_adjudication_benchmark.py` | operação | JSONL de adjudicação | resumível | gerenciar benchmark chunkado de adjudicação assistida por Codex |
| `src/tools/run_llm_adjudication.py` | operação | JSONL de adjudicação | resumível | chamar a Responses API para adjudicação literal ou `train_annotation`, inclusive em chunks |
| `src/tools/expand_location_spans_with_markers.py` | limpeza | JSON, JSONL | sobrescreve saída | expandir spans de `Location` para incluir marcadores locativos como `rua`, `trav`, `trv`, `av` quando estiverem imediatamente antes |
| `src/tools/clean_generic_spans.py` | limpeza | JSON, JSONL | cuidado com `--inplace` | remover spans genéricos por banlist |
| `src/tools/build_refit_pseudolabel_dataset.py` | conversão | JSONL de adjudicação | sobrescreve saída | projetar `06_llm_adjudicated` para um `pseudolabel_path` compatível com refit |
| `src/tools/compare_spacy_predictions.py` | auditoria | JSON, JSONL | sobrescreve saída | comparar previsões existentes contra spaCy no mesmo conjunto |
| `src/tools/compare_gliner_predictions.py` | auditoria | JSON, JSONL | sobrescreve saída | comparar previsões existentes contra outro modelo GLiNER no mesmo conjunto |
| `src/tools/convert_sanity_jsonl_to_bio_csv.py` | conversão | JSONL | sobrescreve saída | converter JSONL de sanidade para CSV BIO |
| `src/tools/compare_tokenizers.py` | auditoria | JSON, JSONL | sobrescreve saída | comparar tokenização fast vs slow em textos selecionados |
| `src/tools/count_dataset_entities.py` | inspeção | JSON, JSONL | seguro | contar spans e distribuição de labels em um corpus |
| `src/tools/evaluate_chunk_quality.py` | auditoria | artefatos locais | sobrescreve saída | avaliar qualidade de um chunk a partir dos artefatos do ciclo |
| `src/tools/export_dissertation_tables.py` | exportação | artefatos locais | sobrescreve saída | wrapper para exportar tabelas de dissertação |
| `src/tools/export_thesis_tables.py` | exportação | artefatos locais | sobrescreve saída | consolidar artefatos em CSV/Markdown para escrita |
| `src/tools/inspect_dense_tips.py` | auditoria | JSON, JSONL | sobrescreve saída | filtrar e visualizar tips com muitas entidades |
| `src/tools/prune_pseudolabel_tips.py` | limpeza | JSON, JSONL | sobrescreve saída | podar entidades de pseudolabel por score e densidade por tip |
| `src/tools/rank_pseudolabel_candidates.py` | auditoria | JSON, JSONL | sobrescreve saída | ranquear candidatos de pseudolabel para revisão manual |
| `src/tools/review_model_predictions.py` | auditoria | conjunto anotado | sobrescreve saída | gerar revisão HTML lado a lado de gold vs predição do modelo |
| `src/tools/review_adjudication_cases.py` | auditoria | JSON, JSONL | sobrescreve saída | gerar revisão HTML multicamada de baseline, GLiNER2, seeds e entidades finais adjudicadas |
| `src/tools/reshuffle_train_test_split.py` | split | JSON, JSONL | sobrescreve saída | recombinar train/test, opcionalmente remover duplicatas exatas entre inputs, e reemitir novos splits |
| `src/tools/list_distinct_labels.py` | inspeção | JSON, JSONL | seguro | listar labels distintas encontradas em um corpus |
| `src/tools/profile_pseudolabelling_inference.py` | profiling | JSONL | sobrescreve saída opcional | medir custo de inferência do pipeline de pseudolabel |
| `src/tools/render_ner_html.py` | visualização | JSON, JSONL | sobrescreve saída | renderizar corpus anotado em HTML |
| `src/tools/replace_label_in_jsonl.py` | edição | JSON, JSONL | cuidado com `--inplace` | renomear labels em um corpus JSON/JSONL |
| `src/tools/run_remaining_chunk_probes.py` | operação | chunks JSONL | parcialmente idempotente | rodar probes restantes de chunks 50k com configuração fixa |
| `src/tools/sample_large_corpus.py` | amostragem | JSON, JSONL | sobrescreve saída | gerar amostras reproduzíveis de corpus grande |
| `src/tools/select_train_annotation_cases.py` | seleção | JSONL de adjudicação | sobrescreve saída | selecionar candidatos mais treináveis para adjudicação LLM voltada a treino |
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
- `--span-field`
- `--score-fields`

Observações metodológicas:

- por padrão, o script usa o fallback `spans -> entities -> ner`
- com `--span-field`, você pode renderizar listas alternativas de entidades sem transformar o arquivo antes
- isso é útil para inspeção pré-adjudicação de artefatos como:
  - `review_seed_entities`
  - `baseline_entities`
  - `gliner2_entities`
  - `adjudication.entities_final`
- com `--score-fields`, você pode mostrar a confiança ao lado de cada entidade usando um ou mais campos de score em ordem de fallback

Saída:

- relatório HTML estático

### `src/tools/review_adjudication_cases.py`

Gera um HTML lado a lado para auditoria de casos de adjudicação, mostrando múltiplas camadas de entidades no mesmo relato.

Use quando:

- você quer inspecionar, no mesmo relatório, as entidades do baseline, do GLiNER2, as `review_seed_entities` e as entidades finais adjudicadas
- precisa comparar origem das seeds e scores por camada sem abrir vários HTMLs separados

Entradas principais:

- `--input`
- `--output`
- `--title`
- `--layers`
- `--score-fields`

Saída:

- relatório HTML estático multicamada por registro

## Calibração

### `src/tools/audit_calibration_by_label.py`

Resume um `calibration_predictions.csv` por label, separando positivos e negativos.

Use quando:

- você quer saber se erros continuam superconfiantes após calibração
- precisa comparar `Score` bruto com score calibrado por label
- quer medir rapidamente a fração de negativos com score alto, por exemplo em `Organization`

Entradas principais:

- `--calibration-csv`
- `--calibrator-path` opcional
- `--high-score-threshold`

Saída:

- JSON com contagens, médias e quantis para positivos e negativos, em modo bruto e calibrado

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

### `src/tools/build_calibration_dataset_gliner2.py`

Executa inferência do GLiNER2 e produz dados para calibrar scores.

Use quando:

- você quer ajustar ou reconstruir o calibrador de confiança para GLiNER2
- precisa comparar `GLiNER2 base` e `GLiNER2 + LoRA` com o mesmo formato de CSV do projeto atual

Pontos relevantes:

- usa `src/gliner2_loader.py`
- aceita `--adapter-dir`
- emite progresso durante execução

Entradas principais:

- `--model-path`
- `--adapter-dir` opcional
- `--input`
- `--output-csv`

### `src/tools/split_dataset_for_calibration.py`

Separa um dataset em subconjuntos para calibração, preservando perfil de labels.

Use quando:

- você precisa montar split específico para calibrador
- quer separar subconjuntos sem depender do split principal de treino
- quer evitar preservar ordem temporal dentro dos arquivos de saída

Entradas principais:

- dataset em JSON array
- parâmetros de seed, proporção e campo de label

Saídas:

- arquivos JSON com subconjuntos separados

Observação:

- use `--shuffle-output` para embaralhar a ordem final dentro de cada split

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

### `src/tools/build_refit_pseudolabel_dataset.py`

Converte a saída de `src/tools/run_llm_adjudication.py` em um JSONL pronto para `--pseudolabel-path` do refit.

Pontos relevantes:

- aceita `--top-n` para reaproveitar um lote já adjudicado maior e emitir só os primeiros `n` exemplos no formato consumido por `train_quick.py`
- mantém o filtro por decisão (`accept`, `accept_with_edits`) antes de emitir o dataset final

### `src/tools/run_llm_adjudication.py`

Executa adjudicação automática via Responses API sobre um JSONL de entrada.

Use quando:

- você quer evitar copiar e colar respostas manualmente no ChatGPT/Codex
- quer preencher um chunk inteiro de benchmark de forma programática
- precisa alternar entre protocolo literal e `train_annotation`

Pontos relevantes:

- aceita `--annotation-mode literal_review|train_annotation`
- valida offsets e labels após a resposta
- emite JSONL diretamente consumível pelo benchmark manager

Entradas principais:

- `--input`
- `--output-jsonl`
- `--model`
- `--annotation-mode`
- `--api-key-env`

### `src/tools/expand_location_spans_with_markers.py`

Expande spans de `Location` para incluir o marcador locativo anterior quando ele estiver explicitamente presente no texto.

Use quando:

- você quer reduzir inconsistência entre `Rua X` e `X`
- o corpus mistura spans completos de logradouro com spans sem o marcador
- você quer testar uma normalização mais consistente antes de treinar ou avaliar

Pontos relevantes:

- opera sobre `spans` em corpora JSON ou JSONL
- é conservador: só expande quando encontra um marcador locativo imediatamente antes do span
- suporta abreviações como `tr`, `trv`, `trav`, `av`
- suporta títulos intermediários como `Dr.` em casos como `Trav Dr . Lopes`

### `src/tools/select_train_annotation_cases.py`

Seleciona um lote de textos mais adequados para adjudicação LLM voltada a treino.

Use quando:

- você quer gerar um benchmark separado de `train_annotation`
- o benchmark literal de desacordo já se mostrou inadequado como pseudolabel de treino
- você quer priorizar casos mais estáveis, menos ruidosos e com seeds melhores

Critérios principais:

- favorece `agreement_ratio` moderado/alto
- favorece `baseline_coverage_proxy` mais forte
- favorece seeds com origem `agreed_exact` e `baseline_high_score`
- penaliza ruído alto, textos longos e seeds genéricos
- por padrão, pode ordenar por `adjudication_priority_score` quando esse campo já foi materializado a montante

Saídas:

- JSONL com o lote selecionado
- resumo opcional com distribuição de labels e origens de seeds

### `src/tools/score_adjudication_candidates.py`

Calcula um score de prioridade para adjudicação voltado a utilidade de treino, não apenas confiança do baseline.

Use quando:

- você quer priorizar casos com maior potencial de ganho após revisão LLM
- não quer depender só de `record_score` alto
- quer favorecer casos de domínio plausível com incerteza produtiva

Sinais principais:

- `domain_score`
- `disagreement_midband_score`
- `record_score_midband_score`
- `location_seed_score`
- `adjudicability_score`

Saídas:

- `adjudication_priority_score` em cada linha
- subscores e penalidades em `_adjudication_priority`
- resumo opcional com médias por componente

### `src/tools/audit_refit_regressions.py`

Audita regressões entre um baseline e um refit/candidato sobre o mesmo conjunto gold.

Use quando:

- uma comparação `supervised_only` vs `supervised_plus_pseudolabels` mudou as métricas e você precisa entender o mecanismo do ganho/perda
- você quer medir `wins`, `losses`, `ties`
- você quer diagnosticar `spurious_entity`, `wrong_label`, `boundary_or_partial` e `missing_entity`

Saídas:

- `summary.json`
- `regressions.jsonl`
- `wins.jsonl`
- `ties.jsonl`
- `top_regressions.md`

Diagnósticos adicionais:

- `loss_reason_counts_by_label`
- `wrong_label_confusions`

### `src/tools/build_train_annotation_prompt_probe.py`

Monta um probe pequeno e diagnóstico para testar prompts de adjudicação voltados a treino em ChatGPT/Codex antes de abrir um benchmark novo.

Use quando:

- você quer testar rapidamente um prompt novo com `5-10` casos
- precisa cobrir falhas observadas nos audits, como:
  - `Location -> Person`
  - `Location -> Organization`
  - `boundary_or_partial`
  - `spurious_entity`

Entradas:

- um `regressions.jsonl` de auditoria
- um `wins.jsonl` de auditoria
- o lote fonte original de adjudicação para treino

Saídas:

- JSONL limpo com:
  - `source_id`
  - `text`
  - `review_seed_entities`
  - `_probe_meta`

### `src/tools/manage_codex_adjudication_benchmark.py`

Gerencia um benchmark chunkado para comparar adjudicação do GPT com adjudicação assistida por Codex.

Use quando:

- você quer congelar um benchmark input único
- precisa trabalhar em chunks pequenos e resumíveis
- quer validar incrementalmente as respostas antes de consolidar a saída final

Subcomandos principais:

- `init`
- `status`
- `next`
- `ingest`
- `build-output`

Wrapper operacional recomendado:

- `scripts/codex_benchmark.sh`
- o wrapper relembra a restricao central do benchmark: `accept` e `accept_with_edits` so podem manter entidades ja presentes em `review_seed_entities`

Exemplos:

```bash
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 next
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 open-next
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 auto-next
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 show chunk_001
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 show-latest
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 response-path chunk_001
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 ingest chunk_001
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 ingest-latest
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 auto-complete-next
scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 status
```

Use quando:

- você já tem `06_llm_adjudicated`
- quer treinar com `small_train + pseudolabels` sem materializar um dataset combinado
- precisa filtrar apenas decisões aprovadas pelo LLM

Saídas:

- JSONL com:
  - `text`
  - `entities`
- summary JSON opcional com contagens por decisão e por label

Entradas principais:

- `--input`
- `--output-jsonl`
- `--summary-json` opcional
- `--allowed-decisions` com default `accept,accept_with_edits`

### `src/tools/manage_codex_adjudication_benchmark.py`

Gerencia um benchmark chunkado para comparar a adjudicação do `gpt-5` com uma adjudicação assistida por Codex sobre os mesmos casos.

Use quando:

- você quer congelar um subconjunto de casos do `05_llm_input`
- precisa emitir chunks pequenos para adjudicação incremental
- quer retomar o benchmark sem reprocessar chunks já concluídos
- precisa validar e consolidar respostas estruturadas em um único JSONL final

Subcomandos:

- `init`
  - cria `state.json`, chunks e um benchmark input congelado
- `status`
  - mostra progresso por status de chunk
- `next`
  - marca e imprime o próximo chunk pendente
- `ingest`
  - valida respostas de um chunk e salva o resultado consolidado daquele bloco
- `build-output`
  - junta todos os chunks concluídos em um output final JSONL

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

Seleciona tips com muitas entidades e exporta uma visão legível para auditoria de densidade.

Use quando:

- você quer investigar outliers com densidade alta de spans
- precisa abrir rapidamente os tips kept mais carregados
- quer exportar um subconjunto para HTML e revisão manual

Não use como etapa principal de `04_ranked_candidates`.

Para ranking operacional de candidatos de pseudolabel, use `src/tools/rank_pseudolabel_candidates.py`.

Saídas possíveis:

- JSONL filtrado
- HTML para leitura
- summary JSON com contagens agregadas

### `src/tools/prune_pseudolabel_tips.py`

Limpa um conjunto de pseudolabels já kept, removendo entidades fracas e limitando densidade por tip.

Use quando:

- você quer testar se o problema está dentro dos tips kept, e não apenas no `kept_count`
- precisa gerar um `kept.jsonl` mais limpo para refit experimental
- quer podar entidades por score antes de mexer no threshold de split

Controles principais:

- `--min-entity-score`
- `--max-entities-per-tip`
- `--drop-tips-over-max`
- `--drop-empty-tips`
- `--allowed-labels`

Saídas possíveis:

- JSONL limpo
- HTML opcional para revisão manual
- summary JSON com contagens do que foi removido

### `src/tools/review_model_predictions.py`

Roda um modelo sobre um conjunto anotado e gera material de revisão qualitativa.

Use quando:

- você quer inspecionar diretamente se o baseline parece pior do que o F1 sugere
- precisa abrir os piores casos primeiro, com gold e predição lado a lado
- quer um `comparison.jsonl` para auditoria manual mais detalhada

Saídas:

- `comparison.jsonl`
- `metrics.json`
- `summary.json`
- `review.html`

### `src/tools/review_gliner2_predictions.py`

Executa um modelo GLiNER2 base ou GLiNER2 + LoRA em um dataset anotado e gera revisão lado a lado.

Aceita `--model-path` como repo id do Hugging Face ou caminho local real.

Use quando:

- quer comparar GLiNER2 com o baseline atual usando o mesmo holdout anotado
- precisa de `metrics.json`, `summary.json` e `review.html` no mesmo estilo do pipeline atual
- está validando se GLiNER2 base ou LoRA vale uma migração

### `src/gliner2_training/train_quick.py`

Treina rapidamente um modelo GLiNER2 em split único e avalia no holdout anotado.

Use quando:

- quer um análogo do `base_model_training.train_quick` para GLiNER2
- precisa testar rápido `GLiNER2 base`, `LoRA` ou hiperparâmetros antes de um experimento maior
- quer gerar `quick_summary.json` e `eval_test/metrics.json` para comparação com o stack atual

Observações metodológicas:

- agora aceita `--pseudolabel-path` e `--train-mode`
- em `supervised_plus_pseudolabels`, os pseudolabels são apensados apenas ao split de treino
- o split de validação continua supervisionado-only
- não é necessário materializar um dataset combinado `small_train + pseudolabels`

### `src/base_model_training/train_quick.py`

Treina rapidamente um modelo GLiNER em split único e avalia no holdout anotado.

Use quando:

- quer uma probe rápida no stack base antes de nested CV maior
- precisa comparar `supervised_only` contra `supervised_plus_pseudolabels`
- quer consumir adjudicações convertidas em `--pseudolabel-path` sem materializar um dataset combinado

Observações metodológicas:

- agora aceita `--pseudolabel-path` e `--train-mode`
- em `supervised_plus_pseudolabels`, os pseudolabels entram apenas no split de treino
- o split de validação permanece supervisionado-only
- a deduplicação por `text`, quando habilitada, preserva a linha supervisionada

### `src/tools/reshuffle_train_test_split.py`

Recombina dois splits existentes, embaralha com seed fixa e gera novos `train` e `test`.

Opcionalmente remove duplicatas exatas entre os inputs, preservando a cópia do `train` e descartando a cópia correspondente do `test`. Nesse modo, o `train` mantém seu tamanho efetivo e o `test` pode encolher.

Use quando:

- você suspeita que `small_train` e `small_test` foram criados a partir de ordem temporal
- quer um split aleatório rápido sem voltar imediatamente à fonte anotada original
- precisa de um experimento controlado para medir sensibilidade à composição dos splits

Saídas:

- novo arquivo de train
- novo arquivo de test
- summary JSON opcional com tamanhos, duplicatas exatas entre os insumos, remoções aplicadas e distribuição de labels

### `src/tools/rank_pseudolabel_candidates.py`

Ranqeia candidatos de pseudolabel para revisão manual a partir de scores de registro e entidade.

Use quando:

- quer revisar os top candidatos antes de escalar pseudolabelling
- precisa misturar score de registro com penalizações por densidade, spans curtos e dominância de `Organization`
- quer exportar CSV/JSONL/HTML dos candidatos priorizados

Use este script como etapa padrão de `04_ranked_candidates`.

Observações metodológicas:

- o script penaliza densidade excessiva, cauda de scores baixos e sobrecarga de `Organization`
- `Location` dominante não deve ser tratada como suspeita por default neste domínio; por isso `--max-location-ratio` fica desabilitado e não é a recomendação operacional
- spans curtos agora usam exceções sensíveis ao corpus: abreviações locativas válidas como `RJ`, `SG`, `SJM` e `rio` não contam automaticamente como spans curtos suspeitos, enquanto marcadores isolados como `tr` e `av` continuam suspeitos
- o `candidate_quality_score` agora combina `record_score`, média de entidades e `min_entity_score` para evitar que duas entidades muito boas lavem uma entidade muito ruim

Saídas:

- CSV com features e ranking
- JSONL opcional com `_candidate_rank`
- HTML opcional para revisão visual
- summary JSON opcional com filtros e estatísticas dos selecionados

### `src/tools/compare_spacy_predictions.py`

Compara as entidades já previstas em um corpus com as entidades produzidas por um modelo spaCy no mesmo texto.

Use quando:

- quer verificar se a fragmentação vista no baseline também aparece em uma estratégia mais simples
- precisa de uma revisão lado a lado entre baseline e spaCy
- quer um controle qualitativo rápido em tips problemáticos

Saídas:

- JSONL com `baseline_entities` e `spacy_entities`
- HTML com baseline e spaCy renderizados lado a lado por registro
- summary JSON opcional com contagens por label

### `src/tools/compare_gliner_predictions.py`

Compara as entidades já previstas em um corpus com as entidades produzidas por outro modelo GLiNER no mesmo texto.

Use quando:

- quer comparar backbone puro com modelo fine-tuned
- precisa verificar se um erro já existe no backbone ou foi introduzido pelo fine-tuning
- quer uma revisão HTML lado a lado usando o mesmo conjunto de casos problemáticos

Saídas:

- JSONL com `baseline_entities` e `model_entities`
- HTML com baseline e GLiNER de comparação renderizados lado a lado
- summary JSON opcional com contagens por label

### `src/tools/compare_gliner2_predictions.py`

Compara previsões existentes com GLiNER2 base e, opcionalmente, com GLiNER2 + LoRA.

Use quando:

- quer avaliar se GLiNER2 lida melhor com casos problemáticos do corpus
- precisa comparar baseline atual do projeto contra `gliner2-base` e um adapter LoRA
- quer um HTML reprodutível para decidir se vale migrar para GLiNER2

Saídas:

- JSONL com `baseline_entities`, `gliner2_base_entities` e `gliner2_adapter_entities`
- HTML com renderização lado a lado
- summary JSON opcional com contagens por label

### `src/tools/compare_tokenizers.py`

Compara a tokenização `fast` e `slow` de um modelo HF para textos selecionados.

Use quando:

- quer investigar warnings de byte fallback e tokenizer fast convertido
- suspeita que diferenças de tokenização estão contribuindo para boundaries estranhos
- precisa inspecionar tokens e contagem de `UNK` em tips problemáticos

Saídas:

- JSON com tokens fast/slow por registro
- inclui por token: `input_id`, offsets e trecho original quando disponível
- HTML opcional para inspeção manual
- summary JSON opcional com contagem de diferenças e `UNK`

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
- quer evitar preservar a ordem temporal no arquivo amostrado

Observação:

- por padrão, a amostra preserva a ordem original dos índices sorteados
- use `--shuffle-output` para escrever os registros em ordem embaralhada

### `src/tools/sanitize_dd_corpus.py`

Sanitiza o corpus grande antes do pseudolabelling, removendo ruído estrutural e separando casos suspeitos para auditoria.

Use quando:

- você quer reduzir custo de inferência sobre relatos claramente inadequados ao domínio de denúncia
- precisa remover duplicatas e lixo textual antes do pipeline caro
- quer segregar casos suspeitos em `flagged_review` para auditoria offline

Observações metodológicas:

- além de higiene superficial, o utilitário agora descarta listas nominais off-domain e textos muito curtos sem contexto narrativo nem locativo
- relatos curtos com marcadores plausíveis de denúncia ou localização continuam preservados

### `src/tools/split_large_corpus_into_chunks.py`

Divide um corpus grande em chunks JSONL de tamanho fixo.

Use quando:

- você quer rodar pseudolabeling iterativo por partes
- precisa controlar custo por lote
- quer evitar que cada chunk represente uma janela temporal contígua

Entradas principais:

- `--input`
- `--output-dir`
- `--chunk-size`
- `--chunk-prefix`

Saída opcional:

- `--summary-json`

Observação:

- use `--shuffle-first` com `--seed` para embaralhar o corpus antes de particionar

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
- `--boosted-entities-jsonl`
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
  --model-path ../artifacts/base_model_training/experiments/multi_lr_wd_grid_bs16_with_negatives/best_overall_gliner_model \
  --input ../data/dd_corpus_small_calibration.json \
  --output-csv ../artifacts/calibration/multi_with_negatives/calibration_dataset.csv \
  --output-predictions-jsonl ../artifacts/calibration/multi_with_negatives/calibration_predictions.jsonl \
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
  --model-path ../artifacts/base_model_training/experiments/multi_lr_wd_grid_bs16_with_negatives/best_overall_gliner_model \
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
  --details-jsonl ../artifacts/pseudolabelling/multi_with_negatives_chunk02_50k_t037_cuda_v12/03_context_boost_details.jsonl \
  --summary-json ../artifacts/pseudolabelling/context_boost_audit_v12_chunk02/summary.json \
  --rows-csv ../artifacts/pseudolabelling/context_boost_audit_v12_chunk02/boosted_entities.csv \
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
  --run-root ../artifacts/pseudolabelling \
  --summary-csv ../artifacts/pseudolabelling/chunk_probe_status_t037_v13.csv
```

### `evaluate_chunk_quality.py`

```bash
cd src
python3 tools/evaluate_chunk_quality.py \
  --run-glob '../artifacts/pseudolabelling/multi_with_negatives_chunk*_50k_t037_cuda_v*' \
  --output-csv ../artifacts/pseudolabelling/chunk_quality_t037.csv \
  --output-json ../artifacts/pseudolabelling/chunk_quality_t037.json
```

### `inspect_dense_tips.py`

Use este comando apenas para auditoria de outliers densos. Para `04_ranked_candidates`, prefira `rank_pseudolabel_candidates.py`.

```bash
cd src
python3 tools/inspect_dense_tips.py \
  --input ../artifacts/pseudolabelling/multi_with_negatives_chunk04_50k_t037_cuda_v13/05_split/kept.jsonl \
  --min-entities 30 \
  --output-jsonl ../artifacts/pseudolabelling/chunk04_dense_tips.jsonl \
  --output-html ../artifacts/pseudolabelling/chunk04_dense_tips.html \
  --summary-json ../artifacts/pseudolabelling/chunk04_dense_tips_summary.json \
  --title "Chunk 04 Dense Tips"
```
