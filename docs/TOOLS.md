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
| `src/tools/apply_ner_score_calibrator.py` | calibração | JSONL de predições + calibrador JSON | sobrescreve saída | aplicar calibrador OOF salvo às scores de entidades NER |
| `src/tools/build_calibration_dataset.py` | calibração | JSON, JSONL | sobrescreve saída | montar dataset de calibração a partir de previsões do modelo |
| `src/tools/build_metadata_location_pseudolabels.py` | seleção | JSON, JSONL | sobrescreve saída | montar um pool conservador de pseudolabels `Location` por match literal de metadado no relato |
| `src/tools/build_metadata_multilabel_pseudolabels.py` | seleção | JSON, JSONL | sobrescreve saída | montar um pool conservador `Person+Location+Organization` a partir do candidate pool metadata-based de `Location` |
| `src/tools/profile_metadata_multilabel_signal.py` | auditoria | JSON, JSONL | sobrescreve saída | medir sinal conservador de `Person` e `Organization` dentro de um pool metadata-based já ancorado em `Location` |
| `src/tools/profile_train_oof_coverage.py` | auditoria | treino anotado + OOF JSONL | sobrescreve saída | mapear cobertura do treino e erros OOF por forma, contexto e frequência de entidade |
| `src/tools/audit_location_only_pseudolabels.py` | auditoria | pseudorrótulos `Location` + predições completas | sobrescreve saída | medir o risco de descartar predições `Person` e `Organization` em relatos `Location`-only |
| `src/tools/build_political_lexicon.py` | léxico | TSE Dados Abertos | cache + sobrescreve saída | baixar dados de candidaturas do TSE e gerar CSV de nomes/nomes de urna de políticos do RJ |
| `src/tools/political_lexicon_builder.py` | léxico | TSE Dados Abertos | biblioteca | lógica reutilizável para montar léxico político a partir de `consulta_cand` |
| `src/tools/build_train_annotation_prompt_probe.py` | auditoria | audits + lote fonte | sobrescreve saída | montar um probe pequeno e diagnóstico para testar prompts de adjudicação voltados a treino |
| `src/tools/manage_codex_adjudication_benchmark.py` | operação | JSONL de adjudicação | resumível | gerenciar benchmark chunkado de adjudicação assistida por Codex |
| `src/tools/run_llm_adjudication.py` | operação | JSONL de adjudicação | resumível | chamar a Responses API para adjudicação literal ou `train_annotation`, inclusive em chunks |
| `src/tools/optimize_context_boost_factor.py` | seleção | OOF predictions JSONL | sobrescreve saída | simular fatores de context boost sobre predições OOF e recomendar um fator |
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
| `src/tools/extract_app_dd_metadata_matches.py` | metadados | XLSX + JSON | sobrescreve saída | recuperar metadados originais de `data/app_dd.xlsx` para os corpora anotados por match textual |
| `src/tools/fit_ner_score_calibrator_oof.py` | calibração | OOF predictions JSONL | sobrescreve saída | ajustar calibrador por label a partir de predições out-of-fold |
| `src/tools/inspect_dense_tips.py` | auditoria | JSON, JSONL | sobrescreve saída | filtrar e visualizar tips com muitas entidades |
| `src/tools/prune_pseudolabel_tips.py` | limpeza | JSON, JSONL | sobrescreve saída | podar entidades de pseudolabel por score e densidade por tip |
| `src/tools/rank_pseudolabel_candidates.py` | seleção | JSON, JSONL scoreado | sobrescreve saída | selecionar top-k candidatos de pseudolabel por score de registro |
| `src/tools/select_diverse_pseudolabels.py` | seleção | JSON, JSONL scoreado | sobrescreve saída | selecionar top-k pseudorrótulos com deduplicação e limites por entidade/assinatura |
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
| `src/tools/select_top_utility_candidates.py` | seleção | JSON, JSONL scoreado | sobrescreve saída | selecionar top-K candidatos por score de utilidade e exportar JSONL/CSV/HTML |
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

### `src/tools/select_top_utility_candidates.py`

Seleciona o top-K de um corpus já scoreado por um campo numérico de utilidade.

Use quando:

- você já materializou scores como `adjudication_priority_score`, `novelty_adjusted_priority_score` ou `novelty_pool_adjusted_priority_score`
- quer comparar rapidamente diferentes critérios de ranking sem reimplementar a seleção
- precisa exportar o top-K para JSONL, CSV e HTML

Entradas principais:

- `--input`
- `--output-jsonl`
- `--ranking-field`
- `--top-n`
- `--output-html`
- filtros conservadores opcionais:
  - `--min-base-score`
  - `--max-seed-count`
  - `--exclude-low-separator-mixed-case`
  - `--prefer-location-only`

Saída:

- JSONL com os candidatos selecionados
- CSV opcional com metadados compactos
- HTML opcional usando o viewer multicamada de adjudicação

## Léxicos

### `src/tools/build_political_lexicon.py`

Driver de linha de comando para construir um léxico CSV de pessoas políticas a partir dos dados abertos de candidaturas do TSE.

Use quando:

- você precisa gerar nomes completos e nomes de urna de vereadores, deputados estaduais ou deputados federais do RJ
- quer alimentar regras ou filtros léxicos de `Person` com nomes públicos de políticos locais
- precisa reconstruir o léxico de forma reprodutível, mantendo cache local dos ZIPs do TSE

Fonte de dados:

- TSE Dados Abertos, arquivos `consulta_cand_{ano}.zip`
- URL base: `https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand/consulta_cand_{ano}.zip`

Pontos relevantes:

- baixa os ZIPs do TSE sob `--cache-dir` quando ainda não existem
- filtra por UF, cargo, ano eleitoral e município
- por padrão filtra vereadores de municípios recorrentes da região metropolitana do RJ
- com `--no-municipio-filter`, permite cargos estaduais como `DEPUTADO FEDERAL` e `DEPUTADO ESTADUAL`
- gera aliases a partir de `NM_CANDIDATO`, `NM_URNA_CANDIDATO` e, opcionalmente, primeiro token derivado do nome de urna
- grava no CSV a procedência do alias (`campo_fonte`) e uma confiança simples (`confianca`)

Entradas principais:

- `--municipios`
- `--municipios-file`
- `--no-municipio-filter`
- `--anos`
- `--uf`
- `--cargo`
- `--cache-dir`
- `--elected-only`
- `--no-derived-aliases`

Saída:

- CSV UTF-8 em `--output`
- padrão: `data/processed/lexico_politicos_locais.csv`

Exemplos:

```bash
python3 src/tools/build_political_lexicon.py \
  --anos 2024,2020 \
  --cargo VEREADOR \
  --output data/processed/lexico_politicos_locais.csv
```

```bash
python3 src/tools/build_political_lexicon.py \
  --anos 2022,2018 \
  --cargo "DEPUTADO FEDERAL,DEPUTADO ESTADUAL" \
  --no-municipio-filter \
  --output data/processed/lexico_deputados_rj.csv
```

### `src/tools/political_lexicon_builder.py`

Módulo reutilizável usado por `src/tools/build_political_lexicon.py`.

Use quando:

- você quer incorporar a construção do léxico em outro script sem passar pela CLI
- precisa testar ou reaproveitar funções de normalização, download, leitura de ZIP e geração de aliases
- quer customizar filtros ou pós-processamento mantendo a mesma estrutura de saída

Funções principais:

- `build_political_person_lexicon(...)`
- `write_lexicon_csv(...)`
- `read_municipalities_file(...)`
- `download_tse_zip(...)`
- `iter_tse_candidate_rows(...)`
- `generate_aliases(...)`

Estrutura da saída:

- cada linha é uma `LexiconRow`
- campos principais:
  - `municipio`
  - `uf`
  - `ano_eleicao`
  - `cargo`
  - `nome_completo`
  - `nome_urna`
  - `alias`
  - `partido`
  - `numero_candidato`
  - `sq_candidato`
  - `situacao_candidatura`
  - `situacao_totalizacao`
  - `fonte`
  - `campo_fonte`
  - `confianca`

Observações metodológicas:

- o módulo registra proveniência dos aliases, mas não faz inferência sobre fatos narrados no corpus
- aliases derivados têm confiança mais fraca que nome completo e nome de urna multi-token
- `--elected-only` deve ser usado só quando o objetivo for restringir o léxico a candidatos eleitos; para recall de NER, manter todos os candidatos costuma ser mais útil

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

### `src/tools/build_metadata_location_pseudolabels.py`

Monta um lote conservador de pseudolabels `Location` usando campos de metadado que aparecem literalmente no `relato`.

Use quando:

- você quer testar uma trilha `Location-only` antes de voltar a incluir `Person`
- precisa de um pool inicial de alta precisão ancorado em `bairroLocal` e `logradouroLocal`
- quer ranquear candidatos por novidade relativa ao treino supervisionado e por prioridade de `assunto`

Pontos relevantes:

- aceita JSON e JSONL
- produz um pool completo e um top-N separado para revisão manual
- pode emitir diretamente um `refit_pseudolabels.jsonl` já compatível com `train_quick.py`
- o ranking atual favorece matches de `logradouroLocal` e `bairroLocal` e penaliza topônimos já muito saturados no treino

Entradas principais:

- `--input`
- `--train`
- `--output-pool-jsonl`
- `--output-review-jsonl`
- `--summary-json`
- `--output-pseudolabel-jsonl` opcional
- `--top-n`
- `--metadata-fields`

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

### `src/tools/profile_train_oof_coverage.py`

Relaciona a cobertura do corpus de treino anotado aos erros estritos de predições OOF para uma classe alvo.

Use quando:

- você quer decidir quais lacunas de `Location` têm maior potencial para orientar a seleção de pseudorrótulos;
- precisa separar erros em menções inéditas, menções raras e menções frequentes no treino;
- quer identificar designadores, extensões de span e contextos com muitos falsos negativos;
- precisa quantificar quantos relatos com a classe alvo também contêm outras classes anotadas, antes de usar pseudorrótulos apenas de `Location`.

As métricas usam matching estrito de span (`start`, `end` e `label`) nos dados OOF. Os buckets incluem designador, comprimento da menção, frequência normalizada da menção no treino e palavra adjacente à esquerda/direita. O script não escolhe pseudorrótulos: ele produz o diagnóstico para orientar essa política.

Exemplo:

```bash
PYTHONPATH=src python3 src/tools/profile_train_oof_coverage.py \
  --train data/dd_corpus_small_train.json \
  --oof-predictions artifacts/error_analysis/train_oof_regex_for_boost_factor/oof_predictions.jsonl \
  --output-dir artifacts/pseudolabelling_analysis/train_oof_location_coverage \
  --target-label Location \
  --pred-field pred_spans_eval \
  --min-bucket-support 10
```

Saídas:

- `coverage_summary.json`: visão geral do treino e das métricas OOF da classe alvo;
- `target_bucket_metrics.csv`: métricas por bucket;
- `train_label_combinations.csv`: combinações de labels por relato de treino;
- `coverage_review.html`: tabela priorizada pelos buckets com mais falsos negativos.

### `src/tools/audit_location_only_pseudolabels.py`

Audita o risco de supervisão incompleta quando um conjunto de pseudorrótulos conserva somente entidades `Location` de relatos que originalmente receberam predições para todas as classes.

Use quando:

- você quer medir se relatos selecionados para um refit `Location`-only contêm predições `Person` ou `Organization` que seriam descartadas;
- precisa separar predições não retidas de baixa confiança daquelas que superam um limiar de risco explícito;
- quer gerar uma amostra HTML com todas as entidades originalmente previstas antes de decidir por um gate de elegibilidade.

O script associa cada pseudorrótulo selecionado à predição completa pelo identificador (`source_id`, quando disponível) ou por texto normalizado único. Correspondências ambíguas permanecem marcadas e não entram nas estatísticas de risco.

Exemplo para o atual `top500`:

```bash
PYTHONPATH=src python3 src/tools/audit_location_only_pseudolabels.py \
  --selected-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500.jsonl \
  --predictions-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions_calibrated.jsonl \
  --output-dir artifacts/pseudolabelling_analysis/location_only_top500_audit \
  --target-label Location \
  --omitted-labels Person,Organization \
  --score-fields score_calibrated,score \
  --credible-score-threshold 0.6 \
  --max-review-rows 200
```

Saídas:

- `location_only_supervision_summary.json`: taxas agregadas de predições não retidas;
- `location_only_supervision_audit.csv`: uma linha por candidato, incluindo match, contagens e nível de risco;
- `location_only_supervision_review.jsonl` e `.html`: predições completas dos relatos associados, ordenadas por risco.

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

### `src/tools/audit_ner_errors_by_label.py`

Gera uma análise de erros focada em uma classe NER a partir de um dataset gold e de um `predictions.jsonl`.

Use quando:

- você quer revisar uma classe problemática, como `Organization`, sem reler todos os erros do modelo
- precisa separar falsos positivos, falsos negativos, confusões de label e erros de fronteira
- quer priorizar revisão manual por menções repetidas e relatos com mais erros

Exemplo:

```bash
python3 src/tools/audit_ner_errors_by_label.py \
  --gold-json data/dd_corpus_small_test.json \
  --pred-jsonl artifacts/base_model_training/quick_supervised_only_regex/eval_test/predictions.jsonl \
  --output-dir artifacts/error_analysis/test_organization_regex_t06 \
  --label Organization \
  --labels Person,Location,Organization \
  --title "Organization error analysis - regex t=0.6"
```

Saídas:

- `organization_errors.jsonl`
- `organization_errors.csv`
- `organization_false_positives.csv`
- `organization_false_negatives.csv`
- `organization_review_rows.jsonl`
- `organization_error_summary.json`
- `organization_error_review.html`

### `src/tools/calibrate_ner_scores.py`

Avalia empiricamente se os scores das predições NER são úteis como confiança.

Use quando:

- você quer escolher thresholds por label para pseudorotulação
- precisa medir precisão real por faixa de score
- quer identificar erros de alta confiança antes de confiar em pseudo-labels automáticos

Exemplo com predições OOF:

```bash
cd ~/ailab/nerdd
python3 src/tools/calibrate_ner_scores.py \
  --pred-jsonl artifacts/error_analysis/train_oof_organization_regex_after_corpus_fixes/oof_predictions.jsonl \
  --output-dir artifacts/calibration/ner_scores_oof_after_corpus_fixes \
  --labels Person,Location,Organization \
  --bins 10 \
  --thresholds 0.5,0.6,0.7,0.8,0.85,0.9,0.95
```

Exemplo com predições do teste:

```bash
cd ~/ailab/nerdd
python3 src/tools/calibrate_ner_scores.py \
  --gold-json data/dd_corpus_small_test.json \
  --pred-jsonl artifacts/base_model_training/quick_supervised_only_regex/eval_test/predictions.jsonl \
  --output-dir artifacts/calibration/ner_scores_test_after_corpus_fixes \
  --labels Person,Location,Organization
```

Saídas:

- `prediction_score_rows.csv`: uma linha por entidade prevista, com `score`, `target` e `outcome`
- `precision_by_label_and_score_bin.csv`: precisão por label e faixa de score
- `precision_at_threshold_by_label.csv`: precision/recall/F1 por label em cada threshold
- `outcome_counts_by_label.csv`: contagem de `exact`, `boundary_mismatch`, `label_confusion` e `spurious`
- `calibration_summary.json`: resumo e erros de alta confiança

### `src/tools/fit_ner_score_calibrator_oof.py`

Ajusta um calibrador reutilizável de scores NER a partir de predições
out-of-fold. Cada predição vira um exemplo binário: `1` quando `(start, end,
label)` coincide exatamente com o gold, `0` caso contrário.

Use quando:

- quer corrigir superconfiança dos scores GLiNER sem usar o conjunto de teste;
- já tem um `oof_predictions.jsonl` emitido por `mine_train_oof_errors.py`;
- precisa calibrar scores antes de aplicar context boost ou seleção top-k.

Saídas:

- `calibrator.json`
- `calibration_examples.csv`
- `reliability_raw_by_label.csv`
- `reliability_calibrated_by_label.csv`
- `calibration_summary.json`

Exemplo:

```bash
PYTHONPATH=src python3 src/tools/fit_ner_score_calibrator_oof.py \
  --oof-predictions artifacts/error_analysis/train_oof_regex_for_boost_factor/oof_predictions.jsonl \
  --output-dir artifacts/calibration/ner_score_calibrator_oof_regex \
  --labels Person,Location,Organization \
  --method isotonic \
  --score-field score \
  --pred-field pred_spans \
  --gold-field gold_spans \
  --min-positive 20 \
  --min-negative 20 \
  --bins 10 \
  --log-level INFO
```

### `src/tools/apply_ner_score_calibrator.py`

Aplica um `calibrator.json` salvo a um JSONL de predições, adicionando
`score_calibrated` em cada entidade com score válido.

Exemplo:

```bash
PYTHONPATH=src python3 src/tools/apply_ner_score_calibrator.py \
  --input-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/01_predictions_calibrated.jsonl \
  --calibrator artifacts/calibration/ner_score_calibrator_oof_regex/calibrator.json \
  --score-field score \
  --output-score-field score_calibrated \
  --entity-key entities \
  --log-level INFO
```

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
- o default de `--batch-size` em `train_quick.py` é `16`
- aceita configuração por JSON com `--config-json` e `--experiment-id`
- salva o modelo treinado em `best_model/` dentro do diretório de saída

Exemplo:

```bash
cd ~/ailab/nerdd
python3 src/base_model_training/train_quick.py \
  --config-json configs/experiments/base_model_finetuned.json \
  --experiment-id quick_supervised_only_regex_seed42
```

### `src/tools/run_experiment_config.py`

Executa experimentos declarados em arquivos JSON.

Use quando:

- quer versionar configurações de experimento fora da linha de comando
- quer rodar um experimento específico de uma lista
- quer rodar todos os experimentos declarados em um arquivo
- quer expandir um experimento em múltiplas seeds usando `n_repeats` e `seed_start`

Exemplos:

```bash
cd ~/ailab/nerdd
python3 src/tools/run_experiment_config.py \
  --config-json configs/experiments/base_model_finetuned.json \
  --experiment-id quick_supervised_only_regex_seed42
```

```bash
cd ~/ailab/nerdd
python3 src/tools/run_experiment_config.py \
  --config-json configs/experiments/base_model_finetuned.json \
  --all
```

Quando `n_repeats > 1`, o runner cria subdiretórios por repetição, por exemplo:

```text
artifacts/base_model_training/quick_supervised_only_regex/
  repeat_01_seed53/
    best_model/
  repeat_02_seed54/
    best_model/
```

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

Seleciona candidatos de pseudolabel para revisão manual a partir de um ou mais campos de score de registro.

Use quando:

- quer revisar os top candidatos antes de escalar pseudolabelling
- já computou um score por relato com `pseudolabelling.compute_record_scores`
- precisa fixar um orçamento top-k, por exemplo `1000`, `3000` ou `5000`
- quer exportar CSV/JSONL/HTML dos candidatos priorizados

Use este script como etapa padrão de `04_ranked_candidates`.

Controles principais:

- `--score-fields`: campos candidatos de score, em ordem de preferência
- `--required-labels`: exige pelo menos uma entidade com os labels listados
- `--min-score`: descarta registros abaixo do score mínimo
- `--top-n`: limita o volume final

Saídas:

- JSONL com os registros selecionados e metadados em `_pseudolabel_selection`
- CSV com ranking e contagens por label
- HTML opcional para revisão visual
- summary JSON opcional com filtros, score field usado e estatísticas dos selecionados

Exemplo para o piloto `Location`:

```bash
PYTHONPATH=src python3 src/tools/rank_pseudolabel_candidates.py \
  --input artifacts/pseudolabelling/frozen_baseline_regex_seed42/05d_scored_context_boost_location_p75.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000.jsonl \
  --output-csv artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000.csv \
  --output-html artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000.html \
  --summary-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/07_location_p75_top1000_summary.json \
  --score-fields record_score_location \
  --required-labels Location \
  --min-score 0.80 \
  --top-n 1000 \
  --title "Location p75 top 1000 pseudolabel candidates"
```

### `src/tools/select_diverse_pseudolabels.py`

Seleciona um top-k diverso a partir de um pool de pseudorrótulos já scoreado.
É a alternativa preferida quando o top-k puro concentra muitos relatos quase
duplicados ou repete excessivamente as mesmas entidades `Location`.

Use quando:

- o ranking por score gerou concentração alta em poucos locais;
- você quer manter um orçamento fixo, como `top500`, mas aumentar variedade;
- precisa reduzir quase duplicatas textuais antes do refit;
- quer limitar quantos relatos de uma mesma entidade ou conjunto de entidades entram no lote.

Controles principais:

- `--score-fields`: campos candidatos de score, inclusive caminhos aninhados como `_pseudolabel.record_score_location`;
- `--target-labels`: labels usados para deduplicação semântica, normalmente `Location`;
- `--max-per-entity`: máximo de relatos selecionados por termo normalizado;
- `--max-per-signature`: máximo de relatos por assinatura de conjunto de entidades;
- `--signature-max-terms`: quantos termos entram na assinatura;
- `--top-n`: orçamento final de registros.

Saídas:

- JSONL com registros selecionados e `_pseudolabel_selection`;
- summary JSON com contadores de seleção/rejeição;
- CSV opcional com auditoria de decisões;
- HTML opcional para revisão visual.

Exemplo para gerar um `top500` diverso do pool `Location` calibrado:

```bash
PYTHONPATH=src python3 src/tools/select_diverse_pseudolabels.py \
  --input artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097.jsonl \
  --output-jsonl artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse.jsonl \
  --summary-json artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse_summary.json \
  --audit-csv artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse_audit.csv \
  --output-html artifacts/pseudolabelling/frozen_baseline_regex_seed42/04_pseudolabels_location_calibrated_t097_top500_diverse.html \
  --top-n 500 \
  --score-fields record_score_location,_pseudolabel.record_score_location \
  --target-labels Location \
  --max-per-entity 10 \
  --max-per-signature 2 \
  --signature-max-terms 8 \
  --title "Diverse Location pseudolabels t097 top500"
```

### `src/tools/optimize_context_boost_factor.py`

Simula fatores de context boost sobre predições OOF já geradas, sem retreinar modelo.

Use quando:

- quer substituir um `boost_factor` arbitrário por um valor estimado no treino em regime OOF;
- tem um `oof_predictions.jsonl` com `pred_spans`, `gold_spans` e scores;
- os metadados já estão no OOF ou podem ser recuperados por match exato normalizado de texto com `--metadata-sources`;
- precisa escolher o fator antes de aplicar boost ao corpus não anotado.

Observação:

- não use `data/large/large_sanitized_no_labeled_overlap.jsonl` como fonte de metadados para OOF de treino, pois esse arquivo exclui por construção relatos que aparecem nos corpora anotados;
- para recuperar metadados de OOF de treino, prefira as fontes sanitizadas originais com sobreposição possível, como `data/large_sanitized/large_sanitized.jsonl`, `data/large_sanitized/large_dropped.jsonl` e `data/large_sanitized/large_flagged.jsonl`;
- o script só enriquece linhas com match de texto único e não ambíguo.

Saídas:

- `boost_factor_metrics.csv`
- `boost_factor_summary.json`
- `boost_factor_recommendation.json`
- `promoted_entities.jsonl`
- `promoted_records.jsonl`
- `boost_factor_review.html`

Exemplo:

```bash
PYTHONPATH=src python3 src/tools/optimize_context_boost_factor.py \
  --oof-predictions artifacts/error_analysis/train_oof_regex_for_boost_factor/oof_predictions.jsonl \
  --output-dir artifacts/boost_factor_optimization/context_location_oof_with_metadata_sources \
  --metadata-sources data/large_sanitized/large_sanitized.jsonl,data/large_sanitized/large_dropped.jsonl,data/large_sanitized/large_flagged.jsonl \
  --metadata-source-text-fields relato,text \
  --boost-factors 1.00,1.05,1.10,1.15,1.20,1.30,1.50 \
  --target-label Location \
  --score-thresholds 0.6,0.7,0.8,0.9,0.95 \
  --record-score-aggregation p75 \
  --record-thresholds 0.8,0.9,0.95 \
  --precision-floor 0.90 \
  --boost-scope location-matched-only \
  --match-policy any-metadata-in-text \
  --log-level INFO
```

### `src/tools/extract_app_dd_metadata_matches.py`

Recupera metadados originais de localização do arquivo legado `data/app_dd.xlsx`
e os associa aos corpora anotados atuais por match textual normalizado.

Use quando:

- precisa enriquecer `train`, `test` e `calibration` com `cidadeLocal`, `logradouroLocal`, `bairroLocal` e `pontodeReferenciaLocal`;
- quer auditar a cobertura de metadados antes de rodar otimização de context boost;
- o arquivo `app_dd.xlsx` está com linhas quebradas por vírgulas não escapadas.

Observações:

- o script lê `.xlsx` diretamente, sem depender de `openpyxl`;
- corrige mojibake simples como `TrÃ¡fico` -> `Tráfico`;
- tenta reconstruir relatos quebrados e extrai os metadados nas células imediatamente posteriores ao relato casado;
- só considere automaticamente os casos `matched_unique` com `has_geo_metadata=True`; casos ambíguos ou sem match ficam no CSV de auditoria.

Exemplo:

```bash
python3 src/tools/extract_app_dd_metadata_matches.py \
  --app-xlsx data/app_dd.xlsx \
  --labeled-input train:data/dd_corpus_small_train.json \
  --labeled-input test:data/dd_corpus_small_test.json \
  --labeled-input calibration:data/dd_corpus_small_calibration.json \
  --output-jsonl artifacts/metadata/app_dd_labeled_metadata_matches.jsonl \
  --audit-csv artifacts/metadata/app_dd_labeled_metadata_matches_audit.csv \
  --summary-json artifacts/metadata/app_dd_labeled_metadata_matches_summary.json
```

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

### `build_metadata_location_pseudolabels.py`

```bash
cd src
python3 tools/build_metadata_location_pseudolabels.py \
  --input ../artifacts/corpus_sanitization/dd_corpus_large_sanitized.jsonl \
  --train ../data/dd_corpus_small_train.json \
  --output-pool-jsonl ../artifacts/benchmarks/metadata_location_literal_top20_v1/candidate_pool.jsonl \
  --output-review-jsonl ../artifacts/benchmarks/metadata_location_literal_top20_v1/top100_review.jsonl \
  --summary-json ../artifacts/benchmarks/metadata_location_literal_top20_v1/summary.json \
  --output-pseudolabel-jsonl ../artifacts/benchmarks/metadata_location_literal_top20_v1/refit_pseudolabels_top100.jsonl \
  --top-n 100 \
  --metadata-fields logradouroLocal,bairroLocal
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
