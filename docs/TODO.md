# TODO

## Normalizacao De Locais

- Criar um passo opcional de normalizacao toponimica para corpus do Disque Denuncia.
- Tratar variantes ortograficas frequentes de bairros, ruas e municipios, por exemplo `Embarie` -> `Imbariê`.
- Manter separado o texto original do denunciante e a forma normalizada.
- Definir se a normalizacao ocorre:
  - antes da construcao de seeds;
  - depois da adjudicacao literal;
  - ou em uma trilha paralela de enriquecimento geografico.
- Evitar que a normalizacao altere offsets da camada literal usada em treinamento e validacao.
- Avaliar uma estrutura dual:
  - `text` e spans literais originais;
  - `normalized_text` ou `canonical_name` apenas para enriquecimento semantico.
- Levantar um lexicon inicial de aliases topologicos comuns no Rio de Janeiro e arredores.
- Medir quantos falsos negativos atuais decorrem de erro ortografico do denunciante.
- Decidir se `review_seed_entities` pode receber spans literais semanticamente corrigiveis quando houver alta confianca geografica.
- Criar testes com casos reais: `Embarie`/`Imbariê`, `Iraja`/`Irajá`, `Sao Goncalo`/`São Gonçalo`.

## Regras Da Adjudicacao

- Tornar explicito no wrapper do benchmark que `accept` e `accept_with_edits` nao podem adicionar entidades fora de `review_seed_entities`.
- Mostrar essa restricao em `open-next` para reduzir erro operacional durante adjudicacao manual.
- Revisar se o benchmark deve permitir um modo opcional de adjudicacao semantica, separado do modo literal atual.
- Desacoplar melhor os objetivos de benchmark de desacordo e de geracao de pseudolabel de treino.
- Investigar um formato de pseudolabel mais completo por texto; `review_seed_entities` parece insuficiente como supervisao de treino.
- Projetar um lote novo voltado a treino, com anotacao mais completa e menos enviesada por desacordo literal.

## Auditoria De Corpus

- Catalogar erros recorrentes de digitacao, OCR e codificacao que afetam entidades.
- Estimar impacto desses erros em recall por label, especialmente `Location`.
- Decidir se o saneamento entra na pipeline canonica ou apenas em pipelines experimentais.

## Auditoria De Refit

- Consolidar o uso de `src/tools/audit_refit_regressions.py` como passo obrigatorio apos comparacoes `supervised_only` vs `supervised_plus_pseudolabels`.
- Adicionar analise agregada por seed para `wins/losses/ties`, `loss_reason_counts_by_label` e `wrong_label_confusions`.
- Verificar se vale exportar automaticamente os principais casos de regressao para revisao manual.
- Decidir se as tabelas experimentais devem incluir `wins`, `losses` e confusoes de label, alem de `micro_f1` e `macro_f1`.

## Resultado Experimental Atual

- O benchmark `codex_adjudication_disagreement_top100_v2` produziu sinal instavel como pseudolabel de treino.
- O benchmark `codex_adjudication_disagreement_top300_v1` teve um run unico positivo em `gliner2_training`, mas degradou em media quando repetido em multiplas seeds.
- O filtro conservador `high confidence + accept only` continuou negativo nas seeds avaliadas.
- A auditoria automatizada mostrou que o dano dominante nao e apenas `missing_entity`; aparecem principalmente:
  - `spurious_entity`
  - `wrong_label`
  - `boundary_or_partial`
- Conclusao provisoria: benchmarks de desacordo adjudicados por seed literal sao uteis para avaliacao e auditoria, mas nao devem ser promovidos automaticamente a pseudolabels de treino no formato atual.
