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

## Auditoria De Corpus

- Catalogar erros recorrentes de digitacao, OCR e codificacao que afetam entidades.
- Estimar impacto desses erros em recall por label, especialmente `Location`.
- Decidir se o saneamento entra na pipeline canonica ou apenas em pipelines experimentais.
