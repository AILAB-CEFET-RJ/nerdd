# Manual do editor de anotações NER

Este manual descreve como iniciar e usar o editor local de anotações para corpora no formato usado por `data/dd_corpus_small_train.json` e `data/dd_corpus_small_test.json`.

O editor trabalha com as classes:

- `Person`
- `Location`
- `Organization`

As decisões de anotação devem seguir `docs/LABELLING_GUIDE.md`.

## Modo recomendado: servidor com salvamento direto

Use este modo quando quiser editar o arquivo do dataset e salvar as alterações diretamente no disco.

Para o conjunto de treino:

```bash
python3 src/tools/serve_ner_annotation_editor.py \
  --input data/dd_corpus_small_train.json \
  --host 127.0.0.1 \
  --port 8765 \
  --title "NER editor - train" \
  --labels Person,Location,Organization
```

Para o conjunto de teste:

```bash
python3 src/tools/serve_ner_annotation_editor.py \
  --input data/dd_corpus_small_test.json \
  --host 127.0.0.1 \
  --port 8765 \
  --title "NER editor - test" \
  --labels Person,Location,Organization
```

Depois abra no navegador:

```text
http://127.0.0.1:8765/
```

Se a porta `8765` já estiver em uso, troque por outra, por exemplo `8766`.

## Salvamento

No modo servidor, o botão `Save to dataset` grava as anotações no próprio arquivo passado em `--input`.

Antes de substituir o arquivo, o servidor cria automaticamente um backup na mesma pasta, com sufixo semelhante a:

```text
dd_corpus_small_train.json.bak-20260729-153012
```

O salvamento valida se:

- o número de registros continua igual;
- cada registro tem campo `text`;
- `spans` é uma lista;
- cada span tem `start`, `end` e `label` válidos;
- os offsets não ultrapassam o tamanho do texto.

## Modo alternativo: HTML estático

Use este modo quando quiser gerar um arquivo HTML independente e baixar o JSON corrigido pelo navegador.

```bash
python3 src/tools/build_ner_annotation_editor_global.py \
  --input data/dd_corpus_small_train.json \
  --output artifacts/annotation_review/dd_corpus_small_train_editor_global.html \
  --title "NER editor - train" \
  --labels Person,Location,Organization
```

Nesse modo, o botão `Export corrected JSON` baixa um arquivo chamado `annotations_corrected.json`. Ele não substitui automaticamente o dataset original.

## Navegação

Use os botões:

- `Previous`: relato anterior;
- `Next`: próximo relato;
- `Go`: ir para um relato pelo índice 1-based do dataset.

O título `Record #N` mostra o índice original do relato no arquivo. Esse é o índice que deve ser usado para localizar o relato no JSON.

Atalhos:

- `[`: relato anterior;
- `]`: próximo relato;
- `Delete` ou `Backspace`: remover entidade selecionada;
- `N`: adicionar o texto selecionado como entidade.

Os atalhos são ignorados enquanto você está digitando em campos de texto ou seletores.

## Filtro por string

O campo `Contains:` permite mostrar apenas relatos cujo `text` contenha uma string informada.

Características:

- a busca ignora diferença entre maiúsculas e minúsculas;
- a correspondência é literal por substring;
- a navegação `Previous`/`Next` passa a percorrer apenas os relatos filtrados;
- o contador mostra a posição dentro dos resultados filtrados e o índice original do dataset.

Exemplo de contador com filtro ativo:

```text
match 3 / 12 | record 508 / 4226
```

Isso significa: terceiro resultado entre 12 relatos encontrados; o relato corresponde ao índice 508 no dataset original.

Para limpar o filtro, clique em `Clear filter` ou pressione `Escape` dentro do campo `Contains:`.

O botão `Go` sempre usa o índice original do dataset. Se houver filtro ativo, ele limpa o filtro e vai para o relato solicitado.

## Adicionar uma entidade

Para adicionar uma entidade:

1. selecione com o mouse o trecho exato no texto;
2. escolha a classe no seletor;
3. clique em `Add selected text`.

O editor não permite adicionar uma entidade que sobreponha outra já existente. Se precisar corrigir limites, remova a entidade errada e adicione a versão correta.

## Alterar classe de uma entidade

Para alterar a classe:

1. clique na entidade destacada no texto;
2. escolha a nova classe no seletor;
3. clique em `Change label`.

## Remover uma entidade

Para remover uma entidade:

1. clique na entidade destacada no texto;
2. clique em `Remove`.

Também é possível usar `Delete` ou `Backspace` depois de selecionar a entidade.

## Correções globais

O editor inclui ações globais para acelerar correções repetitivas.

### Remover mesma string + classe em todo o corpus

Use `Remove same text+label everywhere` quando uma menção foi anotada indevidamente em vários relatos.

Exemplo:

- texto selecionado: `polícia`;
- classe: `Organization`.

A ação remove todas as anotações com exatamente essa menção normalizada e essa classe. Antes de aplicar, o editor mostra quantos spans e quantos relatos serão afetados.

### Trocar classe da mesma string em todo o corpus

Use `Change same text+label everywhere` quando uma menção recorrente recebeu uma classe errada.

Exemplo:

- texto selecionado: `CV`;
- classe atual: `Location`;
- classe escolhida no seletor: `Organization`.

A ação troca todas as ocorrências anotadas com a mesma string normalizada e a mesma classe original.

## Exportar regras globais

O editor permite exportar:

- `Export remove rules`: regras de remoção;
- `Export all global rules`: regras de remoção, trocas de classe e log de ações.

As regras de remoção podem ser reaplicadas em uma nova sessão com:

```bash
python3 src/tools/serve_ner_annotation_editor.py \
  --input data/dd_corpus_small_train.json \
  --banlist-input annotation_banlist.json
```

## Boas práticas

- Salve com frequência no modo servidor.
- Após uma sequência grande de mudanças, verifique se o arquivo JSON ainda abre corretamente.
- Use o filtro `Contains:` para revisar strings suspeitas, como nomes de marcas, siglas policiais, abreviações de locais ou falsos positivos recorrentes.
- Antes de aplicar correções globais, confira a prévia de spans e relatos afetados.
- Para decisões de fronteira e classe, consulte sempre `docs/LABELLING_GUIDE.md`.

## Documentos relacionados

- `docs/LABELLING_GUIDE.md`: regras de decisão para classe e fronteira de spans.
- `docs/PIPELINE_OVERVIEW.md`: visão atual do pipeline de treino, auditoria e pseudorrotulagem.
- `docs/PSEUDOLABELING_ROADMAP.md`: plano e log de progresso dos experimentos de pseudorrotulagem.
