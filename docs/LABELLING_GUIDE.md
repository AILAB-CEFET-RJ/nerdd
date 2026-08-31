Este documento apresenta as **regras gerais de anotação**. O corpus considera as classes `Person`, `Location` e `Organization`, em relatos informais nos quais erros ortográficos e variações de escrita são preservados. 

# Diretrizes de anotação

## Regra 1. Princípio geral: anotar entidades individualizadas

Uma expressão deve ser anotada quando identifica uma pessoa, um local ou uma organização específica. Expressões meramente genéricas, descritivas ou anafóricas não devem ser anotadas.

Exemplos anotáveis:

* `Roseni` → `Person`
* `favela do Muquiço` → `Location`
* `14 BPM` → `Organization`

Exemplos não anotáveis:

* `um homem`
* `os bandidos`
* `a rua`
* `a comunidade`
* `o local`
* `uma igreja`
* `a polícia`, quando usada genericamente

Numeral ou ordinal isolado deve ser `Organization` quando funcionar textualmente como forma abreviada denominativa de uma unidade policial específica, como em `policiais do [14]_{Organization}`, `PMs do [15]_{Organization}` e `policial do [41]_{Organization}`.

---

## Regra 2. A classificação depende do referente contextual

A classe não deve ser definida apenas pela forma lexical da expressão. Deve-se verificar o que ela representa naquela ocorrência.

Exemplos:

* `Jordão` em “um rapaz chamado `[Jordão]_{Person}`”
* `Jordão` como nome de bairro ou comunidade → `Location`
* `Botafogo` como bairro → `Location`
* `Botafogo` como clube → `Organization`
* `Merck` como empresa que realiza uma ação → `Organization`
* `laboratório Merck` como edifício usado para localizar uma ocorrência → `Location`
* `Saracuruna` como município ou destino → `Location`;
* Quando `Saracuruna` for usado para designar a linha de ônibus (ex. “o Saracuruna foi assaltado”), e não o destino geográfico → não anotar;
* `Petrópolis` em “sentido Petrópolis” → `Location`;
* `Petrópolis` em “a linha Petrópolis estava atrasada” → não anotar.

Assim, uma mesma forma textual pode receber classes diferentes em contextos distintos.

Ocorrências semanticamente equivalentes de uma mesma entidade devem receber a mesma classe. Isso não significa atribuir sempre a mesma classe a formas homônimas: o referente contextual continua sendo decisivo. Exemplos:

* `ABRAXAS`, quando designa a produtora → `Organization`;
* `Vila 3`, quando designa a localidade → `Location`;
* `Queimados`, quando designa o município → `Location`.

---

## Regra 3. Delimitar a menção completa, sem elementos externos

O span deve corresponder à expressão nominal que identifica a entidade, incluindo seus elementos internos naturais, mas excluindo preposições, artigos ou palavras que apenas introduzem externamente a menção.

Exemplos:

* em “na rua Nina Ribeiro”, anotar `rua Nina Ribeiro`;
* em “da favela do Muquiço”, anotar `favela do Muquiço`;
* em “chamado Jordão”, anotar apenas `Jordão`;
* em “senhor Tafarel”, anotar apenas `Tafarel`.

Não devem ser incluídos:

* pontuação adjacente;
* preposições externas, como `na`, `da` ou `em`;
* formas de tratamento, como `senhor`, `doutor` ou `dona`, quando não integram o nome;
* números de imóveis, CEPs e complementos de endereço.

---

## Regra 4. Incluir designadores geográficos imediatamente associados ao nome

Quando um designador geográfico aparece imediatamente associado a uma denominação própria e ajuda a caracterizar o lugar, ele deve integrar o span de `Location`.

Designadores típicos:

* bairro;
* centro;
* favela;
* comunidade;
* morro;
* vila;
* rua;
* avenida;
* autoestrada;
* alameda;
* travessa;
* praça;
* ponte;
* estação;
* condomínio;
* cidade;
* município;
* estado.

Exemplos:

* `bairro Lins de Vasconcelos`
* `bairro de Inhauma`
* `centro do Alcântara`
* `favela do Muquiço`
* `comunidade Mangueirinha`
* `morro da Casa Branca`
* `Rua Tenente Souza`
* `Auto estrada Grajaú Jacarepaguá`
* `estação de trem de Manguinhos`, quando a expressão identificar uma estação específica.

Não anotar o designador isolado quando não houver denominação específica:

* `o bairro`
* `a rua`
* `uma comunidade`
* `a estação`

> Quando palavras como `cidade`, `município` e `estado` integrarem a expressão nominal que identifica o lugar, elas devem fazer parte do span, juntamente com as preposições internas:
>
> * `cidade de São Gonçalo` → `Location`
> * `cidade São Gonçalo` → `Location`
> * `município de Queimados` → `Location`
> * `município Queimados` → `Location`
> * `estado do Rio de Janeiro` → `Location`
>
> Não incluir o designador quando ele funcionar apenas como rótulo externo, título de campo ou indicação metalinguística:
>
> * “Cidade: `PARATY`” → anotar apenas `PARATY`
> * “Município: `Queimados`” → anotar apenas `Queimados`

Essa distinção é importante porque resolve uma aparente tensão com o exemplo da Regra 8:

> Cidade: PARATY RJ

Nesse caso, `Cidade:` é um **rótulo de campo**, e não parte da denominação. Portanto, continuam corretos os spans separados:

* `PARATY` → `Location`
* `RJ` → `Location`

---

## Regra 5. Manter preposições internas à denominação

Artigos, preposições e contrações devem integrar o span quando conectam internamente o designador ao nome do local.

Exemplos:

* `favela do Muquiço`
* `comunidade do triângulo de Deodoro`
* `bairro de Inhauma`
* `centro do Alcântara`
* `estação de trem de Manguinhos`, quando a expressão identificar uma estação específica.

Esses elementos não são externos à entidade: fazem parte da forma textual pela qual o local foi mencionado.

---

## Regra 6. Separar locais coordenados

Quando dois ou mais locais distintos aparecem coordenados, cada um deve receber um span próprio.

Exemplo:

> na esquina das ruas Inhumay com Nina Ribeiro

Anotação:

* `Inhumay` → `Location`
* `Nina Ribeiro` → `Location`

Não anotar `ruas Inhumay com Nina Ribeiro` como uma única entidade, pois a passagem identifica dois logradouros.

A mesma regra vale para:

* cruzamentos;
* enumerações;
* trajetos;
* construções com `e`, `com`, `entre` ou `até`.

---

## Regra 7. Deixar fora o designador compartilhado por uma lista

Quando um único designador introduz duas ou mais entidades coordenadas, ele deve permanecer fora dos spans.

Exemplos:

> entre as ruas Dona Luisa e Engenho da Rainha

Anotação:

* `Dona Luisa` → `Location`
* `Engenho da Rainha` → `Location`

> nas ruas General Canrobert e Lins Rego

Anotação:

* `General Canrobert` → `Location`
* `Lins Rego` → `Location`

Em ocorrências independentes, contudo, o designador deve ser incluído normalmente:

* `rua Dona Luisa`
* `rua Engenho da Rainha`

---

## Regra 8. Separar unidades geográficas diferentes

Unidades geográficas distintas devem ser anotadas em spans separados, mesmo quando aparecem lado a lado e mantêm uma relação hierárquica.

Exemplos:

> Bairro São Mateus São João de Meriti

* `Bairro São Mateus` → `Location`
* `São João de Meriti` → `Location`

> Cidade: PARATY RJ

* `PARATY` → `Location`
* `RJ` → `Location`

> Engenheiro Pedreira Japeri

* `Engenheiro Pedreira` → `Location`
* `Japeri` → `Location`

Não se deve fundir bairro, município e estado em uma única entidade.

> A inclusão de um designador geográfico não autoriza reunir unidades geográficas distintas. Em “município de Queimados/RJ”, anotar:
>
> * `município de Queimados` → `Location`
> * `RJ` → `Location`

Assim, a regra operacional fica:

* designador integrado à expressão → incluir: `município de Queimados`;
* designador usado como rótulo → excluir: `Município: Queimados`;
* unidade hierárquica adicional → span separado: `município de Queimados` + `RJ`.

---

## Regra 9. Estruturas físicas exigem individualização

Estações, pontes, passarelas, praças, viadutos, túneis, terminais, condomínios e outras estruturas físicas devem ser anotadas como `Location` somente quando forem individualizadas por nome próprio ou denominação específica.

Anotar como `Location` quando a expressão funcionar como nome ou denominação específica do estabelecimento:

* `condomínio da Merck`, quando a expressão identificar um condomínio específico;
* `estação de trem de Manguinhos`, quando a expressão identificar uma estação específica;
* `Ponte do Colégio Murilo Braga`;
* `praça Benfica`;
* `Bar do Ruan`, quando essa for a denominação pela qual o estabelecimento é identificado;
* `Mercado do João`, quando essa for a denominação do estabelecimento.

Não anotar:

* `a estação`
* `o metrô do bairro`
* `a passarela do metrô`
* `o largo`
* `uma ponte`
* `o posto de combustível`
* `o ponto de moto táxi`

A mera capacidade de indicar uma posição espacial não transforma uma expressão genérica em entidade nomeada.

A simples presença de um nome de pessoa após um nome comum de lugar não torna automaticamente toda a expressão uma entidade `Location`. Deve-se distinguir uma denominação do lugar de uma descrição de posse, residência ou associação circunstancial.

Quando o nome de uma pessoa fizer parte da denominação de um estabelecimento físico, anote toda a expressão como `Location`, sem criar um span interno sobreposto de `Person`.

Exemplo:

 * `[Bar do Ruan]_{Location}`, quando `Bar do Ruan` funcionar como nome do estabelecimento;
 * não anotar simultaneamente `Ruan` como `Person`.

Se a expressão apenas descrever um estabelecimento associado a uma pessoa, sem funcionar como denominação, anote somente a pessoa:

 * `o bar do [Ruan]_{Person}`, quando significar apenas “o bar pertencente a Ruan”;
 * `o mercado do [João]_{Person}`, quando significar apenas “o mercado pertencente a João”.

A diferença entre maiúsculas e minúsculas pode servir como indício, mas **não deve ser o critério decisivo**, dada a escrita irregular do corpus.

Estruturas físicas associadas a uma organização podem ser anotadas integralmente como `Location` quando essa associação, juntamente com o contexto, individualizar uma estrutura ou ponto de referência específico. O simples uso espacial da expressão não é suficiente.
Exemplos:

* `Próximo à [antena da Oi]_{Location}`, quando identificar uma antena específica;
* `Na [torre da Vivo]_{Location}`, quando identificar uma torre específica;
* `Escondidos no [depósito da Petrobras]_{Location}`, quando identificar um depósito específico.

Nesses casos, não se deve criar simultaneamente um span interno para a organização:

* `antena da Oi` → `Location`
* não anotar `Oi` separadamente como `Organization`.

Ver também Regra 14 para o critério de quando uma marca ou empresa deve ser anotada como `Organization` em vez de `Location`.

Complexos industriais, refinarias, centros comerciais e unidades de saúde individualizadas devem ser classificados como `Location` quando forem usados como pontos de embarque, desembarque, destino, referência espacial ou local de atendimento.

Exemplos:

* “os assaltantes descem na `[Reduc]_{Location}`”;
* “embarcam no `[Caxias Shopping]_{Location}`”;
* “foi levado ao `[posto de saúde de Campos Elísios]_{Location}`”, quando a expressão identificar convencionalmente uma unidade específica.

Caso a entidade atue institucionalmente, usar `Organization`:

* “a `[Reduc]_{Organization}` divulgou uma nota”;
* “o `[Caxias Shopping]_{Organization}` alterou seu horário”.

### Residências associadas a pessoas

 Expressões formadas por um nome comum de residência e um nome de pessoa, como `casa de X`, `apartamento de X` e `sítio de X`, normalmente descrevem uma relação de posse, residência ou associação. Nesses casos, não anote toda a expressão como `Location`; anote somente o nome da pessoa como `Person`.

Exemplos:

* `casa da [MC Carol]_{Person}`;
* `casa do [Gabriel]_{Person}`;
* `apartamento de [Roseni]_{Person}`;
* `sítio do [Zé]_{Person}`.

O fato de a expressão permitir localizar uma ocorrência não a transforma, por si só, em nome próprio de lugar. Ela deve ser tratada como descrição relacional quando equivaler a expressões como “a casa dela” ou “a residência pertencente a Gabriel”.

A expressão completa somente deve ser anotada como `Location` quando funcionar como nome próprio ou denominação convencional do lugar, e não apenas como indicação de seu proprietário, morador ou ocupante.

Exemplos:

* `[Casa de Cultura Laura Alvim]_{Location}`;
* `[Casa da Pedra]_{Location}`, quando for o nome pelo qual o lugar é identificado;
* `[Sítio do Picapau Amarelo]_{Location}`, quando funcionar como denominação própria;
* `[Casa da MC Carol]_{Location}`, somente se o contexto indicar que essa é a denominação de um estabelecimento, espaço cultural, casa de festas ou outro lugar nomeado.

Não anotar referências genéricas ou anafóricas:

* `a casa`;
* `casa dela`;
* `sua residência`;
* `uma casa com piscina`.

---

## Regra 10. Instituições com uso espacial são classificadas pelo referente

Nomes de igrejas, escolas, hospitais, empresas, laboratórios, quartéis e instituições semelhantes podem designar tanto a organização quanto seu edifício ou estabelecimento.

Quando a instituição atua como agente, classificar como `Organization`:

* “A `[Assembléia de Deus]_{Organization}` organizou o evento.”
* “A `[Merck]_{Organization}` divulgou uma nota.”
* “O `[Colégio Murilo Braga]_{Organization}` suspendeu as aulas.”

Quando o texto se refere ao edifício ou espaço físico, classificar como `Location`:

* “escondidos na `[igreja evangélica Assembléia de Deus]_{Location}`”;
* “entre o `[laboratório Merck]_{Location}` e o posto”;
* “em frente ao `[condomínio da Merck]_{Location}`”, quando a expressão identificar um condomínio específico;
* “na entrada do `[Colégio Murilo Braga]_{Location}`”.

A função referencial no contexto prevalece sobre a natureza institucional abstrata.

Instituições também podem ser mencionadas por uma forma abreviada, popular ou metonímica. Quando essa expressão designar o estabelecimento físico no qual alguém está, esteve, entrou ou saiu, deve ser classificada como `Location`.

Exemplos:

* “recém-saído do `[Padre Severino]_{Location}`”;
* “foi levado para o `[Salgado Filho]_{Location}`” → referência é ao hospital;
* “está internado no `[Getúlio Vargas]_{Location}`” → referência é ao hospital.

Se a mesma denominação representar a instituição como agente administrativo, usar `Organization`.

---

## Regra 11. Evitar spans sobrepostos ou aninhados

O esquema adotado não deve manter spans sobrepostos. Quando uma entidade integra a denominação de outra entidade mais ampla, deve-se anotar apenas a expressão completa correspondente ao referente contextual.

Exemplo:

* `UPP da Cidade de Deus` → `Organization`
* não anotar simultaneamente `Cidade de Deus` como `Location` dentro do mesmo span.

Em outra ocorrência independente:

* “oriundos da `[Cidade de Deus]_{Location}`”

Outro exemplo:

* `Ponte do Colégio Murilo Braga` → `Location`
* não anotar simultaneamente `Colégio Murilo Braga` como entidade interna.

A proibição de spans sobrepostos aplica-se apenas depois de determinado o referente principal da expressão. Ela não obriga a escolher sempre o span mais longo. Quando uma construção como `casa de X`, `apartamento de X` ou `sítio de X` for uma descrição relacional, e não o nome próprio de um lugar, anote somente `X` como `Person`. Quando a expressão completa funcionar como denominação do lugar, anote-a como `Location` e não crie um span interno de `Person`. Exemplos:

| Expressão no contexto                         | Anotação                           |
| --------------------------------------------- | ---------------------------------- |
| residência da cantora                         | casa da `[MC Carol]_{Person}`      |
| simples posse                                 | apartamento de `[Roseni]_{Person}` |
| estabelecimento denominado “Bar do Ruan”      | `[Bar do Ruan]_{Location}`         |
| apenas o bar pertencente a Ruan               | bar do `[Ruan]_{Person}`           |
| espaço cultural denominado “Casa da MC Carol” | `[Casa da MC Carol]_{Location}`    |

---

## Regra 12. Anotar todas as ocorrências explícitas de entidades nomeadas

Todas as ocorrências em que uma entidade for mencionada explicitamente por um nome, sigla, apelido ou denominação individualizadora devem ser anotadas, mesmo que a mesma entidade já tenha aparecido anteriormente no relato.

Exemplo:

* primeira ocorrência de `Rua Nina Ribeiro` → `Location`;
* segunda ocorrência de `Rua Nina Ribeiro` → `Location`;
* ocorrência posterior de `Nina Ribeiro` → `Location`, pois a expressão ainda contém o nome individualizador do local.

Uma ocorrência anterior não elimina a necessidade de anotar as ocorrências posteriores. Entretanto, a anotação deve seguir a convenção de **NER estrito**: cada ocorrência precisa conter, por si própria, uma expressão nomeada ou individualizadora. O contexto pode ajudar a determinar o referente e a classe da menção, mas não transforma uma expressão genérica em entidade nomeada.

Não devem ser anotadas referências anafóricas ou descrições genéricas, ainda que retomem inequivocamente uma entidade mencionada anteriormente:

* `essa rua`;
* `o local`;
* `a comunidade`;
* `lá`;
* `o brisolão`;
* `a empresa`;
* `a organização`.

Compare:

* primeira ocorrência: `rua Nonato Farias` → `Location`;
* ocorrência posterior: `Nonato Farias` → `Location`, pois conserva o nome próprio individualizador;
* ocorrência posterior: `essa rua` → não anotar;
* ocorrência posterior: `a rua` → não anotar.

Da mesma forma:

* primeira ocorrência: `brisolão Sérgio Cardoso` → `Location`;
* ocorrência posterior: `Sérgio Cardoso` → `Location`, se a expressão for empregada como forma abreviada do nome do local;
* ocorrência posterior: `o brisolão` → não anotar, mesmo que o antecedente seja inequívoco.

Quando o nome completo e uma sigla ou forma abreviada nomeada aparecerem explicitamente, cada ocorrência deve receber seu próprio span:

* `[Polícia Rodoviária Federal]_{Organization}`;
* `[PRF]_{Organization}`.

Formas populares, alternativas ou abreviadas também devem ser anotadas quando funcionarem como denominação própria da entidade, e não apenas como referência anafórica genérica.

O critério decisivo é:

> A expressão presente no texto contém uma denominação que individualiza a entidade?

* Se sim, anote a ocorrência.
* Se a identificação depender exclusivamente da recuperação de um antecedente, não anote.

A resolução de que diferentes menções se referem à mesma entidade pertence a uma etapa posterior de correferência ou vinculação de entidades e não altera os spans de NER.

---

## Regra 13. Nomes próprios, apelidos e nomes alternativos que individualizam pessoas

Nomes próprios, prenomes isolados, apelidos, nomes alternativos e nomes informais devem ser anotados como `Person` quando o contexto indicar que identificam um indivíduo específico.

O fato de a menção conter apenas um prenome, apelido ou outro elemento informal não impede sua anotação.

Exemplos:

* `[Roseni]_{Person}`;
* `[Leo]_{Person}`;
* `[Tafarel]_{Person}`;
* `[Ruan]_{Person}`;
* `[Daniel]_{Person}`;
* `[Bomba]_{Person}`;
* `[Netinho]_{Person}`;
* `[Negão]_{Person}`;
* `[Vando Perereca]_{Person}`;
* `[Jiló]_{Person}`.

Expressões que apenas introduzem ou qualificam o nome, o apelido ou a alcunha devem permanecer fora do span.

Exemplos:

* `vulgo [Negão]_{Person}`;
* `conhecido como [Jiló]_{Person}`;
* `chamado [Jordão]_{Person}`;
* `tem o apelido de [Negão]_{Person}`.

Não devem ser anotadas expressões genéricas ou meramente descritivas que não individualizem uma pessoa.

Exemplos:

* `o rapaz`;
* `a vítima`;
* `a esposa`;
* `o menor`;
* `os moradores`;
* `seis traficantes`.

O contexto pode determinar se uma forma funciona como nome ou apelido de uma pessoa, mas não transforma uma descrição genérica em entidade nomeada.

---

## Regra 14. Marcas, produtos, plataformas e veículos de empresas

Quando o nome de uma empresa ou instituição integrar uma expressão que, no contexto, individualize uma estrutura física ou um ponto de referência específico, anote a expressão completa como `Location`, sem criar um span interno de `Organization`. O simples fato de a estrutura ser usada como cenário, origem ou destino do evento não é suficiente: expressões genéricas ou não individualizadas, como `uma loja da BMW` ou `uma agência do banco`, não devem ser anotadas como `Location`.

Nomes de marcas, empresas e instituições devem ser anotados como `Organization` quando designarem a própria entidade organizacional no contexto, inclusive quando houver vínculo institucional ou de pertencimento com ela. Não devem ser anotados quando servirem apenas para identificar a marca de um produto, veículo, aparelho, mercadoria ou serviço, nem quando designarem aplicativos, plataformas ou meios utilizados para realizar uma ação. Nomes presentes apenas em URLs também não devem ser anotados.

Exemplos:

* `A [BMW]_{Organization} anunciou um novo modelo` → empresa agente.
* `O governo multou a [BMW]_{Organization}` → `BMW` é `Organization`: empresa afetada, embora não seja agente.
* `Funcionários da [BMW]_{Organization} entraram em greve` → vínculo institucional.
* `Traficantes do [CV]_{Organization} foram presos` → pertencimento à organização.
* `Anda de BMW branca` → não anotar `BMW`: é marca do veículo.
* `Carga de carne Friboi` → não anotar `Friboi`: marca do produto.
* `Enviou pelo WhatsApp` → não anotar `WhatsApp`: plataforma usada como meio.

Um teste prático: no contexto, a expressão permite identificar uma organização participante ou relacionada ao evento, ou apenas responde "de que marca/tipo é este objeto ou meio"?

Assim:

* "traficantes do CV" → identifica pertencimento organizacional; anotar `CV`;
* "carne Friboi" → responde apenas qual é a marca da carne; não anotar `Friboi`;
* "carga pertencente à Friboi" → identifica a empresa proprietária; anotar `Friboi`;
* "carga de carne Friboi" → marca do produto; não anotar `Friboi`.

**Relação com a Regra 9 (estruturas físicas associadas a uma organização)**. Quando a menção da marca ou empresa designa o edifício, a loja, a agência, o depósito ou outro espaço físico associado a ela, usado como local do fato, ponto de referência, origem ou destino, a classe correta é `Location`, conforme a Regra 9 — **não** `Organization`. O critério de vínculo/pertencimento desta regra não se sobrepõe ao uso espacial: a presença de uma preposição de posse (`da`, `do`) não torna a menção automaticamente uma `Organization`. Exemplos:

* “assaltaram a `[loja da BMW]_{Location}`”, quando a expressão identificar uma loja específica;
* “escondido no `[depósito da Friboi]_{Location}`”, quando a expressão funcionar como denominação individualizadora de um depósito específico;
* “a `[BMW]_{Organization}` multou o revendedor” (empresa como agente).

**Sobre veículos identificados pela empresa proprietária**. Quando um veículo é mencionado apenas por marca ou modelo, sem que a empresa proprietária seja identificada como vítima, agente ou parte lesada do fato, a marca não deve ser anotada (Regra 23). Quando o relato identificar explicitamente a empresa como proprietária ou vítima do veículo (e não apenas como fabricante), a empresa deve ser anotada como `Organization`. Exemplos:

* "roubaram uma van Sprinter da empresa" → não anotar `Sprinter`: marca sem organização nomeada;
* "roubaram a van da Friboi" → anotar `Friboi` como `Organization`, pois identifica a empresa afetada pelo roubo;
* "fugiram num Gol prata" → não anotar: marca do veículo (Regra 23).

**Sobre contas e perfis em plataformas**. Quando a menção a uma plataforma ou aplicativo designa uma conta, perfil ou canal individualizado que é alvo, vítima ou objeto da ação (por exemplo, invadido, clonado ou usado para criar um perfil falso), a plataforma continua não sendo anotada como `Organization`: ela permanece o meio ou suporte da ação, não uma organização agente. Exceção: quando o relato atribuir a ação à própria empresa como agente institucional. Exemplos:

* "clonaram o WhatsApp da vítima" → não anotar `WhatsApp`;
* "O Facebook dele é aberto" → não anotar `Facebook`;
* "criou um perfil falso no Instagram" → não anotar `Instagram`;
* "o Instagram removeu o perfil" → `[Instagram]_{Organization}`, por metonímia institucional: ação é atribuída à empresa (isto é, empresa como agente).

---

## Regra 15. Nome civil e apelido recebem spans próprios

Quando um nome e um apelido aparecem como formas distintas de identificação, cada forma deve ser anotada separadamente.

Exemplos:

> Rogério, vulgo Gerinho

* `Rogério` → `Person`
* `Gerinho` → `Person`

> Tafarel, que também tem apelido de Negão

* `Tafarel` → `Person`
* `Negão` → `Person`

Não se deve criar um único span incluindo palavras como `vulgo`, `apelido de` ou `conhecido como`.

---

## Regra 16. Denominações alternativas para o mesmo local

Quando um mesmo local for mencionado por diferentes denominações explícitas — como nome oficial, nome antigo, apelido, forma popular ou sigla —, cada menção individualizadora deve ser anotada como `Location`.

Na tarefa de NER, as diferentes denominações são anotadas conforme aparecem no texto. A identificação de que elas se referem ao mesmo lugar pertence a uma etapa posterior de normalização ou vinculação de entidades e não altera a delimitação dos spans.

### Denominações no mesmo relato

Quando duas ou mais denominações do mesmo local aparecerem em um único relato, todas devem ser anotadas separadamente.

Exemplo:

> A operação ocorreu na `[comunidade Mangueirinha]_{Location}`, também conhecida como `[Palha Seca]_{Location}`.

Nesse caso, são anotados dois spans de `Location`, ainda que as expressões designem o mesmo lugar.

Outros exemplos:

* `[Rodovia Presidente Dutra]_{Location}`, também chamada `[Via Dutra]_{Location}`;
* `[Avenida Brasil]_{Location}`, conhecida na região como `[Brasil]_{Location}`, quando esta forma funcionar textualmente como denominação da via;
* `[Complexo do Alemão]_{Location}`, também referido como `[Alemão]_{Location}`.

Expressões meramente introdutórias, como `também conhecida como`, `chamada`, `denominada` ou `mais conhecida como`, não integram o span.

### Denominações em relatos diferentes

Quando diferentes relatos empregarem denominações distintas para um mesmo lugar, cada denominação explícita deve ser anotada normalmente no relato em que ocorrer.

Exemplo:

* Relato 1: “A operação aconteceu na `[Palha Seca]_{Location}`”;
* Relato 2: “Houve confronto na `[comunidade Mangueirinha]_{Location}`”.

A eventual associação entre `Palha Seca` e `comunidade Mangueirinha` pertence à etapa posterior de normalização ou vinculação de entidades, caso essa etapa faça parte do projeto.

### Expressões genéricas ou anafóricas

Não devem ser anotadas expressões que apenas retomem ou descrevam genericamente o local, sem apresentar uma denominação individualizadora.

Exemplos não anotáveis:

* `a comunidade`;
* `o local`;
* `a região`;
* `o mesmo lugar`;
* `aquela área`.

O fato de o contexto permitir identificar a qual lugar essas expressões se referem não as transforma em entidades nomeadas.

### Critério de decisão

| Situação                                     | Procedimento na anotação NER                                   |
| -------------------------------------------- | -------------------------------------------------------------- |
| Duas denominações explícitas no mesmo relato | Anotar cada denominação em seu próprio span                    |
| Denominações diferentes em relatos distintos | Anotar cada denominação normalmente                            |
| Expressão genérica ou anafórica              | Não anotar                                                     |
| Denominações que se referem ao mesmo lugar   | A eventual associação pertence à etapa posterior de vinculação |

A equivalência entre os nomes não elimina nenhuma menção nem produz um span único. Cada denominação explícita e individualizadora deve ser anotada conforme aparece no texto, independentemente de eventual normalização ou vinculação posterior.

---

## Regra 17. Unidades policiais numeradas ou individualizadas por complemento

Expressões formadas por número cardinal, número ordinal ou ordinal escrito por extenso seguido de `BPM` identificam uma unidade policial específica e devem ser anotadas integralmente. A classe depende do referente no contexto:

* usar `Organization` quando a expressão designar a unidade policial como instituição, agente ou grupo;
* usar `Location` quando a expressão designar claramente o prédio, a sede ou o espaço físico ocupado pela unidade.

Exemplos como `Organization`:
* “policiais do `[9 BPM]_{Organization}` foram acionados”;
* “o `[14 BPM]_{Organization}` realizou uma operação”;
* “o comando do `[7º BPM]_{Organization}` informou o ocorrido”.

Exemplos como `Location`:
* “o baile acontece a menos de 300 metros do `[9 BPM]_{Location}`”;
* “o veículo foi abandonado em frente ao `[14 BPM]_{Location}`”;
* “o suspeito foi visto próximo ao `[7º BPM]_{Location}`”.

Também se aplica às formas expandidas:

* `14º Batalhão de Polícia Militar`;
* `Décimo Quarto Batalhão de Polícia Militar`;
* `9º Batalhão`, quando o contexto policial for inequívoco.

O número ou ordinal deve fazer parte do span. Não anotar apenas `BPM`.

Variações claramente decorrentes de erro de digitação, como `BMP`, podem ser anotadas integralmente quando o contexto tornar inequívoca a referência a um BPM específico.

Unidades policiais não numeradas podem ser individualizadas por complemento geográfico:

* “há corrupção no `[DPO dá Maria Paula]_{Organization}`”;
* “o carro foi abandonado em frente ao `[DPO dá Maria Paula]_{Location}`”.

A forma `dá` reproduz exatamente a grafia presente na ocorrência original do corpus e deve ser preservada conforme a Regra 20.

---

## Regra 18. Outras unidades e siglas policiais específicas

Siglas e denominações que identificam órgãos, unidades ou setores policiais específicos devem ser anotadas. A classe depende do referente da expressão no contexto:

* quando designar a instituição, a unidade policial, seus integrantes como agente coletivo ou o setor funcional, anotar como `Organization`;
* quando designar inequivocamente o prédio, a sede, a base ou outro espaço físico ocupado pela unidade, anotar como `Location`.

### Uso institucional

Exemplos:

* “O `[BOPE]_{Organization}` realizou a operação”;
* “Agentes da `[CORE]_{Organization}` foram acionados”;
* “O `[GAT]_{Organization}` prendeu os suspeitos”;
* “A `[PMERJ]_{Organization}` divulgou uma nota”;
* “A `[P2]_{Organization}` investigava o caso”;
* “A `[UPP da Cidade de Deus]_{Organization}` intensificou o patrulhamento”.

Nesses casos, as expressões designam órgãos, unidades, setores ou agentes coletivos, e não suas instalações físicas.

### Uso espacial

Quando a expressão designar inequivocamente as instalações da unidade policial, deve ser anotada como `Location`.

Exemplos:

* “O veículo foi abandonado em frente ao `[BOPE]_{Location}`”, quando `BOPE` designar claramente a sede ou a base da unidade;
* “O suspeito foi levado para a `[UPP da Cidade de Deus]_{Location}`”, quando a referência for inequivocamente ao prédio ou à base física;
* “A vítima esperava na porta da `[CORE]_{Location}`”, quando a expressão designar as instalações da unidade.

O simples emprego de uma preposição espacial, como `em`, `no`, `na`, `para` ou `perto de`, não determina automaticamente a classe. É necessário verificar se a expressão designa a instituição ou o espaço físico:

* “A denúncia foi encaminhada à `[CORE]_{Organization}`” — destinatário institucional;
* “O carro estava estacionado diante da `[CORE]_{Location}`” — instalação física.

### Unidades identificadas por complemento

Expressões que contenham complemento explicitamente individualizador também devem ser anotadas quando designarem uma unidade policial específica:

* “O `[batalhão de São Gonçalo]_{Organization}` realizou a operação”;
* “Policiais da `[UPP da Cidade de Deus]_{Organization}` foram acionados”.

O complemento deve estar presente na própria expressão e individualizar a unidade. Não basta que sua identidade possa ser recuperada apenas pelo contexto ou por uma menção anterior.

### Referências genéricas

Não anotar expressões que não individualizem uma organização ou unidade específica:

* `a polícia`;
* `os policiais`;
* `as autoridades`;
* `o batalhão`, sem individualização;
* `a unidade`;
* `a equipe policial`.

O critério decisivo é aquilo que a expressão individualizadora designa no contexto: a unidade ou o setor policial como entidade institucional recebe `Organization`; suas instalações físicas recebem `Location`.


---

## Regra 19. Facções, milícias e outros grupos criminosos nomeados

Nomes, siglas, nomes alternativos e denominações informais que individualizem uma facção, milícia ou outro grupo criminoso devem ser anotados como `Organization`.

Exemplos:

* `[CV]_{Organization}`;
* `[ADA]_{Organization}`;
* `[Terceiro Comando Puro]_{Organization}`;
* `[Liga da Justiça]_{Organization}`;
* `[milícia de Santa Cruz]_{Organization}`, quando a expressão denominar um grupo específico.

### Inclusão do designador organizacional

Designadores como `facção`, `grupo`, `milícia`, `quadrilha` e `organização` devem integrar o span quando estiverem diretamente associados ao nome ou à sigla e compuserem, juntamente com eles, uma menção nominal à organização.

Preposições e artigos internos também devem ser incluídos.

Exemplos:

* `[facção CV]_{Organization}`;
* `[facção do CV]_{Organization}`;
* `[facção Comando Vermelho]_{Organization}`;
* `[milícia de Santa Cruz]_{Organization}`.

### Designador acompanhado de modificador descritivo

Quando houver um modificador meramente descritivo entre o designador e o nome ou a sigla, anote somente o elemento que individualiza a organização. O designador, o modificador e as preposições que apenas os conectem ao nome permanecem fora do span.

Exemplos:

* `facção criminosa do [CV]_{Organization}`;
* `facção rival [ADA]_{Organization}`;
* `antiga facção rival [ADA]_{Organization}`;
* `facção covarde [CV]_{Organization}`;
* `Grupo Terrorista [Liga da Justiça]_{Organization}`;
* `grupo criminoso denominado [Liga da Justiça]_{Organization}`.

A presença apenas de preposição ou artigo não interrompe a menção nominal:

* `[facção do CV]_{Organization}`;
* `[milícia de Santa Cruz]_{Organization}`.

Por outro lado, expressões introdutórias não integram o span:

* `a organização chamada [Liga da Justiça]_{Organization}`;
* `o grupo conhecido como [Comando Vermelho]_{Organization}`;
* `integrante da facção rival [ADA]_{Organization}`.

### Expressões genéricas

Não devem ser anotadas expressões que apenas descrevam um tipo de agrupamento criminoso, sem identificar um grupo específico.

Exemplos não anotáveis:

* `a facção`;
* `facção rival`;
* `facção criminosa`;
* `líderes da facção`;
* `mesma facção`;
* `grupo criminoso`;
* `crime organizado`;
* `organização criminosa`;
* `uma milícia`;
* `integrante da facção`;
* `integrante de uma facção rival`.

O fato de o contexto permitir inferir de qual grupo se trata não transforma uma expressão genérica ou anafórica em entidade nomeada. O nome, a sigla ou outro elemento individualizador deve estar presente na própria expressão.

Compare:

* `integrante do [CV]_{Organization}`;
* `integrante da [facção CV]_{Organization}`;
* `integrante da facção rival [ADA]_{Organization}`;
* `integrante da facção` → não anotar.

### Nomes de locais na denominação de grupos criminosos

Quando o nome de um local fizer parte da denominação que individualiza o grupo criminoso, toda a menção deve receber apenas a classe `Organization`. Não deve ser criado um span sobreposto de `Location`.

Exemplo:

* `[milícia de Santa Cruz]_{Organization}`.

Nesse caso, `Santa Cruz` não deve ser anotado separadamente como `Location`.

Quando a expressão designar apenas criminosos ou um grupo não nomeado que atua em determinado lugar, anote o local normalmente:

* `criminosos de [Santa Cruz]_{Location}`;
* `uma milícia que atua em [Santa Cruz]_{Location}`;
* `um grupo criminoso de [Queimados]_{Location}`.

O critério decisivo é verificar se o nome geográfico integra a denominação de uma organização específica ou apenas informa seu local de atuação.

---

## Regra 20. Preservar a forma original do texto

A anotação deve reproduzir exatamente o trecho presente no relato, incluindo:

* erros ortográficos;
* ausência de acentos;
* uso inconsistente de maiúsculas;
* abreviações;
* grafias não padronizadas.

Exemplos:

* `sao gonçalo`
* `inhumay`
* `marechal hermes`
* `negao`
* `7BPM`

Não se deve corrigir o texto dentro da anotação. Eventual normalização deve ocorrer em uma etapa separada.

---

## Regra 21. Não acrescentar entidades por inferência externa

Somente entidades explicitamente presentes no relato devem ser anotadas.

Se uma rua pertence a um município conhecido, o município não deve ser acrescentado caso seu nome não apareça no texto. Da mesma forma, não se deve completar:

* siglas;
* nomes incompletos;
* hierarquias geográficas;
* nomes de instituições;

com base apenas em conhecimento externo.

O contexto pode ser usado para escolher a classe, mas não para inventar uma menção ausente.

---

## Regra 22. Números e elementos de endereço ficam fora do span

Normalmente não são entidades das três classes:

* número de imóvel;
* CEP;
* bloco;
* apartamento;
* protocolo;
* placa de veículo;
* horário;
* idade;
* quantidade de pessoas.

Exemplo:

> Rua Tenente Souza, 190, numeração antiga 236

Anotar apenas:

* `Rua Tenente Souza` → `Location`

Não incluir `190` ou `236`.

Exceção: números que integram o próprio nome do lugar ou da organização:

* `Vila 3` → `Location`
* `Rua 72` → `Location`
* `14 BPM` → `Organization`

---

## Regra 23. Veículos, placas e frases não pertencem às classes

Não devem ser anotados:

* marcas ou modelos de veículos, usados apenas para descrever o veículo em si (ver Regra 14 para o caso em que a marca identifica a própria empresa como agente, vítima ou proprietária do bem);
* cores de veículos;
* placas;
* objetos;
* slogans;
* frases religiosas;
* tipos de armas;
* horários e datas.

Exemplos não anotáveis:

* `Logan Prata`
* `Gol Vermelho`
* `LQK 9672`
* `pistola 380`
* `20:00`
* "Não foi sorte, foi Deus"
* `van da empresa`, quando o texto não nomeia a empresa proprietária

---

## Regra 24. Linhas, itinerários e veículos de transporte não são entidades anotáveis

Nomes de linhas, itinerários, rotas, serviços e veículos de transporte não devem ser anotados, mesmo quando coincidem com nomes geográficos.

A anotação depende do referente da expressão no contexto:

* quando designar uma localidade ou um destino geográfico, anotar como `Location`;
* quando designar uma linha, rota, itinerário, serviço, veículo ou composição, não anotar.

### Linha ou veículo

Exemplo:

> As linhas são Saracuruna e Piabetá, da Viação União.

Anotação:

* `Saracuruna` → não anotar, pois designa uma linha;
* `Piabetá` → não anotar, pois designa uma linha;
* `[Viação União]_{Organization}`.

Exemplo:

> O Saracuruna foi assaltado duas vezes.

Nesse contexto, `Saracuruna` designa a linha ou o veículo, e não a localidade. Portanto, não deve ser anotado.

Outros exemplos não anotáveis:

* `o ônibus Saracuruna`;
* `a linha Petrópolis`;
* `o Piabetá das seis horas`;
* `o coletivo Magé`;
* `o trem Japeri`, quando designar o serviço ou a composição.

### Localidade ou destino geográfico

Exemplo:

> As linhas seguem para Saracuruna e Piabetá.

Anotação:

* `[Saracuruna]_{Location}`;
* `[Piabetá]_{Location}`.

Nesse caso, as expressões designam os destinos geográficos das linhas, e não as próprias linhas.

### Critério de decisão

O anotador deve identificar o referente contextual, observando elementos como:

* `linha`, `ônibus`, `coletivo`, `trem`, `composição`, `itinerário` ou horário associado → não anotar;
* `para`, `até`, `em`, `na direção de`, `chegar a` ou outra indicação de lugar ou destino → anotar como `Location`, quando a expressão efetivamente designar a localidade.

A simples coincidência lexical com o nome de um lugar não transforma a menção em `Location`. O que determina a anotação é aquilo que a expressão designa no contexto.

---

## Regra 25. Estabelecimentos, estruturas e pontos de referência sem nome próprio formal

### Princípio geral

Expressões que apenas descrevem um estabelecimento, uma estrutura física ou um ponto de referência, sem conter elemento que funcione como denominação individualizadora, não devem ser anotadas.

Exemplos não anotáveis:

* `uma igreja`;
* `um bar`;
* `a barraca de frutas`;
* `o laboratório`, sem nome;
* `o colégio`, sem nome;
* `o condomínio`, sem nome;
* `o posto`;
* `a passarela`;
* `o campo`;
* `o bar da esquina`.

O fato de o contexto permitir inferir qual lugar está sendo mencionado não transforma uma expressão genérica ou anafórica em entidade nomeada.

### Estruturas individualizadas pela própria expressão

Uma estrutura sem nome próprio formal pode ser anotada como `Location` quando a própria expressão contiver uma denominação convencional ou um complemento que funcione como elemento individualizador do lugar.

Para que a expressão seja anotada:

1. o elemento individualizador deve estar explicitamente presente no texto;
2. a expressão deve funcionar como denominação de um estabelecimento, estrutura ou ponto de referência específico;
3. a individualização não pode depender exclusivamente de uma menção anterior, de inferência contextual ou do conhecimento particular do anotador.

Exemplos anotáveis:

* `[passarela do metrô de Manguinhos]_{Location}`, quando a expressão funcionar como denominação do ponto de referência;
* `[viaduto de Benfica]_{Location}`, quando for a denominação convencional do viaduto;
* `[igreja da Candelária]_{Location}`;
* `[sede do Flamengo]_{Location}`, quando identificar uma sede específica;
* `[Posto Shell da BR-101]_{Location}`, quando a expressão individualizar um estabelecimento específico.

Exemplos não anotáveis:

* `o posto da BR-101`, se a expressão não identificar convencionalmente um único posto;
* `uma igreja`;
* `um campo do Flamengo`;
* `o campo do Flamengo`, quando constituir apenas uma descrição relacional;
* `o condomínio da empresa`, sem elemento que individualize o condomínio.

A presença de um complemento geográfico ou institucional não garante, por si só, a anotação. O complemento deve fazer a expressão funcionar como denominação individualizadora.

### Decisão contextual entre `Organization` e `Location`

Quando a expressão contiver um nome próprio ou outro elemento individualizador, a classe dependerá do referente contextual:

* entidade institucional, empresa ou grupo como agente, responsável, fonte ou destinatário → `Organization`;
* edifício, estabelecimento, estrutura ou ponto físico usado como lugar, origem, destino ou referência espacial → `Location`.

Exemplos:

* “A `[Assembleia de Deus]_{Organization}` realizou um culto”;
* “Os suspeitos estavam escondidos na `[igreja da Candelária]_{Location}`”;
* “O `[Posto Shell da BR-101]_{Location}` foi assaltado”;
* “A `[Shell]_{Organization}` divulgou uma nota sobre o ocorrido”;
* “A manifestação ocorreu em frente à `[sede do Flamengo]_{Location}`”;
* “O `[Flamengo]_{Organization}` divulgou a escalação”.

### Estabelecimentos associados a pessoas

A associação com uma pessoa somente torna o estabelecimento anotável como `Location` quando a expressão funcionar como sua denominação.

Exemplos:

* `[Bar do João]_{Location}`, quando `Bar do João` for o nome pelo qual o estabelecimento é identificado;
* `o bar de [João]_{Person}`, quando a expressão indicar apenas posse e não denominar o estabelecimento;
* `um bar do João` → anotar somente `[João]_{Person}`, se aplicável.

### Síntese para o anotador

| Expressão               | Anotação                                           | Justificativa                                                           |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------------------------- |
| `um bar`                | Não anotar                                         | Expressão genérica                                                      |
| `o bar da esquina`      | Não anotar                                         | Descrição espacial, sem denominação individualizadora                   |
| `Bar do João`           | `Location`, se for denominação                     | Nome do estabelecimento                                                 |
| `o bar de João`         | Somente `João` como `Person`, se indicar posse     | A expressão não denomina necessariamente o estabelecimento              |
| `o posto da BR-101`     | Não anotar automaticamente                         | A rodovia, por si só, pode não individualizar um único posto            |
| `Posto Shell da BR-101` | `Location`, se individualizar um estabelecimento   | A expressão funciona como denominação específica                        |
| `viaduto de Benfica`    | `Location`, se for denominação convencional        | Ponto de referência individualizado pela própria expressão              |
| `igreja da Candelária`  | `Location` quando designar o edifício              | Denominação reconhecível de uma estrutura física                        |
| `campo do Flamengo`     | Não anotar automaticamente                         | Pode indicar diferentes campos e constituir apenas descrição relacional |
| `sede do Flamengo`      | `Location`, quando identificar uma sede específica | Estrutura física individualizada                                        |

Em caso de dúvida sobre se a expressão funciona como denominação individualizadora ou constitui apenas uma descrição contextual, prevalece a **não anotação**. O caso deve ser sinalizado para revisão conforme o procedimento de adjudicação do corpus.

---

## Regra 26. Casos ambíguos devem ser resolvidos pelo contexto ou reservados para revisão

Expressões como `Botafogo`, `Jordão`, `Beira Mar`, `Armando`, `Ampla` e `Providência` podem ter interpretações diferentes.

Procedimento recomendado:

1. examinar verbos, preposições e expressões introdutórias;
2. verificar se a menção identifica pessoa, espaço ou instituição;
3. não decidir apenas pela capitalização;
4. encaminhar para revisão quando o texto não fornecer evidência suficiente.

Exemplos de pistas:

* “chamado X” favorece `Person`;
* “no bairro X” favorece `Location`;
* “a empresa X informou” favorece `Organization`;
* “escondidos na igreja X” favorece `Location`.

Não atribua simultaneamente duas classes à mesma menção. Se a ambiguidade persistir, encaminhe o caso para adjudicação. A anotação final deve conter uma única classe.

---

## Regra 27. Prefeituras e governos

Anotar como `Organization` quando a expressão individualizar uma administração pública específica.

**Anotar:**

* `[Prefeitura de São Gonçalo]_{Organization}`
* `[Governo do Estado do Rio de Janeiro]_{Organization}`

Não anotar formas isoladas e genéricas, mesmo que o contexto permita inferir a instituição pretendida.

**Não anotar:**

* `a prefeitura`
* `o governo`
* `o município`
* `o Estado`

---

## Regra 28. Números e códigos que identificam rodovias

Números ou códigos devem ser anotados como `Location` quando, no contexto, funcionarem como denominação suficientemente individualizadora de uma rodovia ou via específica.

Exemplos:

* `BR-116`
* `Rodovia BR-116`
* `116`, quando funcionar textualmente como forma abreviada denominativa da rodovia BR-116;
* `040`, quando funcionar textualmente como forma abreviada denominativa da rodovia BR-040.

Exemplo:

> passando agora pela `[116]_{Location}` e daqui a pouco estará na `[040]_{Location}`

Deve-se preservar exatamente a forma presente no texto. Não acrescentar elementos que não aparecem no relato. Portanto, se o texto contém apenas `116`, o span deve ser `116`, e não `BR-116`.

Números sem função denominativa ou sem contexto suficiente para identificar uma via específica não devem ser anotados.

O número isolado somente deve ser anotado quando funcionar textualmente como forma abreviada da denominação da rodovia; não basta ser possível inferir, por conhecimento externo, a qual via ele se refere.

---

## Regra 29. Pontos de atividade criminosa individualizados

Expressões como `boca`, `boca de fumo`, `ponto de venda`, `ponto do tráfico` e semelhantes podem ser anotadas como `Location` quando a própria expressão funcionar como denominação individualizadora de um ponto físico específico.

A presença de um complemento geográfico, por si só, não garante a anotação da expressão integral. É necessário distinguir entre:

* uma denominação que individualiza o ponto físico;
* uma descrição que apenas informa o local em que a atividade ocorre.

### Expressões individualizadoras

Exemplos:

* `[boca do Campo Novo]_{Location}`, quando a expressão funcionar como denominação de um ponto específico;
* `[boca da Pedreira]_{Location}`, quando a expressão denominar convencionalmente um ponto específico;
* `[ponto do tráfico da Rua Nonato Farias]_{Location}`, quando toda a expressão funcionar como denominação individualizadora.

Quando o nome geográfico integrar a denominação completa do ponto, deve-se anotar somente a expressão integral, sem criar um span interno sobreposto:

* `[boca do Campo Novo]_{Location}`;
* não anotar simultaneamente `Campo Novo` como `Location` dentro desse span.

### Localização sem denominação do ponto

Quando a expressão apenas indicar que uma atividade criminosa ocorre em determinado local, não se deve anotar integralmente o ponto. O nome geográfico continua sendo anotado normalmente.

Exemplos:

* `uma boca em [Campo Novo]_{Location}`;
* `uma boca que funciona na [Pedreira]_{Location}`;
* `um ponto de venda na [Rua Nonato Farias]_{Location}`.

### Expressões genéricas ou anafóricas

Não anotar expressões que não contenham elemento individualizador:

* `a boca está aberta`;
* `fecham na boca`;
* `uma boca de fumo`;
* `o ponto de venda`;
* `o ponto do tráfico`;
* `o tráfico instalou uma boca`.

O contexto pode ajudar a determinar se a expressão funciona como denominação, mas não transforma uma referência genérica ou anafórica em entidade nomeada. Em caso de dúvida, prevalece a não anotação e o caso deve ser encaminhado para adjudicação.

---

# Síntese operacional

Para cada candidato a entidade, o anotador deve responder, nesta ordem:

1. **A expressão individualiza um referente?**
   Se não, não anotar.

2. **O referente é pessoa, local ou organização?**
   Determinar pelo contexto.

3. **Qual é a menção textual completa?**
   Incluir designadores e elementos internos; excluir introdutores externos.

4. **Há duas entidades coordenadas?**
   Criar spans separados.

5. **Há uma entidade interna dentro de outra mais ampla?**
   Anotar apenas o referente completo e evitar sobreposição.

6. **A forma está repetida em outro ponto do relato?**
   Anotar novamente cada ocorrência explícita.

7. **Há erro ortográfico?**
   Preservá-lo exatamente como aparece.

8. **A decisão depende de inferência externa?**
   Não acrescentar informação ausente; encaminhar casos realmente ambíguos para revisão.
