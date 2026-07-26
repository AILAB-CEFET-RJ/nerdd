Este documento apresenta as **regras gerais de anotação**. O corpus considera as classes `Person`, `Location` e `Organization`, em relatos informais nos quais erros ortográficos e variações de escrita são preservados. 

# Diretrizes consolidadas de anotação

## 1. Princípio geral: anotar entidades individualizadas

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

---

## 2. A classificação depende do referente contextual

A classe não deve ser definida apenas pela forma lexical da expressão. Deve-se verificar o que ela representa naquela ocorrência.

Exemplos:

* `Jordão` em “um rapaz chamado Jordão” → `Person`
* `Jordão` como nome de bairro ou comunidade → `Location`
* `Botafogo` como bairro → `Location`
* `Botafogo` como clube → `Organization`
* `Merck` como empresa que realiza uma ação → `Organization`
* `laboratório Merck` como edifício usado para localizar uma ocorrência → `Location`
* `Saracuruna` como município ou destino → `Location`;
* `Saracuruna` como linha ou ônibus em “o Saracuruna foi assaltado” → não anotar;
* `Petrópolis` em “sentido Petrópolis” → `Location`;
* `Petrópolis` em “a linha Petrópolis estava atrasada” → não anotar.

Isso tornaria mais concreta a afirmação já existente de que “a classificação depende do referente contextual”.

Assim, uma mesma forma textual pode receber classes diferentes em contextos distintos.

---

## 3. Delimitar a menção completa, sem elementos externos

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

## 4. Incluir designadores geográficos imediatamente associados ao nome

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
* condomínio.

Exemplos:

* `bairro Lins de Vasconcelos`
* `bairro de Inhauma`
* `centro do Alcântara`
* `favela do Muquiço`
* `comunidade Mangueirinha`
* `morro da Casa Branca`
* `Rua Tenente Souza`
* `Auto estrada Grajaú Jacarepaguá`
* `estação de trem de Manguinhos`

Não anotar o designador isolado quando não houver denominação específica:

* `o bairro`
* `a rua`
* `uma comunidade`
* `a estação`

### Designadores organizacionais diretamente associados ao nome

Quando um designador organizacional aparece diretamente associado ao nome ou à sigla de uma organização e contribui para formar a menção completa, ele deve integrar o span.

Exemplos:

* `facção CV` → `Organization`
* `Grupo Terrorista Liga da Justiça` → `Organization`
* `Banco Citibank` → `Organization`
* `empresa LMB Empreendimentos` → `Organization`, quando a expressão completa funcionar como denominação da entidade.

Expressões meramente introdutórias podem ficar fora do span quando não integram a forma denominativa:

* “uma empresa denominada `LMB Empreendimentos`”
* “a organização chamada `Liga da Justiça`”

A decisão deve considerar se o designador compõe a menção nominal ou apenas introduz o nome.

---

## 5. Manter preposições internas à denominação

Artigos, preposições e contrações devem integrar o span quando conectam internamente o designador ao nome do local.

Exemplos:

* `favela do Muquiço`
* `comunidade do triângulo de Deodoro`
* `bairro de Inhauma`
* `centro do Alcântara`
* `estação de trem de Manguinhos`

Esses elementos não são externos à entidade: fazem parte da forma textual pela qual o local foi mencionado.

---

## 6. Separar locais coordenados

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

## 7. Deixar fora o designador compartilhado por uma lista

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

## 8. Separar unidades geográficas diferentes

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

---

## 9. Estruturas físicas exigem individualização

Estações, pontes, passarelas, praças, viadutos, túneis, terminais, condomínios e outras estruturas físicas devem ser anotadas como `Location` somente quando forem individualizadas por nome próprio ou denominação específica.

Anotar:

* `estação de trem de Manguinhos`
* `Ponte do Colégio Murilo Braga`
* `praça Benfica`
* `condomínio da Merck`
* `bar do Ruan`
* `bar do Zé`
* `mercado do João`

Não anotar:

* `a estação`
* `o metrô do bairro`
* `a passarela do metrô`
* `o largo`
* `uma ponte`
* `o posto de combustível`
* `o ponto de moto táxi`

A mera capacidade de indicar uma posição espacial não transforma uma expressão genérica em entidade nomeada.

Quando o nome de uma pessoa fizer parte da denominação de um estabelecimento físico, deve-se anotar toda a expressão como `Location`, sem criar um span interno sobreposto de `Person`.

Exemplo:

* `bar do Ruan` → `Location`
* não anotar simultaneamente `Ruan` como `Person`.

Estruturas físicas individualizadas por sua associação a uma organização podem ser anotadas integralmente como `Location` quando forem usadas como lugar ou ponto de referência.

Exemplos:

* `antena da Oi` → `Location`
* `torre da Vivo` → `Location`
* `depósito da Petrobras` → `Location`, quando a referência for ao espaço físico.

Nesses casos, não se deve criar simultaneamente um span interno para a organização:

* `antena da Oi` → `Location`
* não anotar `Oi` separadamente como `Organization`.

Ver também Regra 30 para o critério de quando uma marca ou empresa deve ser anotada como `Organization` em vez de `Location`.

Complexos industriais, refinarias, centros comerciais e unidades de saúde individualizadas devem ser classificados como `Location` quando forem usados como pontos de embarque, desembarque, destino, referência espacial ou local de atendimento.

Exemplos:

* “os assaltantes descem na Reduc” → `Location`;
* “embarcam no Caxias Shopping” → `Location`;
* “foi levado ao posto de saúde de Campos Elísios” → `Location`.

Caso a entidade atue institucionalmente, usar `Organization`:

* “a Reduc divulgou uma nota” → `Organization`;
* “o Caxias Shopping alterou seu horário” → `Organization`.

### Residências individualizadas

Casas, apartamentos, sítios e outras residências podem ser anotados como Location quando forem individualizados pelo nome, apelido ou identificação inequívoca de seu ocupante.

Exemplos:

* `casa da MC Carol` → `Location`;
* `casa do Gabriel` → `Location`;
* `apartamento de Roseni` → `Location`, quando usado para identificar um lugar específico;
* `sítio do Zé` → `Location`.

Nesses casos, não criar simultaneamente um span interno para a pessoa:

* `casa da MC Carol` → `Location`;
* não anotar `MC Carol` separadamente como `Person` dentro desse span.

Não anotar referências genéricas ou anafóricas:

* `a casa`;
* `casa dela`;
* `sua residência`;
* `uma casa com piscina`.

---

## 10. Instituições com uso espacial são classificadas pelo referente

Nomes de igrejas, escolas, hospitais, empresas, laboratórios, quartéis e instituições semelhantes podem designar tanto a organização quanto seu edifício ou estabelecimento.

Quando a instituição atua como agente, classificar como `Organization`:

* “A `Assembléia de Deus` organizou o evento.”
* “A `Merck` divulgou uma nota.”
* “O `Colégio Murilo Braga` suspendeu as aulas.”

Quando o texto se refere ao edifício ou espaço físico, classificar como `Location`:

* “escondidos na `igreja evangélica Assembléia de Deus`”;
* “entre o `laboratório Merck` e o posto”;
* “em frente ao `condomínio da Merck`”;
* “na entrada do `Colégio Murilo Braga`”.

A função referencial no contexto prevalece sobre a natureza institucional abstrata.

Instituições também podem ser mencionadas por uma forma abreviada, popular ou metonímica. Quando essa expressão designar o estabelecimento físico no qual alguém está, esteve, entrou ou saiu, deve ser classificada como `Location`.

Exemplos:

* “recém-saído do `Padre Severino`” → `Location`
* “foi levado para o `Salgado Filho`” → `Location`, quando a referência for ao hospital;
* “está internado no `Getúlio Vargas`” → `Location`, quando a referência for ao hospital.

Se a mesma denominação representar a instituição como agente administrativo, usar `Organization`.

---

## 11. Evitar spans sobrepostos ou aninhados

O esquema adotado não deve manter spans sobrepostos. Quando uma entidade integra a denominação de outra entidade mais ampla, deve-se anotar apenas a expressão completa correspondente ao referente contextual.

Exemplo:

* `UPP da Cidade de Deus` → `Organization`
* não anotar simultaneamente `Cidade de Deus` como `Location` dentro do mesmo span.

Em outra ocorrência independente:

* “oriundos da `Cidade de Deus`” → `Location`

Outro exemplo:

* `Ponte do Colégio Murilo Braga` → `Location`
* não anotar simultaneamente `Colégio Murilo Braga` como entidade interna.

---

## 12. Anotar todas as ocorrências explícitas

Todas as menções de uma entidade devem ser anotadas, mesmo que a mesma entidade já tenha aparecido anteriormente no relato.

Exemplo:

* primeira ocorrência de `Rua Nina Ribeiro` → `Location`;
* segunda ocorrência de `Rua Nina Ribeiro` → `Location`;
* terceira ocorrência de `Nina Ribeiro` → `Location`.

Uma menção anterior não elimina a necessidade de anotar as ocorrências posteriores.

Referências anafóricas genéricas não são anotadas:

* `essa rua`;
* `o local`;
* `a comunidade`;
* `lá`.

Uma ocorrência posterior pode apresentar uma forma abreviada da entidade, com omissão do designador. Essa ocorrência também deve ser anotada quando o contexto tornar inequívoco que a expressão continua identificando o mesmo local.

Exemplo:

* primeira ocorrência: `rua Nonato Farias` → `Location`;
* ocorrência posterior: `Nonato Farias` → `Location`.

A omissão do designador não impede a anotação, desde que permaneça uma denominação individualizadora. Referências puramente anafóricas, como `essa rua`, `o local` ou `lá`, continuam sem anotação.

Quando o nome completo e a sigla aparecem explicitamente, ambos devem receber spans próprios.

* `Polícia Rodoviária Federal` → `Organization`;
* `PRF` → `Organization`.

Uma forma abreviada ou popular também pode ser anotada quando, naquele relato, retomar inequivocamente uma entidade individualizada anteriormente.

Exemplo:

* primeira ocorrência: `brisolao Sérgio Cardoso` → `Location`;
* ocorrência posterior: `brisolao` → `Location`, quando não houver outro brisolão possível no contexto.

Isso exige cautela. Uma ocorrência isolada de `o brisolão`, sem antecedente explícito, continuaria sem anotação. A diferença está na existência de um antecedente inequívoco no mesmo relato.

---

## 13. Nomes próprios incompletos podem ser pessoas

Nomes próprios de um único elemento devem ser anotados como `Person` quando o contexto indicar que identificam um indivíduo específico.

Exemplos:

* `Roseni`
* `Leo`
* `Tafarel`
* `Ruan`
* `Daniel`

O fato de aparecer apenas um prenome não impede a anotação.

Não anotar referências genéricas:

* `o rapaz`;
* `a vítima`;
* `a esposa`;
* `o menor`;
* `os moradores`;
* `seis traficantes`.

---

## 14. Apelidos e alcunhas são pessoas quando identificam indivíduos

Apelidos, alcunhas e nomes informais devem ser anotados como `Person` quando individualizam uma pessoa.

Exemplos:

* `Bomba`
* `Netinho`
* `Negão`
* `Vando Perereca`
* `Jiló`

Expressões introdutórias ficam fora do span:

* “vulgo `Negão`”;
* “conhecido como `Jiló`”;
* “chamado `Jordão`”;
* “tem o apelido de `Negão`”.

---

## 15. Nome civil e apelido recebem spans próprios

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

## 16. Nomes alternativos de locais recebem spans próprios

Quando um local é apresentado por nome oficial, nome antigo, apelido ou forma popular, cada denominação explicitamente presente deve ser anotada separadamente como `Location`.

Exemplo:

> antiga rua 72 do Jardim Catarina (rua Expedicionário Francisco Dias)

Anotar:

* `antiga rua 72` → `Location`
* `Jardim Catarina` → `Location`
* `rua Expedicionário Francisco Dias` → `Location`

Outro exemplo:

> comunidade Mangueirinha, conhecida também como Palha Seca

* `comunidade Mangueirinha` → `Location`
* `Palha Seca` → `Location`

A equivalência entre os nomes não elimina nenhuma das menções.

---

## 17. Unidades policiais numeradas

Expressões formadas por número cardinal, número ordinal ou ordinal escrito por extenso seguido de `BPM` identificam uma unidade policial específica e devem ser anotadas integralmente. A classe depende do referente no contexto:

* usar `Organization` quando a expressão designar a unidade policial como instituição, agente ou grupo;
* usar `Location` quando a expressão designar claramente o prédio, a sede ou o espaço físico ocupado pela unidade.

### Exemplos como `Organization`

* “policiais do **[9 BPM]Organization** foram acionados”;
* “o **[14 BPM]Organization** realizou uma operação”;
* “o comando do **[7º BPM]Organization** informou o ocorrido”.

### Exemplos como `Location`

* “o baile acontece a menos de 300 metros do **[9 BPM]Location**”;
* “o veículo foi abandonado em frente ao **[14 BPM]Location**”;
* “o suspeito foi visto próximo ao **[7º BPM]Location**”.

Também se aplica às formas expandidas:

* `14º Batalhão de Polícia Militar`;
* `Décimo Quarto Batalhão de Polícia Militar`;
* `9º Batalhão`, quando o contexto policial for inequívoco.

O número ou ordinal deve fazer parte do span. Não anotar apenas `BPM`.

Variações claramente decorrentes de erro de digitação, como `BMP`, podem ser anotadas integralmente quando o contexto tornar inequívoca a referência a um BPM específico.

Unidades policiais não numeradas podem ser individualizadas por complemento geográfico:

* “há corrupção no `DPO dá Maria Paula`” → `Organization`;
* “o carro foi abandonado em frente ao `DPO dá Maria Paula`” → `Location`.

---

## 18. Outras unidades e siglas policiais específicas são organizações

Siglas que identificam órgãos, unidades ou setores específicos devem ser anotadas como `Organization`.

Exemplos:

* `BOPE`
* `CORE`
* `GAT`
* `PMERJ`
* `P2`
* `UPP da Cidade de Deus`

Expressões descritivas, mas suficientemente individualizadas pelo contexto, também podem ser organizações:

* `batalhão de São Gonçalo`

Não anotar referências genéricas:

* `a polícia`;
* `os policiais`;
* `as autoridades`;
* `o batalhão`, sem individualização.

---

## 19. Facções, milícias e outros grupos criminosos nomeados

Nomes, siglas e denominações que individualizam facções, milícias ou outros grupos criminosos devem ser anotados como `Organization`.

Exemplos:

* `CV` → `Organization`
* `ADA` → `Organization`
* `Terceiro Comando Puro` → `Organization`
* `milícia de Santa Cruz` → `Organization`
* `Liga da Justiça` → `Organization`

Quando um designador organizacional aparece diretamente associado ao nome ou à sigla, ele deve integrar o span:

* `facção CV` → `Organization`
* `grupo Liga da Justiça` → `Organization`
* `milícia de Santa Cruz` → `Organization`

Expressões genéricas, que não individualizam um grupo específico, não devem ser anotadas:

* `a facção`
* `facção rival`
* `a milícia`
* `crime organizado`
* `organização criminosa`

Quando o nome de um local estiver contido na denominação completa da organização, deve-se anotar somente a organização completa, sem criar um span sobreposto para o local:

* `milícia de Santa Cruz` → `Organization`
* não anotar simultaneamente `Santa Cruz` como `Location`.

---

## 20. Preservar a forma original do texto

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

## 21. Não acrescentar entidades por inferência externa

Somente entidades explicitamente presentes no relato devem ser anotadas.

Se uma rua pertence a um município conhecido, o município não deve ser acrescentado caso seu nome não apareça no texto. Da mesma forma, não se deve completar:

* siglas;
* nomes incompletos;
* hierarquias geográficas;
* nomes de instituições;

com base apenas em conhecimento externo.

O contexto pode ser usado para escolher a classe, mas não para inventar uma menção ausente.

---

## 22. Números e elementos de endereço ficam fora do span

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

## 23. Veículos, placas e frases não pertencem às classes

Não devem ser anotados:

* marcas ou modelos de veículos, usados apenas para descrever o veículo em si (ver Regra 30 para o caso em que a marca identifica a própria empresa como agente, vítima ou proprietária do bem);
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

## 24. Linhas, itinerários e veículos de transporte não pertencem às classes

Nomes de linhas de ônibus, itinerários, serviços de transporte e veículos não devem ser anotados, mesmo quando são formados por nomes geográficos.

A decisão depende do referente contextual:

quando a expressão designar uma localidade ou destino geográfico, usar Location;
quando designar a linha, o itinerário, o ônibus ou outro serviço de transporte, não anotar.

Exemplo:

As linhas são Saracuruna e Piabetá da Viação União.

Anotação:

Saracuruna → Location, quando apresentado como destino da linha;
Piabetá → Location, quando apresentado como destino da linha;
Viação União → Organization.

Exemplo:

O Saracuruna foi assaltado duas vezes.

Nesse contexto, Saracuruna designa a linha ou o ônibus, e não a localidade. Portanto, não deve ser anotado.

Outros exemplos não anotáveis:

o ônibus Saracuruna;
a linha Petrópolis;
o Piabetá das seis horas;
o coletivo Magé;
o trem Japeri, quando a referência for ao serviço ou composição.

A simples coincidência lexical com um nome de lugar não transforma a menção em Location.

---

## 25. Estabelecimentos genéricos não são entidades

Expressões que apenas descrevem um estabelecimento, sem nome individualizador, não devem ser anotadas.

Exemplos:

* `uma igreja`
* `um bar`
* `a barraca de frutas`
* `o posto de combustível`
* `o laboratório`, sem nome
* `o colégio`, sem nome
* `o condomínio`, sem nome

Quando houver denominação própria, aplicar a regra contextual:

* instituição como agente → `Organization`;
* edifício ou ponto físico → `Location`.

---

## 26. Casos ambíguos devem ser resolvidos pelo contexto ou reservados para revisão

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

---

## 27. Consistência global sem ignorar o contexto

A mesma entidade deve receber a mesma classe em ocorrências semanticamente equivalentes.

Exemplos:

* `ABRAXAS`, quando designa a produtora → `Organization`;
* `Vila 3`, quando designa a localidade → `Location`;
* `Queimados`, quando designa o município → `Location`.

Entretanto, consistência lexical não significa atribuir sempre a mesma classe a qualquer homônimo. O referente contextual continua sendo decisivo.

---

## 28. Números e códigos que identificam rodovias

Números ou códigos devem ser anotados como `Location` quando, no contexto, funcionarem como denominação suficientemente individualizadora de uma rodovia ou via específica.

Exemplos:

* `BR-116`
* `Rodovia BR-116`
* `116`, quando o contexto indicar claramente a rodovia BR-116
* `040`, quando o contexto indicar claramente a rodovia BR-040

Exemplo:

> passando agora pela **[116]Location** e daqui a pouco estará na **[040]Location**

Deve-se preservar exatamente a forma presente no texto. Não acrescentar elementos que não aparecem no relato. Portanto, se o texto contém apenas `116`, o span deve ser `116`, e não `BR-116`.

Números sem função denominativa ou sem contexto suficiente para identificar uma via específica não devem ser anotados.

---

## 29. Pontos de atividade criminosa individualizados

Expressões como `boca`, `boca de fumo`, `ponto de venda`, `ponto do tráfico` e semelhantes podem ser anotadas como `Location` quando identificarem um ponto físico específico por meio de nome próprio, complemento geográfico ou outra denominação individualizadora.

Exemplos:

* `boca do campo novo` → `Location`;
* `boca da Pedreira` → `Location`;
* `ponto do tráfico da Rua Nonato Farias` → `Location`, quando toda a expressão funcionar como denominação de um ponto específico.

Não anotar quando a expressão for genérica ou apenas anafórica:

* `a boca está aberta`;
* `fecham na boca`;
* `uma boca de fumo`;
* `o ponto de venda`;
* `o tráfico instalou uma boca`, sem individualização.

Quando um local integrar a denominação completa do ponto, anotar somente a expressão integral, sem span interno sobreposto:

* `boca do campo novo` → `Location`;
* não anotar simultaneamente `campo novo` dentro desse span.

Em ocorrências independentes, `campo novo` continua sendo anotado como `Location`.

---

## 30. Marcas, produtos, plataformas e veículos de empresas

Nomes de marcas, empresas e instituições devem ser anotados como `Organization` quando designarem a própria entidade organizacional no contexto, inclusive quando ela atuar como agente, paciente, responsável, proprietária, fonte, destinatária ou vínculo institucional/de pertencimento. Não devem ser anotados quando servirem apenas para identificar a marca de um produto, veículo, aparelho, mercadoria ou serviço, nem quando designarem aplicativos, plataformas ou meios utilizados para realizar uma ação. Nomes presentes apenas em URLs também não devem ser anotados.

Exemplos:

* `A BMW anunciou um novo modelo` → `Organization`: empresa agente.
* `O governo multou a BMW` → `Organization`: empresa afetada, embora não seja agente.
* `Funcionários da BMW entraram em greve` → `Organization`: vínculo institucional.
* `Traficantes do CV foram presos` → `Organization`: pertencimento à organização.
* `Anda de BMW branca` → não anotar: marca do veículo.
* `Carga de carne Friboi` → não anotar: marca do produto.
* `Enviou pelo WhatsApp` → não anotar: plataforma usada como meio.

Um teste prático: no contexto, a expressão permite identificar uma organização participante ou relacionada ao evento, ou apenas responde "de que marca/tipo é este objeto ou meio"?

Assim:

* "traficantes do CV" → identifica pertencimento organizacional; anotar;
* "carne Friboi" → responde apenas qual é a marca da carne; não anotar;
* "carga pertencente à Friboi" → identifica a empresa proprietária; anotar;
* "carga de carne Friboi" → marca do produto; não anotar.

### Relação com a Regra 9 (estruturas físicas associadas a uma organização)

Quando a menção da marca ou empresa designa o edifício, a loja, a agência, o depósito ou outro espaço físico associado a ela, usado como local do fato, ponto de referência, origem ou destino, a classe correta é `Location`, conforme a Regra 9 — **não** `Organization`. O critério de vínculo/pertencimento desta regra não se sobrepõe ao uso espacial: a presença de uma preposição de posse (`da`, `do`) não torna a menção automaticamente uma `Organization`.

Exemplos:

* "assaltaram a loja da BMW" → `Location` (Regra 9: espaço físico usado como local do fato);
* "escondido na concessionária Friboi" → `Location` (Regra 9);
* "a BMW multou o revendedor" → `Organization` (Regra 30: empresa como agente).

### Veículos identificados pela empresa proprietária

Quando um veículo é mencionado apenas por marca ou modelo, sem que a empresa proprietária seja identificada como vítima, agente ou parte lesada do fato, a marca não deve ser anotada (Regra 23). Quando o relato identificar explicitamente a empresa como proprietária ou vítima do veículo — e não apenas como fabricante —, a empresa deve ser anotada como `Organization`.

Exemplos:

* "roubaram uma van Sprinter da empresa" → não anotar `Sprinter`: marca sem organização nomeada;
* "roubaram a van da Friboi" → `Organization`: identifica a empresa proprietária/vítima do roubo;
* "fugiram num Gol prata" → não anotar: marca do veículo (Regra 23).

### Contas e perfis em plataformas

Quando a menção a uma plataforma ou aplicativo designa uma conta, perfil ou canal individualizado que é alvo, vítima ou objeto da ação (por exemplo, invadido, clonado ou usado para criar um perfil falso), a plataforma continua não sendo anotada como `Organization`: ela permanece o meio ou suporte da ação, não uma organização agente. Exceção: quando o relato atribuir a ação à própria empresa como agente institucional.

Exemplos:

* "clonaram o WhatsApp da vítima" → não anotar `WhatsApp`;
* "criou um perfil falso no Instagram" → não anotar `Instagram`;
* "o Instagram removeu o perfil" → `Organization` (empresa como agente).

---

## Síntese operacional

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
