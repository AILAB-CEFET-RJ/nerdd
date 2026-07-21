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

Não anotar:

* `a estação`
* `o metrô do bairro`
* `a passarela do metrô`
* `o largo`
* `uma ponte`
* `o posto de combustível`
* `o ponto de moto táxi`

A mera capacidade de indicar uma posição espacial não transforma uma expressão genérica em entidade nomeada.

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

## 17. Unidades policiais numeradas são organizações

Expressões formadas por número cardinal, número ordinal ou ordinal escrito por extenso seguido de `BPM` devem ser anotadas integralmente como `Organization`.

Exemplos:

* `14 BPM`
* `9 BPM`
* `9º BPM`
* `9° BPM`
* `sétimo BPM`
* `7º BPM`
* `18o BPM`

Também se aplica às formas expandidas:

* `14º Batalhão de Polícia Militar`
* `Décimo Quarto Batalhão de Polícia Militar`
* `9º Batalhão`, quando o contexto policial for inequívoco.

O número ou ordinal deve fazer parte do span. Não anotar apenas `BPM`.

Variações claramente decorrentes de erro de digitação, como `BMP`, podem ser anotadas integralmente quando o contexto tornar inequívoca a referência a um BPM específico.

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

## 19. Preservar a forma original do texto

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

## 20. Não acrescentar entidades por inferência externa

Somente entidades explicitamente presentes no relato devem ser anotadas.

Se uma rua pertence a um município conhecido, o município não deve ser acrescentado caso seu nome não apareça no texto. Da mesma forma, não se deve completar:

* siglas;
* nomes incompletos;
* hierarquias geográficas;
* nomes de instituições;

com base apenas em conhecimento externo.

O contexto pode ser usado para escolher a classe, mas não para inventar uma menção ausente.

---

## 21. Números e elementos de endereço ficam fora do span

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

## 22. Veículos, placas e frases não pertencem às classes

Não devem ser anotados:

* marcas ou modelos de veículos, salvo mudança futura do esquema;
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
* “Não foi sorte, foi Deus”

---

## 23. Estabelecimentos genéricos não são entidades

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

## 24. Casos ambíguos devem ser resolvidos pelo contexto ou reservados para revisão

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

## 25. Consistência global sem ignorar o contexto

A mesma entidade deve receber a mesma classe em ocorrências semanticamente equivalentes.

Exemplos:

* `ABRAXAS`, quando designa a produtora → `Organization`;
* `Vila 3`, quando designa a localidade → `Location`;
* `Queimados`, quando designa o município → `Location`.

Entretanto, consistência lexical não significa atribuir sempre a mesma classe a qualquer homônimo. O referente contextual continua sendo decisivo.

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
