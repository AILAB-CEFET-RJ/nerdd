import streamlit as st
import spacy
from spacy import displacy
import openai
import json
import os

         

openai.api_key = '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'

# Título
st.title("Identificação de Relatos Criminais")

# Descrição
st.write("Indentificação de Locais, Organizações e Pessoas em relatos criminais do Disque Denúncia")








def load_models():
    sm_model = spacy.load("pt_core_news_sm")
    lg_model = spacy.load("pt_core_news_lg")
    testeSpacy = spacy.load("Spacy-Modelos.h5")
    chat_gpt =  ""
    models = {"sm": sm_model, "lg": lg_model,"testeSpacy": testeSpacy,"Chat GPT":chat_gpt}
    return models

models = load_models()
selected_type = st.sidebar.selectbox("Selecione o tipo do modelo", options=["sm", "lg","testeSpacy","chat_gpt"])

if selected_type == "chat_gpt":
    # Caixa de texto para a entrada do usuário
    user_input = st.text_input("Digite sua mensagem:",'Policia prende traficante em Copacabana')

    # Botão para enviar a mensagem
    if st.button("Enviar"):
        # Chame a API do OpenAI para obter a resposta do ChatGPT
       response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Identifique o bairro,organização,pessoas na frase:"  + user_input},
        ]
        )
        # Exiba a resposta do ChatGPT
    
else:
    
    # Caixa de entrada de texto
    input_text = st.text_input('Insira o texto a ser analisado:', 'Policia prende traficante em Copacabana')
    selected_model = models[selected_type]
    doc = selected_model(input_text)
    # Cores
    colors = {"PERSON": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
            "LOC": "#ffedd5",
            "ORG": "#fae8ff",
    }

    options = {"ents": ["PERSON", "LOC", "ORG"],
            "colors": colors}
    
if selected_type == "chat_gpt":
    # Exiba a resposta do ChatGPT
    chat_response = response['choices'][0]['message']['content']
    st.text("\n {}".format(response.choices[0].message.content))

else:
    # Renderizando o HTML das entidades encontradas
    ent_html = displacy.render(doc, style="ent", options=options, jupyter=False)
    # Exibindo o visualizador de entidades no Web App
    st.markdown(ent_html, unsafe_allow_html=True)