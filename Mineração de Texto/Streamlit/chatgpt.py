import streamlit as st
import openai
import re

# Configure a chave da API do OpenAI
openai.api_key = 'YOUR-TOKEN'

# Título do aplicativo
st.title("ChatGPT com Identificação Simples de Entidades usando chat.completions.create")

# Caixa de texto para a entrada do usuário
user_input = st.text_input("Digite sua mensagem:")

# Botão para enviar a mensagem
if st.button("Enviar"):
    # Chame a API do OpenAI para obter a resposta do ChatGPT usando chat.completions.create
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_input},
        ]
    )

    # Exiba a resposta do ChatGPT
    chat_response = response['choices'][0]['message']['content']
    st.text("Resposta do ChatGPT: {}".format(chat_response))

    # Use expressões regulares para identificar entidades simples (por exemplo, números e e-mails)
    numbers = re.findall(r'\b\d+\b', user_input)
    emails = re.findall(r'\S+@\S+', user_input)

    # Exiba as entidades identificadas
    if numbers:
        st.text("Números identificados:")
        for number in numbers:
            st.text(number)

    if emails:
        st.text("E-mails identificados:")
        for email in emails:
            st.text(email)
