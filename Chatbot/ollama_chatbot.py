import streamlit as st
import ollama

st.title("ChatGPT-like Clone with Ollama")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Ollama 모델 호출
    response = ollama.chat(model="gemma2:2b", messages=st.session_state.messages)
    reply = response["message"]["content"]
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    with st.chat_message("assistant"):
        st.markdown(reply)
