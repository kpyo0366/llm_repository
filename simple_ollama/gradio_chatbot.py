import gradio as gr
from langchain_ollama import ChatOllama  # 최신 import 방식

## 모델 임포트
model = ChatOllama(model="gemma2:2b", temperature=0)

## echo 함수 정의 
def echo(message, history):
    response = model.invoke(message)
    return response.content  # 가비지 데이터를 제외하고 content만 반환

## 정의한 리턴값을 인터페이스를 통해 담기
demo = gr.ChatInterface(fn=echo, examples=["hello", "holla", "안녕"], title="Echo Bot")

# 🚨 함수 실행을 위해 () 추가
demo.launch()
