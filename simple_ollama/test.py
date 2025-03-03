# 올라마 모델 로드 및 테스트
from langchain.chat_models import ChatOllama

model = ChatOllama(model="gemma2:2b", temperature = 0)
# 비교적 안정적인 모델 온도일수록 0에 가까움

response = model.invoke("대한민국의 수도는 어디입니까?")

print(response)
