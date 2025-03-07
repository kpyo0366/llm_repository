import streamlit as st
import whisper
import pandas as pd
from datetime import datetime
import subprocess
import json
import tempfile
import os

# FFmpeg 경로 설정 (필요한 경우 수동 설정)
os.environ["PATH"] += os.pathsep + "/usr/local/bin"  # Mac/Linux용 기본 FFmpeg 경로 추가

# 스트림릿 UI 설정
st.title("회의록 자동 생성기")

uploaded_file = st.file_uploader("녹음 파일을 업로드하세요 (MP3, WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    st.write("파일 분석 중...")
    
    # 파일을 임시 저장 후 Whisper에 전달
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name
    
    # Whisper 모델 불러오기 및 음성 인식
    try:
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path, word_timestamps=True, diarization=True)  # 화자 분리 활성화
        segments = result["segments"]
        speakers = result.get("speakers", {})  # Whisper가 제공하는 화자 정보
    except FileNotFoundError:
        st.error("FFmpeg가 설치되지 않았습니다. 아래 명령어로 설치하세요:\n\nMac: `brew install ffmpeg`\nUbuntu: `sudo apt install ffmpeg`\nWindows: `ffmpeg.org`에서 다운로드 후 환경 변수 추가")
        segments = []
        speakers = {}
    except AttributeError:
        st.error("Whisper 라이브러리를 다시 설치하세요: `pip install openai-whisper`")
        segments = []
        speakers = {}
    
    # 사용자 정의 이름 매핑을 위한 딕셔너리
    speaker_names = {spk: f"Speaker {spk}" for spk in set(speakers.values())}
    
    # 화자 이름 편집 기능 추가
    st.subheader("화자 이름 편집")
    for spk in speaker_names:
        new_name = st.text_input(f"{speaker_names[spk]}의 이름 변경", value=speaker_names[spk])
        speaker_names[spk] = new_name
    
    # 로그 표시 방식 선택 (디폴트: 문장 단위, 옵션: 전체 머지)
    view_mode = st.radio("로그 표시 방식 선택", ["문장 단위", "전체 머지"], index=0)
    
    st.subheader("회의 로그")
    
    if segments:
        if view_mode == "문장 단위":
            for seg in segments:
                start_time = int(seg["start"] // 60), int(seg["start"] % 60)  # 분:초 변환
                start_str = f"[{start_time[0]:02d}:{start_time[1]:02d}]"
                text = seg["text"]
                speaker = speaker_names.get(speakers.get(seg["id"], "Unknown"), "Unknown")
                
                # 해당 문장을 클릭하면 오디오를 해당 지점부터 재생하도록 구성
                if st.button(f"{start_str} {speaker}: {text}", key=f"jump_{seg['start']}"):
                    st.audio(uploaded_file, format='audio/mp3', start_time=seg["start"])
        
        elif view_mode == "전체 머지":
            full_transcript = "\n".join([
                f"[{int(seg['start']//60):02d}:{int(seg['start']%60):02d}] {speaker_names.get(speakers.get(seg['id'], 'Unknown'), 'Unknown')}: {seg['text']}" 
                for seg in segments
            ])
            st.text_area("전체 회의 로그", full_transcript, height=300)
        
        # Ollama의 gemma2:2b를 활용한 회의록 요약 (기본적으로 한글로 출력)
        def summarize_meeting(segments):
            transcript = "\n".join([f"{speaker_names.get(speakers.get(seg['id'], 'Unknown'), 'Unknown')}: {seg['text']}" for seg in segments])
            prompt = f"""
            다음은 한국어 회의록입니다. 회의의 개요(일시, 주제), Action Item, 주요 회의 내용을 한국어로 요약해주세요.
            회의록:
            {transcript}
            """
            
            response = subprocess.run([
                "ollama", "run", "gemma2:2b"],
                input=prompt,
                capture_output=True, text=True
            )
            
            return response.stdout.strip()
        
        if st.button("회의록 요약 생성"):
            summary = summarize_meeting(segments)
            st.subheader("회의록 요약")
            st.write(summary)
            
            # 다운로드 기능 추가
            st.download_button("회의록 다운로드", summary, file_name="meeting_summary.txt")
