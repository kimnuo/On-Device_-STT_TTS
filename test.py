import os
import pyaudio
import numpy as np
import wave
import keyboard
from pydub import AudioSegment
from faster_whisper import WhisperModel

# 오디오 설정
CHUNK = 1024  # 오디오 청크 크기
FORMAT = pyaudio.paInt16  # 오디오 포맷
CHANNELS = 1  # 오디오 채널 수
RATE = 16000  # 샘플링 레이트
THRESHOLD = 500  # 음성 인식 임계값

INIT_SILENCE_DURATION_MS = 5000  # 최초 음성 감지 전 무음 시간 (5초 = 5000ms)
POST_VOICE_SILENCE_DURATION_MS = 2000  # 음성 감지 후 무음 시간 (2초 = 2000ms)

model_size = "large-v3"

# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# PyAudio 객체 생성
p = pyaudio.PyAudio()
# 스트림 생성
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
silent_chunks = 0  # 무음 구간을 측정할 변수
voice_detected = False  # 음성이 감지되었는지 여부
init_silent_chunks = int((INIT_SILENCE_DURATION_MS / 1000) * RATE / CHUNK)  # 최초 무음 구간 청크 수 계산
post_voice_silent_chunks = int((POST_VOICE_SILENCE_DURATION_MS / 1000) * RATE / CHUNK)  # 음성 감지 후 무음 구간 청크 수 계산
frames = []  # 녹음된 데이터를 저장할 리스트
print("음성인식중입니다... (키보드를 누르면 종료)")
try:
    while True:
        if keyboard.is_pressed('q'):  # 'q' 키를 누르면 녹음 중단
            print("녹음 중단됨 (키보드 입력)")
            break
        data = stream.read(CHUNK)
        frames.append(data)
        
        # 무음 감지를 위한 신호 에너지 계산
        audio_data = np.frombuffer(data, dtype=np.int16)  # 데이터를 numpy 배열로 변환
        signal_energy = np.abs(audio_data).mean()
        if signal_energy < THRESHOLD: 
            silent_chunks += 1 # 음성 감지 안됨
        else:
            silent_chunks = 0  # 소리가 있으면 무음 구간 초기화
            voice_detected = True  # 음성이 감지됨
        # 최초 음성 감지 전: 5초 동안 무음이 지속되면 종료
        if not voice_detected and silent_chunks >= init_silent_chunks:
            print("녹음 중단됨")
            break
        # 음성 감지 후: 1초 동안 무음이 지속되면 종료
        if voice_detected and silent_chunks >= post_voice_silent_chunks:
            print("녹음 중단됨")
            break
except KeyboardInterrupt:
    print("녹음 중단됨")
finally:
    # 녹음 종료 및 스트림 닫기
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # output 디렉토리 생성
    if not os.path.exists("output"):
        os.makedirs("output")
    
    output_filename = "output/audio.wav"
    # 녹음된 데이터를 WAV 파일로 저장
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    # WAV 파일을 MP3 파일로 변환
    sound = AudioSegment.from_wav(output_filename)
    mp3_filename = "output/audio.mp3"
    sound.export(mp3_filename, format="mp3")
    segments, info = model.transcribe(mp3_filename, beam_size=5)
    print("인식된 언어 '%s' 일치확률 %f" % (info.language, info.language_probability))
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))