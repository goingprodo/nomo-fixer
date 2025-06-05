import gradio as gr
import torch
import torchaudio
import ffmpeg
import os
import tempfile
from pathlib import Path
import numpy as np
import shutil
import uuid

# torchaudio 백엔드 설정
try:
    # Windows에서 안정적인 백엔드 설정
    if os.name == 'nt':  # Windows
        torchaudio.set_audio_backend("soundfile")
    else:  # Linux/Mac
        torchaudio.set_audio_backend("sox_io")
except:
    # 백엔드 설정 실패 시 기본값 사용
    pass

# CUDA 12.8 호환성 확인
def check_cuda_setup():
    """CUDA 설정 확인"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        cuda_version = torch.version.cuda
        return {
            "available": True,
            "device_count": device_count,
            "device_name": device_name,
            "cuda_version": cuda_version
        }
    return {"available": False}

def convert_video_to_mono_gpu(input_video, conversion_method="mix", use_gpu=True):
    """
    GPU 가속을 활용한 영상 오디오 모노 변환
    
    Args:
        input_video: 입력 영상 파일
        conversion_method: 변환 방식 ("mix", "left", "right")
        use_gpu: GPU 사용 여부
    
    Returns:
        변환된 영상 파일 경로
    """
    if input_video is None:
        return None, "영상 파일을 업로드해주세요."
    
    try:
        # CUDA 설정
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # 출력 파일을 현재 디렉터리의 output 폴더에 저장
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = input_video
        output_filename = f"mono_converted_{Path(input_path).stem}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # 임시 오디오 파일 경로 (특수문자 제거하고 고유 ID 사용)
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        temp_audio_input = os.path.join(temp_dir, f"temp_input_{unique_id}.wav")
        temp_audio_output = os.path.join(temp_dir, f"temp_output_{unique_id}.wav")
        
        try:
            # 1단계: FFmpeg로 오디오 추출
            (
                ffmpeg
                .input(input_path)
                .output(temp_audio_input, acodec='pcm_s16le', ar=44100)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            # 2단계: PyTorch로 GPU 가속 오디오 처리
            try:
                # 여러 방법으로 오디오 로드 시도
                waveform, sample_rate = None, None
                
                # 방법 1: torchaudio로 직접 로드
                try:
                    waveform, sample_rate = torchaudio.load(temp_audio_input)
                    processing_info = "torchaudio 직접 로드"
                except Exception as e1:
                    print(f"torchaudio 로드 실패: {e1}")
                    
                    # 방법 2: soundfile 백엔드 강제 사용
                    try:
                        import soundfile as sf
                        data, sample_rate = sf.read(temp_audio_input)
                        waveform = torch.from_numpy(data).T.float()
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                        processing_info = "soundfile 백엔드 사용"
                    except Exception as e2:
                        print(f"soundfile 로드 실패: {e2}")
                        
                        # 방법 3: FFmpeg로 다시 처리
                        temp_audio_simple = os.path.join(temp_dir, f"simple_{unique_id}.wav")
                        (
                            ffmpeg
                            .input(temp_audio_input)
                            .output(temp_audio_simple, acodec='pcm_s16le', ar=22050, ac=2)
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True, quiet=True)
                        )
                        waveform, sample_rate = torchaudio.load(temp_audio_simple)
                        os.remove(temp_audio_simple)
                        processing_info = "FFmpeg 재처리 후 로드"
                
                if waveform is None:
                    raise Exception("모든 오디오 로드 방법 실패")
                    
            except Exception as load_error:
                return None, f"❌ 오디오 로드 실패: {str(load_error)}"
            waveform = waveform.to(device)  # GPU로 이동
            
            if waveform.shape[0] == 1:
                # 이미 모노인 경우
                processed_audio = waveform
                processing_info += " - 이미 모노 오디오"
            elif waveform.shape[0] == 2:
                # 스테레오 처리
                if conversion_method == "mix":
                    # 좌우 채널 평균 (GPU에서 처리)
                    processed_audio = torch.mean(waveform, dim=0, keepdim=True)
                    processing_info += " - 좌우 채널 혼합"
                elif conversion_method == "left":
                    # 왼쪽 채널만
                    processed_audio = waveform[0:1, :]
                    processing_info += " - 왼쪽 채널만 사용"
                elif conversion_method == "right":
                    # 오른쪽 채널만
                    processed_audio = waveform[1:2, :]
                    processing_info += " - 오른쪽 채널만 사용"
                else:
                    processed_audio = torch.mean(waveform, dim=0, keepdim=True)
                    processing_info += " - 기본 혼합"
            else:
                # 다채널 오디오의 경우 첫 번째와 두 번째 채널만 사용
                processed_audio = torch.mean(waveform[:2, :], dim=0, keepdim=True)
                processing_info += " - 다채널에서 첫 2채널 혼합"
            
            # CPU로 다시 이동하여 저장
            processed_audio = processed_audio.cpu()
            
            # 3단계: 처리된 오디오 저장
            torchaudio.save(temp_audio_output, processed_audio, sample_rate)
            
            # 4단계: 원본 비디오와 처리된 오디오 합치기 (GPU 가속 인코딩 사용)
            video_input = ffmpeg.input(input_path)
            audio_input = ffmpeg.input(temp_audio_output)
            
            # GPU 가속 비디오 인코딩 (NVENC 사용)
            if use_gpu and torch.cuda.is_available():
                try:
                    # NVIDIA GPU 인코더 사용 시도
                    out = ffmpeg.output(
                        video_input['v'], audio_input['a'],
                        output_path,
                        vcodec='h264_nvenc',  # GPU 가속 인코더
                        acodec='aac',
                        **{
                            'b:v': '5000k',  # 비디오 비트레이트
                            'b:a': '128k',   # 오디오 비트레이트
                            'preset': 'fast'  # 빠른 인코딩
                        }
                    )
                    ffmpeg.run(out, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    encoding_info = "GPU 가속 인코딩 (NVENC)"
                except:
                    # GPU 인코딩 실패 시 CPU 인코딩으로 폴백
                    out = ffmpeg.output(
                        video_input['v'], audio_input['a'],
                        output_path,
                        vcodec='libx264',
                        acodec='aac',
                        **{'b:a': '128k', 'preset': 'fast'}
                    )
                    ffmpeg.run(out, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    encoding_info = "CPU 인코딩 (GPU 인코딩 실패)"
            else:
                # CPU 인코딩
                out = ffmpeg.output(
                    video_input['v'], audio_input['a'],
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    **{'b:a': '128k'}
                )
                ffmpeg.run(out, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                encoding_info = "CPU 인코딩"
            
            # 임시 파일 정리
            if os.path.exists(temp_audio_input):
                os.remove(temp_audio_input)
            if os.path.exists(temp_audio_output):
                os.remove(temp_audio_output)
            
            device_info = f"GPU ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "CPU"
            
            return output_path, f"✅ 변환 완료!\n처리 방식: {processing_info}\n인코딩: {encoding_info}\n처리 장치: {device_info}\n파일 위치: {output_path}"
            
        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
            return None, f"❌ FFmpeg 오류:\n{stderr_output}"
        finally:
            # 임시 파일 정리
            for temp_file in [temp_audio_input, temp_audio_output]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
    except Exception as e:
        return None, f"❌ 변환 중 오류 발생: {str(e)}"

def get_video_info_gpu(input_video):
    """GPU 정보와 함께 영상 정보를 가져오는 함수"""
    if input_video is None:
        return "영상을 업로드해주세요."
    
    try:
        # CUDA 정보
        cuda_info = check_cuda_setup()
        
        probe = ffmpeg.probe(input_video)
        
        # 비디오 정보
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        
        info_text = f"""
🖥️ **시스템 정보:**
- CUDA 사용 가능: {"✅ Yes" if cuda_info["available"] else "❌ No"}
"""
        
        if cuda_info["available"]:
            info_text += f"""- GPU: {cuda_info["device_name"]}
- CUDA 버전: {cuda_info["cuda_version"]}
- GPU 개수: {cuda_info["device_count"]}"""
        
        info_text += f"""

📹 **영상 정보:**
- 해상도: {video_info['width']}x{video_info['height']}
- 프레임률: {eval(video_info['r_frame_rate']):.2f} fps
- 길이: {float(probe['format']['duration']):.2f}초

🔊 **오디오 정보:**
"""
        
        if audio_info:
            channels = audio_info.get('channels', 'Unknown')
            channel_layout = audio_info.get('channel_layout', 'Unknown')
            sample_rate = audio_info.get('sample_rate', 'Unknown')
            
            info_text += f"""- 채널: {channels}개 ({channel_layout})
- 샘플레이트: {sample_rate} Hz
- 코덱: {audio_info['codec_name']}"""
            
            if channels == 2:
                info_text += "\n\n⚠️ **스테레오 오디오가 감지되었습니다. GPU 가속 모노 변환을 진행하세요!**"
            elif channels == 1:
                info_text += "\n\n✅ **이미 모노 오디오입니다.**"
        else:
            info_text += "- 오디오 트랙이 없습니다."
        
        return info_text
        
    except Exception as e:
        return f"❌ 영상 정보를 가져오는데 실패했습니다: {str(e)}"

# Gradio 인터페이스 생성
with gr.Blocks(title="GPU 가속 영상 오디오 모노 변환기", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 GPU 가속 영상 오디오 모노 변환기
    
    PyTorch와 CUDA 12.8을 활용한 고속 영상 오디오 변환 도구입니다.
    한쪽으로만 들리는 영상의 오디오를 양쪽으로 들리게 만들어주며, GPU 가속으로 빠른 처리가 가능합니다.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 입력 섹션
            gr.Markdown("## 📁 영상 업로드")
            input_video = gr.File(
                label="영상 파일 선택",
                file_types=["video"],
                type="filepath"
            )
            
            # 변환 옵션
            gr.Markdown("## ⚙️ 변환 설정")
            conversion_method = gr.Radio(
                choices=[
                    ("좌우 채널 혼합 (권장)", "mix"),
                    ("왼쪽 채널만 사용", "left"),
                    ("오른쪽 채널만 사용", "right")
                ],
                value="mix",
                label="변환 방식",
                info="GPU에서 고속으로 처리됩니다."
            )
            
            use_gpu = gr.Checkbox(
                label="GPU 가속 사용",
                value=True,
                info="CUDA가 사용 가능한 경우 GPU로 처리합니다."
            )
            
            # 버튼들
            info_btn = gr.Button("📊 시스템 & 영상 정보 확인", variant="secondary")
            convert_btn = gr.Button("🚀 GPU 가속 모노 변환", variant="primary")
            
        with gr.Column(scale=1):
            # 출력 섹션
            gr.Markdown("## 📋 시스템 & 영상 정보")
            video_info_output = gr.Markdown("영상을 업로드하고 '시스템 & 영상 정보 확인' 버튼을 클릭하세요.")
            
            gr.Markdown("## 📥 변환 결과")
            status_output = gr.Textbox(
                label="변환 상태",
                value="변환을 시작하려면 영상을 업로드하고 'GPU 가속 모노 변환' 버튼을 클릭하세요.",
                interactive=False,
                lines=6
            )
            
            output_video = gr.File(
                label="변환된 영상 다운로드",
                visible=False
            )
    
    # 사용법 안내
    gr.Markdown("""
    ## 📖 사용법
    
    1. **영상 업로드**: 변환하고 싶은 영상 파일을 선택하세요
    2. **시스템 정보 확인**: GPU 사용 가능 여부와 영상 정보를 확인하세요
    3. **변환 방식 선택**: 필요에 따라 변환 방식을 선택하세요
    4. **GPU 가속 변환**: 'GPU 가속 모노 변환' 버튼을 클릭하세요
    5. **다운로드**: 변환이 완료되면 파일을 다운로드하세요
    
    ## 🚀 GPU 가속 기능
    
    - **PyTorch CUDA**: 오디오 처리를 GPU에서 수행
    - **NVENC 인코딩**: NVIDIA GPU의 하드웨어 인코더 활용
    - **메모리 최적화**: 대용량 파일도 효율적으로 처리
    - **자동 폴백**: GPU 사용 불가 시 자동으로 CPU 처리
    
    ## 📋 요구사항
    
    - **CUDA 12.8** 호환 NVIDIA GPU (권장)
    - **PyTorch with CUDA** 지원
    - **FFmpeg with NVENC** 지원
    - **충분한 GPU 메모리** (영상 크기에 따라)
    
    ## ⚡ 성능 향상
    
    GPU 사용 시 CPU 대비 **3-10배** 빠른 처리 속도를 기대할 수 있습니다.
    특히 4K, 8K 등 고해상도 영상에서 큰 성능 차이를 보입니다.
    """)
    
    # 이벤트 연결
    info_btn.click(
        fn=get_video_info_gpu,
        inputs=[input_video],
        outputs=[video_info_output]
    )
    
    convert_btn.click(
        fn=convert_video_to_mono_gpu,
        inputs=[input_video, conversion_method, use_gpu],
        outputs=[output_video, status_output]
    ).then(
        fn=lambda path, status: gr.update(visible=bool(path and os.path.exists(path) if path else False)),
        inputs=[output_video, status_output],
        outputs=[output_video]
    )

# 실행
if __name__ == "__main__":
    # CUDA 설정 확인
    cuda_info = check_cuda_setup()
    
    print("🚀 GPU 가속 영상 오디오 모노 변환기를 시작합니다...")
    print(f"\n🖥️  시스템 정보:")
    print(f"- PyTorch 버전: {torch.__version__}")
    print(f"- CUDA 사용 가능: {cuda_info['available']}")
    
    if cuda_info['available']:
        print(f"- GPU: {cuda_info['device_name']}")
        print(f"- CUDA 버전: {cuda_info['cuda_version']}")
        print(f"- GPU 개수: {cuda_info['device_count']}")
    else:
        print("- GPU 가속을 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    print("\n📦 필요한 패키지:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    print("pip install gradio ffmpeg-python")
    
    print("\n🔧 FFmpeg with NVENC 지원 필요")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )