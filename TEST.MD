# Dots.ocr 실행 가이드

이 문서는 vLLM 모델 서버 실행 및 파싱(Parsing) 스크립트 실행 방법에 대한 상세한 가이드입니다. 복사 및 붙여넣기로 쉽게 실행할 수 있도록 작성되었습니다.

---

## 1. vLLM 서버 모델 실행

파서를 실행하기 전에 먼저 vLLM 서버를 띄워야 합니다. 아래 명령어를 터미널에 입력하여 서버를 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /home/eden/workspace/dots.ocr/weights/DotsOCR_1_5 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --chat-template-content-format string \
  --served-model-name model
```

### 🔹 vLLM 옵션 상세 설명
* `CUDA_VISIBLE_DEVICES=0`: 사용할 GPU 번호를 지정합니다. (0번 GPU 사용)
* `vllm serve [모델경로]`: 지정된 로컬 경로의 모델로 vLLM 서버를 시작합니다.
* `--trust-remote-code`: 허깅페이스 등의 원격 코드 실행을 허용합니다. 사용자 정의 모델 아키텍처를 로드할 때 필수적입니다.
* `--tensor-parallel-size 1`: 모델을 분산 처리할 GPU 개수입니다.
* `--gpu-memory-utilization 0.85`: 프로세스가 사용할 최대 GPU 메모리의 비율(0 ~ 1)입니다. 0.85는 GPU 메모리의 85%를 사용하겠다는 뜻입니다. OOM(Out of Memory) 에러가 날 경우 값을 낮춰줍니다.
* `--max-model-len 16384`: 모델이 처리할 수 있는 최대 컨텍스트 길이(토큰 수) 제한입니다.
* `--max-num-seqs 1`: 한 번의 추론에서 동시에 처리하는 최대 시퀀스(요청) 수입니다.
* `--chat-template-content-format string`: 채팅 템플릿의 형식을 문자열(string)로 강제 지정합니다. VLM 등 여러 모델 버전과의 호환성 문제 방지에 쓰입니다.
* `--served-model-name model`: API 요청 시 클라이언트가 이 모델을 호출할 수 있도록 지정하는 모델 별칭 이름입니다.

---

## 2. 문서 파서(Parser) 실행

vLLM 서버가 성공적으로 실행되면, 새로운 터미널을 열고 파싱(OCR/레이아웃 분석) 스크립트를 실행합니다.

```bash
python3 dots_ocr/parser.py demo/test5.pdf \
  --ip 127.0.0.1 --port 8000 --model_name model \
  --prompt prompt_layout_all_en \
  --dpi 400 \
  --temperature 0 \
  --top_p 1 \
  --num_thread 1 \
  --max_completion_tokens 15000 \
  --save_crop_images \
  --crop_categories Table \
  --include_section_header_in_table_crop \
  --section_header_max_gap 260 \
  --crop_pad 10 \
  --output ./output_vllm3
```

### 🔹 파서(Parser) 옵션 상세 설명

**[ 서버 및 렌더링 설정 ]**
* `demo/test5.pdf`: 분석(파싱)을 진행할 대상 PDF 파일 경로입니다.
* `--ip 127.0.0.1` / `--port 8000`: 요청을 보낼 vLLM 서버의 호스트 주소와 포트 번호입니다.
* `--model_name model`: vLLM에서 띄워둔 서버 이름입니다. 서버의 `--served-model-name`과 일치해야 합니다.
* `--prompt prompt_layout_all_en`: 모델에 전달할 시스템 프롬프트(명령어) 프리셋. (예: 전체 레이아웃을 영어로 추출)
* `--dpi 400`: PDF를 이미지로 렌더링할 때의 해상도 수치입니다. 값이 클수록 선명하지만 시간과 메모리를 더 소모합니다.

**[ 추론 및 성능 설정 ]**
* `--temperature 0`: 모델의 창의성 및 무작위성 정도. 0으로 설정하면 가장 확률이 높은 단어만 항상 선택하여 결과가 일관적으로 나옵니다.
* `--top_p 1`: 다음 단어를 선택할 때 누적 확률을 기준으로 토큰(단어)들을 샘플링하는 옵션입니다. 1은 샘플링 제한을 두지 않음을 의미합니다.
* `--num_thread 1`: 동시에 사용할 스레드의 개수입니다.
* `--max_completion_tokens 15000`: 출력을 생성할 최대 토큰 길이 제한. 표나 텍스트가 많은 문서를 위해 값을 넉넉히 설정합니다.

**[ 이미지 잘라내기(Crop) 상세 설정 ]**
* `--save_crop_images`: 파싱이 완료된 결과에서 특정 객체(표, 이미지 등)를 원본 문서 이미지에서 잘라내어 독립된 파일로 저장합니다.
* `--crop_categories Table`: 자르기를 수행할 카테고리를 지정합니다. 여기서는 오직 표(Table)만을 선택적으로 자릅니다.
* `--include_section_header_in_table_crop`: 표 이미지를 자를 때 표와 관련된 근처의 제목이나 섹션 헤더를 인식하여 함께 포함시켜 자릅니다.
* `--section_header_max_gap 260`: 표와 그 위/아래의 제목 헤더를 연관 지을 때 허용되는 최대 픽셀 거리(gap)입니다.
* `--crop_pad 10`: 표 주변으로 설정할 여백 영역(픽셀 단위 패딩 값)입니다.

**[ 결과 저장 옵션 ]**
* `--output ./output_vllm3`: 결과물(마크다운, JSON 메타데이터, 크롭 이미지 디렉토리 등)을 저장할 최상위 디렉토리(폴더) 위치입니다.
