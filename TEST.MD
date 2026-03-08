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

---

## 3. 크롭 이미지(Crops) 마크다운 병합 실행

앞서 파서를 통해 텍스트를 추출하고 표(Table)나 기타 객체를 잘라냈다면(Crop), 추가적인 내부 스크립트를 통해 잘라낸 이미지를 각자 인식(OCR)하여 하나의 최종 마크다운 파일로 통합할 수 있습니다. 
이 단계 역시 vLLM 서버가 실행 중인 상태에서 진행해야 합니다. 터미널에 아래 명령어를 실행하세요.

```bash
python tools/merge_crops_to_md.py ./crops \
  --merged-md ./output/merged.md \
  --tmp-output-dir ./output_crop_ocr \
  --prompt-mode prompt_layout_all_en \
  --table-preprocess upscale_contrast \
  --enhance-lines \
  --stage2-max-completion-tokens 4096 \
  --protocol http \
  --ip 127.0.0.1 \
  --port 8000 \
  --model-name model \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-completion-tokens 4096
```

### 🔹 병합(Merge) 옵션 상세 설명

**[ 경로 및 파일 설정 ]**
* `./crops`: 앞선 단계에서 `--save_crop_images` 옵션을 통해 저장된 크롭 이미지들이 모여 있는 디렉토리 경로를 적습니다.
* `--merged-md ./output/merged.md`: 크롭 이미지들의 개별 인식 결과물까지 모두 반영된, 최종 병합 마크다운(Markdown) 파일이 저장될 경로입니다.
* `--tmp-output-dir ./output_crop_ocr`: 크롭 이미지 각각의 텍스트 변환 과정 등 중간(임시) 결과물들이 저장될 디렉토리입니다.

**[ 서버 연결 설정 ]**
* `--protocol http`: API 요청에 사용할 통신 프로토콜을 지정합니다. (기본적으로 `http`를 사용)
* `--ip 127.0.0.1` / `--port 8000`: 요청을 받아줄 vLLM 서버의 로컬 IP 주소와 통신 포트 번호입니다.
* `--model-name model`: 요청에 사용할 모델의 식별자입니다. vLLM 서버의 `--served-model-name`과 맞춰야 합니다.

**[ 추론 속성 설정 ]**
* `--prompt-mode prompt_layout_all_en`: 크롭 이미지를 어떤 프롬프트 모드로 파싱할지 지정합니다. 병합셀 보존이 중요하면 `prompt_layout_all_en`을 권장합니다(`prompt_ocr`는 구조 정보가 약함).
* `--table-preprocess`: 2단계 재추론 전에 크롭 이미지를 보정합니다. (`none`, `upscale`, `contrast`, `binarize`, `upscale_contrast`)
* `--enhance-lines`: 표 선을 약하게 강조해 2단계 재추론 정확도를 높입니다.
* `--stage2-max-completion-tokens`: 2단계(테이블 전용 강제 프롬프트) 최대 출력 토큰입니다.
* `--temperature 0.0`: 무작위성을 제어합니다. 정확하게 문서 내용을 추출해야 하므로 일관된 결과를 얻으려면 `0.0`을 유지합니다.
* `--top-p 1.0`: 다음 단어 선택 시 확률 분포 컷오프(1.0은 모든 단어 고려 대상).
* `--max-completion-tokens 4096`: 크롭된 이미지를 글로 풀어낼 때 허용되는 최대 글 길이(토큰) 제한입니다.

### 🔹 병합셀 보존 권장값
* 크롭 생성 시: `--crop_categories Table --crop_pad 16~32`
* PDF 렌더링 시: `--dpi 300` (작은 글자가 많으면 400까지)
