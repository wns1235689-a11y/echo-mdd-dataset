# echo-mdd-dataset

한국인 영어 발음 교정 앱 **ECHO**(2026 졸업 캡스톤)의 오발음 탐지·진단(MDD) 모델 학습 데이터셋을 구축하는 데 쓴 변환·검증 스크립트 모음이다. 졸업논문 「이종 음소 표기 음성 데이터의 통합 정제를 통한 한국인 영어 오발음 진단 학습 데이터셋 구축 — 발음 교정 앱 ECHO 사례를 중심으로」(김형준, 한국외국어대학교, 2026)의 부록 A에 대응한다.

두 공개 데이터 **L2-ARCTIC**(ARPAbet, TextGrid)과 **AI-Hub 교육용 한국인의 영어 음성 데이터**(IPA, JSON)를 39개 ARPAbet 음소 + 묵음의 공통 음소 집합과 공통 JSON 스키마로 변환한다.

## 포함하지 않는 것

- 원본 데이터와 산출 데이터(JSON). L2-ARCTIC은 CC BY-NC 4.0, AI-Hub 데이터는 AI-Hub 이용 조건에 따라 재가공 데이터의 재배포가 제한되므로 각 기관에서 직접 내려받아야 한다.
  - L2-ARCTIC: https://psi.engr.tamu.edu/l2-arctic-corpus/
  - AI-Hub 교육용 한국인의 영어 음성 데이터: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71463
- ECHO 서비스 코드(모델 서버·백엔드·프론트엔드). 팀 저장소 https://github.com/ECHOCapstone 참조.

## 구성

```
pipeline/   데이터셋 구축 스크립트 (프로젝트 기간, 2026. 3~4.)
analysis/   최종 데이터·원본 라벨 전수 검증 스크립트 (논문 집필, 2026. 10.)
```

### pipeline/

| 스크립트 | 역할 | 입력 → 출력 |
|---|---|---|
| `parse_l2arctic.py` | L2-ARCTIC TextGrid 해석, 비표준 표기 정규화(r\*→r, ax→ah, sp/spn→sil 등), CTC 학습 목표 생성 | 화자 폴더 → 화자별 JSON + `l2arctic_all.json` |
| `postprocess_l2arctic.py` | err→er 정규화, 공통 음소 집합 밖 기호를 가진 발화 제외 | `l2arctic_all.json` → `l2arctic_clean.json` |
| `parse_aihub.py` | PRN 문항 선별, 이중모음 병합(a ɪ→ay 등), IPA→ARPAbet 변환, `***` 포함 발화 제외, 정발음 g2p 재생성 | PRN 라벨 폴더 → `aihub_train.json`, `aihub_val.json` |
| `compare_aihub.py` | 서버 정제본 표본을 원본 라벨에서 변환한 결과와 대조 | 표본 JSON + PRN 라벨 폴더 → 불일치 목록 |

### analysis/

| 스크립트 | 역할 |
|---|---|
| `dataset_stats.py` | 최종 학습 데이터(train/val/test.json)의 규모·발화 특성·오류 분포·상위 대치 쌍 |
| `check_aihub_labels_local.py` | AI-Hub 원본 라벨의 세 필드(transAnsw·phonemeText·tagging) 길이 일치와 tagging 도출 규칙 전수 검증, 문단 문항 분포 |
| `check_v1v2_local.py` | 서버 정제본 v1·v2의 구성 규칙 판별(문장 문항·길이 조건) |
| `check_v2_vs_source.py` | v2 전체를 원본 라벨과 대조하여 변환 규칙 역추적, 원본 tagging 기준 오류 분포 |
| `aihub_error_pairs.py` | 원본 tagging 기준 상위 대치·삭제·삽입 집계 |
| `check_alignment.py` | 최종 학습 데이터의 두 정렬 필드 길이 일치, 학습 목표 생성 규칙, 학습 목표에 남은 묵음의 출처 |

## 실행

Python 3. 추가 라이브러리는 `parse_aihub.py`의 g2p_en과 `dataset_stats.py`·`check_alignment.py`의 rapidfuzz뿐이다.

```bash
pip install g2p_en rapidfuzz
```

`parse_aihub.py`, `postprocess_l2arctic.py`, `compare_aihub.py`는 파일 상단 또는 `main()`의 경로 상수(`<path to ...>`)를 환경에 맞게 바꾼 뒤 실행한다. 나머지는 인자로 경로를 받는다.

```bash
# L2-ARCTIC
python pipeline/parse_l2arctic.py --data_dir <L2-ARCTIC 루트> --output_dir <출력 폴더>
python pipeline/postprocess_l2arctic.py

# AI-Hub
python pipeline/parse_aihub.py

# 최종 데이터 통계 (train.json, val.json, test.json이 있는 폴더)
python analysis/dataset_stats.py <폴더>

# 원본 라벨 전수 검증
python analysis/check_aihub_labels_local.py <prn_label_train> <aihub_train.json>
python analysis/check_v1v2_local.py <aihub_clean_v1.json> <aihub_clean_v2.json>
python analysis/check_v2_vs_source.py <prn_label_train> <aihub_clean_v2.json> <aihub_train.json> <aihub_val.json>
python analysis/aihub_error_pairs.py <prn_label_train> <aihub_clean_v2.json>

# 정렬 필드·학습 목표 점검 (train.json, val.json, test.json이 있는 폴더)
python analysis/check_alignment.py <폴더>
```

## 출력 스키마

발화 단위 JSON 객체. 키는 음성 파일 경로.

| 필드 | 설명 |
|---|---|
| `wav` | 음성 파일 경로 |
| `duration` | 발화 길이(초) |
| `spk_id` | 화자 식별자 |
| `canonical_aligned` | 정발음 음소열(묵음 포함, 원본 정렬 보존) |
| `perceived_aligned` | 실제 발음 음소열(묵음 포함) |
| `canonical_train_target` | 정발음 학습 목표(시작·끝 묵음 제거, 연속 묵음 압축) |
| `perceived_train_target` | 실제 발음 학습 목표 |
| `wrd` | 원문 텍스트 |

## 참조

- L2-ARCTIC 전처리 정규화 규칙은 [Mu-Y/mpl-mdd](https://github.com/Mu-Y/mpl-mdd)(MIT)의 `l2arctic_prepare.py`를 기준으로 삼았다.
- 정발음 재생성: [Kyubyong/g2p](https://github.com/Kyubyong/g2p) (g2p_en).

## License

MIT. 데이터 자체의 이용 조건은 각 제공 기관의 라이선스를 따른다.
