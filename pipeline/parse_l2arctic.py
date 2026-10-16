"""
L2-ARCTIC 데이터셋 파싱 스크립트 v3 (최종)
출력: 각 화자별 JSON + 전체 병합 JSON

변경사항 (v3):
  - 비표준 음소 정규화 추가 (레포 참고)
    · 알파벳 외 문자 제거: r* → r, er) → er
    · ax → ah
    · spn → sil
    · err → err (허용 음소로 유지)

사용법:
    python parse_l2arctic.py --data_dir C:/path/to/l2arctic --output_dir ./output
"""

import re
import json
import wave
import argparse
from pathlib import Path


# ── 음소 정규화 ──────────────────────────────────────────────
def normalize_phoneme(ph: str) -> str:
    """
    ARPAbet 음소 정규화
    1. 공백 제거
    2. sp / spn / sil / 빈 문자열 → sil
    3. 알파벳 외 문자 제거 (r* → r, er) → er)
    4. 소문자 변환
    5. 스트레스 번호 제거 (ao1 → ao)
    6. ax → ah
    7. err → err (허용)
    """
    ph = ph.strip()

    # silence 계열
    if ph.lower() in {"sil", "sp", "spn", "pau", ""}:
        return "sil"

    # 알파벳 외 문자 제거 (숫자 포함), 소문자 변환
    ph = re.sub(r"[^a-zA-Z]", "", ph).lower()

    if ph == "":
        return "sil"

    # ax → ah
    if ph == "ax":
        return "ah"

    # err 허용
    if ph == "err":
        return "err"

    return ph


# ── canonical 파싱 (textgrid 폴더) ───────────────────────────
def parse_textgrid_canonical(filepath: str) -> list:
    phones = []
    in_phones_tier = False

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_s = line.strip()
        if 'name = "phones"' in line_s:
            in_phones_tier = True
            continue
        if in_phones_tier and 'name = "words"' in line_s:
            break
        if in_phones_tier and line_s.startswith("text ="):
            ph_raw = line_s.split("=", 1)[1].strip().strip('"')
            norm = normalize_phoneme(ph_raw)
            if norm != "":
                phones.append(norm)

    return phones


# ── perceived 파싱 (annotation 폴더) ─────────────────────────
def parse_annotation_perceived(filepath: str) -> list:
    """
    phones tier 포맷:
      단일값     → 정발음, perceived = 그 음소
      "D,S,s"   → substitution: perceived = S
      "sil,K,a" → addition:     perceived = K
      "D,,d"    → deletion:     perceived 없음 (생략)
    """
    phones = []
    in_phones_tier = False

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_s = line.strip()
        if 'name = "phones"' in line_s:
            in_phones_tier = True
            continue
        if in_phones_tier and ('name = "IPA"' in line_s or 'name = "words"' in line_s):
            break
        if in_phones_tier and line_s.startswith("text ="):
            ph_raw = line_s.split("=", 1)[1].strip().strip('"').strip()
            parts = ph_raw.split(",")
            if len(parts) == 3:
                # 오류 레이블 → perceived는 2번째 값
                perceived_raw = parts[1].strip()
                norm = normalize_phoneme(perceived_raw)
                if norm not in ("", "sil"):  # deletion이면 생략
                    phones.append(norm)
                elif norm == "sil" and perceived_raw.lower() in {"sil", "sp", "spn"}:
                    phones.append("sil")
            else:
                norm = normalize_phoneme(ph_raw)
                if norm != "":
                    phones.append(norm)

    return phones


# ── WAV duration ──────────────────────────────────────────────
def get_wav_duration(filepath: str) -> float:
    with wave.open(filepath, "r") as wf:
        return round(wf.getnframes() / wf.getframerate(), 5)


# ── train_target 변환 (CTC 특성 반영) ────────────────────────
def to_train_target(phones: list) -> list:
    """
    1. 시작 sil 제거
    2. 끝 sil 제거
    3. 연속된 sil → sil 하나로 압축
    """
    phones = phones.copy()

    # 시작 sil 제거
    while phones and phones[0] == "sil":
        phones.pop(0)
    # 끝 sil 제거
    while phones and phones[-1] == "sil":
        phones.pop()
    # 연속 sil 압축
    result = []
    for ph in phones:
        if ph == "sil" and result and result[-1] == "sil":
            continue
        result.append(ph)

    return result


# ── 화자 폴더 파싱 ────────────────────────────────────────────
def parse_speaker(speaker_dir: str) -> dict:
    speaker_dir    = Path(speaker_dir)
    spk_id         = speaker_dir.name
    textgrid_dir   = speaker_dir / "textgrid"
    annotation_dir = speaker_dir / "annotation"
    transcript_dir = speaker_dir / "transcript"
    wav_dir        = speaker_dir / "wav"

    results = {}
    skipped_no_ann = 0

    for wav_file in sorted(wav_dir.glob("*.wav")):
        stem     = wav_file.stem
        wav_path = str(wav_file)

        # annotation 없으면 스킵
        ann_file = annotation_dir / f"{stem}.TextGrid"
        if not ann_file.exists():
            skipped_no_ann += 1
            continue

        # transcript
        txt_file = transcript_dir / f"{stem}.txt"
        if not txt_file.exists():
            print(f"  [SKIP] transcript 없음: {stem}")
            continue

        # canonical (textgrid 폴더)
        tg_file = textgrid_dir / f"{stem}.TextGrid"
        if not tg_file.exists():
            print(f"  [SKIP] textgrid 없음: {stem}")
            continue

        wrd              = txt_file.read_text(encoding="utf-8").strip()
        canonical_phones = parse_textgrid_canonical(str(tg_file))
        perceived_phones = parse_annotation_perceived(str(ann_file))
        duration         = get_wav_duration(wav_path)

        results[wav_path] = {
            "wav":                    wav_path,
            "duration":               duration,
            "spk_id":                 spk_id,
            "canonical_aligned":      " ".join(canonical_phones),
            "perceived_aligned":      " ".join(perceived_phones),
            "canonical_train_target": " ".join(to_train_target(canonical_phones)),
            "perceived_train_target": " ".join(to_train_target(perceived_phones)),
            "wrd":                    wrd,
        }

    print(f"  → {len(results)}개 발화 처리 / annotation 없어서 스킵: {skipped_no_ann}개")
    return results


# ── 메인 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,
                        help="l2arctic 루트 경로 (화자 폴더들이 있는 곳)")
    parser.add_argument("--output_dir", default="./output",
                        help="결과물 저장 경로 (기본값: ./output)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data     = {}
    speaker_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"총 화자 수: {len(speaker_dirs)}")

    for spk_dir in speaker_dirs:
        spk_id = spk_dir.name
        print(f"\n[{spk_id}] 파싱 중...")
        spk_data = parse_speaker(str(spk_dir))

        with open(output_dir / f"{spk_id}.json", "w", encoding="utf-8") as f:
            json.dump(spk_data, f, ensure_ascii=False, indent=2)

        all_data.update(spk_data)

    with open(output_dir / "l2arctic_all.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    total = len(all_data)
    print(f"\n{'='*40}")
    print(f"✅ 완료! 총 {total}개 발화 (annotation 있는 발화만)")
    print(f"   출력 경로: {output_dir}/")


if __name__ == "__main__":
    main()
