"""
AI-Hub 한국인 영어 음성 데이터 (PRN) 파싱 스크립트 v2
출력: aihub_train.json, aihub_val.json

처리 내용:
  1. 이중모음 병합 (split 전 문자열 replace)
  2. IPA → ARPAbet 변환
  3. *** 포함 발화 스킵
  4. perceived에 비정상 음소 있으면 스킵
  5. canonical에 비정상 음소 있으면 CMU(g2p_en)로 재생성
  6. train_target 생성 (시작/끝 sil 제거 + 연속 sil 압축)

사용법:
  python3 parse_aihub.py   # 아래 main()의 경로 3개를 환경에 맞게 수정
"""

import os
import json
import re
from pathlib import Path

try:
    from g2p_en import G2p
    g2p = G2p()
    G2P_AVAILABLE = True
except ImportError:
    print("[WARNING] g2p_en 미설치. canonical 재생성 불가.")
    G2P_AVAILABLE = False

# ── 모델 음소 목록 (41개) ─────────────────────────────────────
VALID_PHONEMES = {
    'sil','aa','ae','ah','ao','aw','ay','b','ch','d','dh','eh','er','ey',
    'f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r',
    's','sh','t','th','uh','uw','v','w','y','z','zh'
}

# ── IPA → ARPAbet 매핑 테이블 ────────────────────────────────
IPA_TO_ARPA = {
    'ə': 'ah', 'ɑ': 'aa', 'ɔ': 'ao', 'ɛ': 'eh', 'æ': 'ae',
    'ɪ': 'ih', 'ʊ': 'uh', 'i': 'iy', 'u': 'uw',
    'ay': 'ay', 'aw': 'aw', 'ow': 'ow', 'ey': 'ey', 'oy': 'oy',
    'ð': 'dh', 'θ': 'th', 'ŋ': 'ng', 'ʃ': 'sh', 'ʒ': 'zh',
    'ʤ': 'jh', 'ʧ': 'ch',
    'b': 'b', 'd': 'd', 'f': 'f', 'g': 'g', 'h': 'hh',
    'j': 'y', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
    'p': 'p', 'r': 'r', 's': 's', 't': 't', 'v': 'v',
    'w': 'w', 'z': 'z',
    'sil': 'sil', 'sp': 'sil', 'spn': 'sil',
}

# ── 이중모음 병합 ─────────────────────────────────────────────
DIPHTHONG_MAP = [
    ("a ɪ", "ay"), ("a ʊ", "aw"), ("o ʊ", "ow"),
    ("e ɪ", "ey"), ("ɔ ɪ", "oy"),
]

def merge_diphthongs(text):
    for ipa, arpa in DIPHTHONG_MAP:
        text = text.replace(ipa, arpa)
    return text

def convert_phoneme(ph):
    ph = ph.strip()
    if ph in ('***', 'null', 'nan', ''):
        return None
    if ph in VALID_PHONEMES:
        return ph
    if ph in IPA_TO_ARPA:
        return IPA_TO_ARPA[ph]
    return None

def convert_sequence(ipa_text):
    if not ipa_text or ipa_text.strip() in ('null', 'nan'):
        return None
    text = merge_diphthongs(ipa_text)
    phones = text.split()
    result = []
    for ph in phones:
        converted = convert_phoneme(ph)
        if converted is None:
            return None
        result.append(converted)
    return result if result else None

def generate_canonical_from_text(sentence):
    if not G2P_AVAILABLE:
        return None
    try:
        phones = g2p(sentence)
        result = []
        for ph in phones:
            if ph == ' ':
                continue
            ph_clean = re.sub(r'\d', '', ph).lower()
            if ph_clean in VALID_PHONEMES:
                result.append(ph_clean)
            elif ph_clean == '':
                continue
            else:
                return None
        return result if result else None
    except Exception:
        return None

def to_train_target(phones):
    phones = phones.copy()
    while phones and phones[0] == 'sil':
        phones.pop(0)
    while phones and phones[-1] == 'sil':
        phones.pop()
    result = []
    for ph in phones:
        if ph == 'sil' and result and result[-1] == 'sil':
            continue
        result.append(ph)
    return result

def parse_json_file(json_path):
    with open(json_path, encoding='utf-8-sig') as f:
        d = json.load(f)

    eval_area = d.get('scriptInfo', {}).get('evaluationArea', '')
    if eval_area != 'PRN':
        return None

    contents = d.get('itemScript', {}).get('contents', [])
    if not contents:
        return None

    c = contents[0]
    trans_answ   = c.get('transAnsw', '') or ''
    phoneme_text = c.get('phonemeText', '') or ''
    sentence     = c.get('sentence', '') or ''

    if '***' in trans_answ or '***' in phoneme_text:
        return None

    perceived_phones = convert_sequence(phoneme_text)
    if perceived_phones is None:
        return None

    canonical_phones = convert_sequence(trans_answ)
    if canonical_phones is None:
        canonical_phones = generate_canonical_from_text(sentence)
        if canonical_phones is None:
            return None

    file_nm  = d.get('fileNm', '')
    duration = d.get('recrdTime', 0.0)
    spk_id   = str(d.get('recorder', {}).get('recorderId', ''))
    wav_path = file_nm + '.wav'

    return {
        'wav':                    wav_path,
        'duration':               duration,
        'spk_id':                 spk_id,
        'canonical_aligned':      ' '.join(canonical_phones),
        'perceived_aligned':      ' '.join(perceived_phones),
        'canonical_train_target': ' '.join(to_train_target(canonical_phones)),
        'perceived_train_target': ' '.join(to_train_target(perceived_phones)),
        'wrd':                    sentence,
    }

def parse_folder(label_dir, label):
    label_dir  = Path(label_dir)
    results    = {}
    skipped    = 0
    json_files = list(label_dir.glob('*.json'))
    total      = len(json_files)
    print(f"\n[{label}] 총 {total}개 파일 처리 중...")

    for i, json_file in enumerate(json_files):
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{total} 처리 중...")
        try:
            entry = parse_json_file(str(json_file))
            if entry is None:
                skipped += 1
                continue
            results[entry['wav']] = entry
        except Exception:
            skipped += 1
            continue

    print(f"  → {len(results)}개 처리 완료 / {skipped}개 스킵")
    return results

def main():
    LABEL_TRAIN = '<path to prn_label_train>'
    LABEL_VAL   = '<path to prn_label_val>'
    OUTPUT_DIR  = '<output dir>'

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=== AI-Hub PRN 파싱 시작 ===")

    # Training
    train_data = parse_folder(LABEL_TRAIN, 'Training')
    train_out  = os.path.join(OUTPUT_DIR, 'aihub_train.json')
    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # Validation
    val_data = parse_folder(LABEL_VAL, 'Validation')
    val_out  = os.path.join(OUTPUT_DIR, 'aihub_val.json')
    with open(val_out, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    train_time = sum(v['duration'] for v in train_data.values())
    val_time   = sum(v['duration'] for v in val_data.values())

    print(f"\n{'='*40}")
    print(f"✅ 완료!")
    print(f"   Training: {len(train_data)}개 / {train_time/3600:.1f}시간 → {train_out}")
    print(f"   Validation: {len(val_data)}개 / {val_time/3600:.1f}시간 → {val_out}")

if __name__ == '__main__':
    main()
