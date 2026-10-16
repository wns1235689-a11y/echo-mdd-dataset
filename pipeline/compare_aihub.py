"""
원본 AI-Hub JSON vs 서버 처리 데이터 비교 검증 스크립트 v3

변경사항:
- ə r → er 병합 로직 추가
- canonical_train_target / perceived_train_target 비교
- canonical_aligned / perceived_aligned 비교

검증 항목:
1. train_target 정확도
2. aligned 정확도
3. duration 일치 여부
4. sentence 일치 여부
"""

import json
import os
import re

# ── 경로 설정 ─────────────────────────────────────────────────
SAMPLE_V1  = '<path to sample_50_aihub_clean_v1.json>'
SAMPLE_V2  = '<path to sample_50_aihub_clean_v2.json>'
ORIG_TRAIN = '<path to prn_label_train>'
ORIG_VAL   = '<path to prn_label_val>'

# ── 이중모음 병합 ─────────────────────────────────────────────
DIPHTHONG_MAP = [
    ("a ɪ", "ay"), ("a ʊ", "aw"), ("o ʊ", "ow"),
    ("e ɪ", "ey"), ("ɔ ɪ", "oy"),
]

def merge_diphthongs(text):
    for ipa, arpa in DIPHTHONG_MAP:
        text = text.replace(ipa, arpa)
    return text

# ── IPA → ARPAbet 변환 ────────────────────────────────────────
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

def convert_ipa_to_arpabet(ipa_text, add_boundary_sil=True):
    """
    원본 IPA 시퀀스 → ARPAbet 변환
    - 이중모음 병합
    - *** → sil
    - ə r → er 병합
    - IPA → ARPAbet
    - add_boundary_sil=True: 시작/끝 sil 추가 (aligned용)
    - add_boundary_sil=False: 시작/끝 sil 없음 (train_target용)
    """
    if not ipa_text or ipa_text.strip() in ('null', 'nan'):
        return None

    # 이중모음 병합
    text = merge_diphthongs(ipa_text)
    phones = text.split()

    # *** → sil, IPA → ARPAbet 변환
    converted = []
    i = 0
    while i < len(phones):
        ph = phones[i].strip()
        if ph == '***':
            converted.append('sil')
        elif ph == 'ə' and i + 1 < len(phones) and phones[i+1].strip() == 'r':
            # ə r → er 병합
            converted.append('er')
            i += 1  # r 스킵
        elif ph in IPA_TO_ARPA:
            converted.append(IPA_TO_ARPA[ph])
        else:
            converted.append(f'[UNK:{ph}]')
        i += 1

    if add_boundary_sil:
        return ['sil'] + converted + ['sil']
    else:
        # train_target: 시작/끝 sil 제거, 연속 sil 유지 (*** → sil 반영)
        while converted and converted[0] == 'sil':
            converted.pop(0)
        while converted and converted[-1] == 'sil':
            converted.pop()
        return converted

def find_orig_file(key, orig_dirs):
    base_key = re.sub(r'_s\d+$', '', key)
    filename = base_key + '.json'
    for d in orig_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def compare_sequences(expected, actual, label, max_show=5):
    issues = []
    if len(expected) != len(actual):
        issues.append(f"{label} 길이 불일치: 예상={len(expected)}, 처리={len(actual)}")
    min_len = min(len(expected), len(actual))
    diffs = [(i, expected[i], actual[i]) for i in range(min_len) if expected[i] != actual[i]]
    if diffs:
        diff_strs = [f"위치{i}: 예상={e}, 처리={a}" for i, e, a in diffs[:max_show]]
        issues.append(f"{label} 불일치 ({len(diffs)}곳):\n     " + "\n     ".join(diff_strs))
    return issues

def compare_entry(key, processed, orig_dirs):
    issues = []

    orig_path = find_orig_file(key, orig_dirs)
    if not orig_path:
        return [f"원본 파일 없음"]

    with open(orig_path, encoding='utf-8-sig') as f:
        orig = json.load(f)

    contents = orig.get('itemScript', {}).get('contents', [])
    if not contents:
        return [f"contents 없음"]

    s_match = re.search(r'_s(\d+)$', key)
    sentence_idx = int(s_match.group(1)) if s_match else 0
    if sentence_idx >= len(contents):
        return [f"sentence_idx {sentence_idx} 범위 초과 (총 {len(contents)}개)"]

    c = contents[sentence_idx]

    # ── 1. duration 비교 ──
    orig_dur = orig.get('recrdTime', 0)
    proc_dur = processed.get('duration', 0)
    if abs(orig_dur - proc_dur) > 0.01:
        issues.append(f"duration 불일치: 원본={orig_dur}, 처리={proc_dur}")

    # ── 2. sentence 비교 ──
    orig_sent = c.get('sentence', '').strip()
    proc_sent = processed.get('sentence', processed.get('wrd', '')).strip()
    if orig_sent != proc_sent:
        issues.append(f"sentence 불일치:\n     원본: {orig_sent}\n     처리: {proc_sent}")

    orig_trans   = c.get('transAnsw', '') or ''
    orig_phoneme = c.get('phonemeText', '') or ''

    # ── 3. canonical_aligned 비교 ──
    if orig_trans and orig_trans not in ('null', 'nan'):
        expected = convert_ipa_to_arpabet(orig_trans, add_boundary_sil=True)
        actual   = processed.get('canonical_aligned', '').split()
        if expected and actual:
            issues.extend(compare_sequences(expected, actual, "canonical_aligned"))

    # ── 4. perceived_aligned 비교 ──
    if orig_phoneme and orig_phoneme not in ('null', 'nan'):
        expected = convert_ipa_to_arpabet(orig_phoneme, add_boundary_sil=True)
        actual   = processed.get('perceived_aligned', '').split()
        if expected and actual:
            issues.extend(compare_sequences(expected, actual, "perceived_aligned"))

    # ── 5. canonical_train_target 비교 ──
    if orig_trans and orig_trans not in ('null', 'nan'):
        expected = convert_ipa_to_arpabet(orig_trans, add_boundary_sil=False)
        actual   = processed.get('canonical_train_target', '').split()
        if expected and actual:
            issues.extend(compare_sequences(expected, actual, "canonical_train_target"))

    # ── 6. perceived_train_target 비교 ──
    if orig_phoneme and orig_phoneme not in ('null', 'nan'):
        expected = convert_ipa_to_arpabet(orig_phoneme, add_boundary_sil=False)
        actual   = processed.get('perceived_train_target', '').split()
        if expected and actual:
            issues.extend(compare_sequences(expected, actual, "perceived_train_target"))

    return issues

def main():
    orig_dirs = [ORIG_TRAIN, ORIG_VAL]

    for sample_path, label in [(SAMPLE_V1, 'v1'), (SAMPLE_V2, 'v2')]:
        print(f"\n{'='*55}")
        print(f"=== aihub_clean_{label}.json 샘플 50개 원본 비교 ===")
        print(f"{'='*55}")

        with open(sample_path, encoding='utf-8') as f:
            samples = json.load(f)

        total       = len(samples)
        clean       = 0
        issue_count = 0
        not_found   = 0

        for key, entry in samples.items():
            issues = compare_entry(key, entry, orig_dirs)
            if not issues:
                clean += 1
            else:
                issue_count += 1
                if any('원본 파일 없음' in i for i in issues):
                    not_found += 1
                print(f"\n⚠ {key}")
                for issue in issues:
                    print(f"   {issue}")

        print(f"\n--- 결과 ---")
        print(f"전체:         {total}개")
        print(f"✅ 정상:      {clean}개")
        print(f"⚠  이슈:     {issue_count}개")
        if not_found:
            print(f"   원본 없음: {not_found}개")

if __name__ == '__main__':
    main()
