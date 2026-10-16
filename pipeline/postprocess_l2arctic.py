import json
from pathlib import Path

# 모델 음소 목록 41개
VALID_PHONEMES = {
    'sil','aa','ae','ah','ao','aw','ay','b','ch','d','dh','eh','er','ey',
    'f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r',
    's','sh','t','th','uh','uw','v','w','y','z','zh'
}

def normalize_phoneme(ph):
    """음소 정규화: err → er"""
    if ph == 'err':
        return 'er'
    return ph

def is_valid_sequence(phones):
    """
    음소 시퀀스가 유효한지 검사
    - 모델 목록에 없는 음소가 하나라도 있으면 False
    - err은 er로 정규화 후 검사
    """
    for ph in phones:
        normalized = normalize_phoneme(ph)
        if normalized not in VALID_PHONEMES:
            return False
    return True

def normalize_sequence(phones):
    """음소 시퀀스 전체 정규화"""
    return [normalize_phoneme(ph) for ph in phones]

def main():
    input_path  = r'<path to l2arctic_all.json>'
    output_path = r'<path to l2arctic_clean.json>'

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    clean_data = {}
    skipped = 0
    skip_reasons = {}

    for wav_path, entry in data.items():

        # 검사할 필드: canonical_aligned, perceived_aligned
        canonical_phones = entry['canonical_aligned'].split()
        perceived_phones = entry['perceived_aligned'].split()

        # 비정상 음소 검사 (err은 er로 정규화 후 검사)
        if not is_valid_sequence(canonical_phones):
            invalid = [ph for ph in canonical_phones if normalize_phoneme(ph) not in VALID_PHONEMES]
            skipped += 1
            for ph in invalid:
                skip_reasons[ph] = skip_reasons.get(ph, 0) + 1
            continue

        if not is_valid_sequence(perceived_phones):
            invalid = [ph for ph in perceived_phones if normalize_phoneme(ph) not in VALID_PHONEMES]
            skipped += 1
            for ph in invalid:
                skip_reasons[ph] = skip_reasons.get(ph, 0) + 1
            continue

        # 정규화 적용 (err → er)
        canonical_aligned      = ' '.join(normalize_sequence(canonical_phones))
        perceived_aligned      = ' '.join(normalize_sequence(perceived_phones))
        canonical_train_target = ' '.join(normalize_sequence(entry['canonical_train_target'].split()))
        perceived_train_target = ' '.join(normalize_sequence(entry['perceived_train_target'].split()))

        clean_data[wav_path] = {
            'wav':                    entry['wav'],
            'duration':               entry['duration'],
            'spk_id':                 entry['spk_id'],
            'canonical_aligned':      canonical_aligned,
            'perceived_aligned':      perceived_aligned,
            'canonical_train_target': canonical_train_target,
            'perceived_train_target': perceived_train_target,
            'wrd':                    entry['wrd'],
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print(f'원본 발화 수:   {len(data)}')
    print(f'스킵된 발화 수: {skipped}')
    print(f'최종 발화 수:   {len(clean_data)}')
    print(f'\n스킵 원인별 음소:')
    for ph, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f'  {cnt}회: "{ph}"')
    print(f'\n출력: {output_path}')

if __name__ == '__main__':
    main()
