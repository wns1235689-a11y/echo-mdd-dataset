"""최종 학습 데이터의 정렬 필드·학습 목표 필드 점검 (졸업논문 4.2.3, 5.3.3, 5.4, 표 13)
사용법: python check_alignment.py <train.json·val.json·test.json이 있는 폴더>

1) 원천·분할별로 두 정렬 필드(canonical_aligned, perceived_aligned)의 길이가 같은 발화 수
   - 모델 평가 코드는 두 정렬 필드의 길이가 다른 발화를 오탐·미탐 집계에서 제외한다
2) 학습 목표 필드의 생성 규칙
   - 공통 규칙: 시작·끝 sil 제거 + 연속 sil 압축 / 시작·끝 sil 제거만
3) AI-Hub 부분: 정렬 필드의 묵음을 자리 유형별로 나누고(정발음만 묵음=삽입 자리, 실제 발음만
   묵음=삭제 자리, 양쪽 묵음=공통), 학습 목표 필드에 남은 묵음을 유형별로 집계
4) L2-ARCTIC 부분: 두 학습 목표 필드를 편집 거리로 정렬했을 때 정발음 음소가 실제 발음의 sil과
   대응하는 자리 수
"""
import json, os, sys
from collections import Counter
from rapidfuzz.distance import Levenshtein

SIL = 'sil'
folder = sys.argv[1]

def strip_ends(seq):
    i, j = 0, len(seq)
    while i < j and seq[i] == SIL: i += 1
    while j > i and seq[j - 1] == SIL: j -= 1
    return i, j

def compress(seq):
    out = []
    for t in seq:
        if t == SIL and out and out[-1] == SIL: continue
        out.append(t)
    return out

def has_consecutive_sil(seq):
    return any(a == SIL and b == SIL for a, b in zip(seq, seq[1:]))

S = Counter()
for split in ('train', 'val', 'test'):
    data = json.load(open(os.path.join(folder, f'{split}.json'), encoding='utf-8'))
    for key, v in data.items():
        src = 'L2-ARCTIC' if '/arctic_' in key else 'AI-Hub'
        ca, pa = v['canonical_aligned'].split(), v['perceived_aligned'].split()
        ct, pt = v['canonical_train_target'].split(), v['perceived_train_target'].split()

        # 1) 정렬 필드 길이
        S[(src, split, 'n')] += 1
        S[(src, split, 'same_len')] += len(ca) == len(pa)

        # 2) 학습 목표 생성 규칙
        S[(src, 'n')] += 1
        for name, a, t in (('정발음', ca, ct), ('실제 발음', pa, pt)):
            i, j = strip_ends(a)
            S[(src, name, '공통 규칙(제거+압축)')] += compress(a[i:j]) == t
            S[(src, name, '시작·끝 제거만')] += a[i:j] == t
            S[(src, name, '연속 sil 남음')] += has_consecutive_sil(t)

        # 3) AI-Hub 묵음 자리 유형
        if src == 'AI-Hub' and len(ca) == len(pa):
            ci, cj = strip_ends(ca); pi, pj = strip_ends(pa)
            for k, (x, y) in enumerate(zip(ca, pa)):
                if x != SIL and y != SIL: continue
                kind = '공통' if (x == SIL and y == SIL) else ('삽입 자리' if x == SIL else '삭제 자리')
                if x == SIL:
                    S[('정발음', '정렬', kind)] += 1
                    S[('정발음', '목표', kind)] += ci <= k < cj
                if y == SIL:
                    S[('실제 발음', '정렬', kind)] += 1
                    S[('실제 발음', '목표', kind)] += pi <= k < pj
            S[('정발음', '목표 묵음 합계')] += ct.count(SIL)
            S[('실제 발음', '목표 묵음 합계')] += pt.count(SIL)

        # 4) L2-ARCTIC: 정발음 음소 ↔ 실제 발음 sil
        if src == 'L2-ARCTIC':
            n = sum(1 for o in Levenshtein.editops(ct, pt)
                    if o.tag == 'replace' and ct[o.src_pos] != SIL and pt[o.dest_pos] == SIL)
            S['l2_phone_vs_sil'] += n
            S['l2_phone_vs_sil_utts'] += n > 0

print('=== 1) 두 정렬 필드의 길이가 같은 발화 (평가 코드의 오탐·미탐 집계 대상) ===')
for src in ('L2-ARCTIC', 'AI-Hub'):
    tot_n = tot_s = 0
    for split in ('train', 'val', 'test'):
        n, s = S[(src, split, 'n')], S[(src, split, 'same_len')]
        tot_n += n; tot_s += s
        print(f'  {src:9s} {split:5s} 발화 {n:6,} | 길이 같음 {s:6,} | 다름 {n - s:5,} ({(n - s) / n * 100:.1f}%)')
    print(f'  {src:9s} 합계  발화 {tot_n:6,} | 길이 같음 {tot_s:6,} | 다름 {tot_n - tot_s:5,} ({(tot_n - tot_s) / tot_n * 100:.1f}%)')

print('\n=== 2) 학습 목표 필드의 생성 규칙 ===')
for src in ('L2-ARCTIC', 'AI-Hub'):
    for name in ('정발음', '실제 발음'):
        print(f"  {src:9s} {name:5s}: 공통 규칙 일치 {S[(src, name, '공통 규칙(제거+압축)')]:,} | "
              f"시작·끝 제거만과 일치 {S[(src, name, '시작·끝 제거만')]:,} | 연속 sil 남은 항목 {S[(src, name, '연속 sil 남음')]:,} / {S[(src, 'n')]:,}")

print('\n=== 3) AI-Hub 묵음: 정렬 필드 → 학습 목표 필드 ===')
for name in ('정발음', '실제 발음'):
    kinds = ('삽입 자리', '공통') if name == '정발음' else ('삭제 자리', '공통')
    parts = ' | '.join(f"{k} {S[(name, '정렬', k)]:,} → {S[(name, '목표', k)]:,}" for k in kinds)
    print(f"  {name}: {parts} | 학습 목표 묵음 합계 {S[(name, '목표 묵음 합계')]:,}")

print('\n=== 4) L2-ARCTIC 학습 목표: 정발음 음소 ↔ 실제 발음 sil (편집 거리 정렬) ===')
print(f"  {S['l2_phone_vs_sil']:,}곳, {S['l2_phone_vs_sil_utts']:,}건 / {S[('L2-ARCTIC', 'n')]:,}건")
