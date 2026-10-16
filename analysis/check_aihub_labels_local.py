"""AI-Hub PRN 원본 라벨 전수 검증 + 로컬 산출(aihub_train.json) 확인
사용법: python check_aihub_labels_local.py <prn_label_train 폴더> <aihub_train.json 경로>
검증 1) transAnsw / phonemeText / tagging 토큰 수가 같은가 (위치 정렬 가설)
검증 2) tagging이 "phonemeText ***→D, transAnsw ***→I, 다르면 S, 같으면 C" 규칙과 일치하는가
검증 3) 파일당 문장(contents) 수 분포, ***가 양쪽에 등장하는 비율
검증 4) 로컬 산출 aihub_train.json의 길이 분포 (B4: 문단 파일 포함 여부)
"""
import json, sys, os
from collections import Counter

label_dir, train_json = sys.argv[1], sys.argv[2]

files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
n_files = n_prn = 0
n_sent = Counter()            # 파일당 문장 수 분포
len_equal = len_diff = 0      # 세 필드 토큰 수 일치 여부 (문장 단위)
rule_ok = rule_bad = 0        # tagging 규칙 일치 여부 (토큰 단위)
tag_other = 0                 # O 태그 수
star_trans = star_phon = star_any = sent_total = 0
missing = Counter()
bad_examples = []

for i, f in enumerate(files):
    if (i + 1) % 20000 == 0:
        print(f'  {i+1}/{len(files)} 처리 중...')
    try:
        d = json.load(open(os.path.join(label_dir, f), encoding='utf-8-sig'))
    except Exception:
        missing['json 읽기 실패'] += 1
        continue
    n_files += 1
    if d.get('scriptInfo', {}).get('evaluationArea') != 'PRN':
        continue
    n_prn += 1
    contents = d.get('itemScript', {}).get('contents', []) or []
    n_sent[len(contents)] += 1
    for c in contents:
        t = (c.get('transAnsw') or '').split()
        p = (c.get('phonemeText') or '').split()
        g = (c.get('tagging') or '').split()
        if not t or not p or not g:
            missing['필드 비어 있음'] += 1
            continue
        sent_total += 1
        st, sp = '***' in t, '***' in p
        star_trans += st; star_phon += sp; star_any += (st or sp)
        if len(t) == len(p) == len(g):
            len_equal += 1
            for a, b, c_ in zip(t, p, g):
                if c_ == 'O':
                    tag_other += 1
                    continue
                exp = 'D' if b == '***' else 'I' if a == '***' else ('C' if a == b else 'S')
                if exp == c_:
                    rule_ok += 1
                else:
                    rule_bad += 1
                    if len(bad_examples) < 5:
                        bad_examples.append((f, a, b, c_, exp))
        else:
            len_diff += 1

print('=== 원본 라벨 전수 검증 ===')
print(f'파일 {n_files:,} | PRN 파일 {n_prn:,} | 문장 항목 {sent_total:,} | 읽기 실패·필드 누락 {dict(missing)}')
print(f'파일당 문장 수 분포: {dict(sorted(n_sent.items()))}')
print(f'세 필드 토큰 수 일치: {len_equal:,} / 불일치: {len_diff:,}')
print(f'tagging 규칙 일치 토큰: {rule_ok:,} / 불일치: {rule_bad:,} / O 태그: {tag_other:,}')
print(f'*** 포함 문장: transAnsw {star_trans:,} | phonemeText {star_phon:,} | 둘 중 하나 {star_any:,} ({star_any/max(sent_total,1)*100:.1f}%)')
for ex in bad_examples:
    print('  불일치 예:', ex)

print('\n=== 로컬 산출 aihub_train.json ===')
d = json.load(open(train_json, encoding='utf-8'))
dur = [v['duration'] for v in d.values()]
print(f'항목 {len(dur):,} | 합계 {sum(dur)/3600:.1f}h | 10초 초과 {sum(x>10 for x in dur):,} | 15초 초과 {sum(x>15 for x in dur):,} | 30초 초과 {sum(x>30 for x in dur):,}')
