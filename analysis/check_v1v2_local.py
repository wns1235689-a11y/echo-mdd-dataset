"""서버 정제본 aihub_clean_v1.json / v2.json 의 구성 규칙 판별 (몇 분 소요)
사용법: python check_v1v2_local.py aihub_clean_v1.json aihub_clean_v2.json
"""
import json, re, sys
from collections import Counter
def load(p): return json.load(open(p, encoding='utf-8'))
v1, v2 = load(sys.argv[1]), load(sys.argv[2])
def file_key(k): return re.sub(r'_s\d+$', '', k)
for name, d in [('v1', v1), ('v2', v2)]:
    keys = list(d)
    files = {file_key(k) for k in keys}
    with_s = sum(1 for k in keys if re.search(r'_s\d+$', k))
    print(f'[{name}] 항목 {len(keys):,} | _sN 접미사 항목 {with_s:,} | 고유 파일 수 {len(files):,}')
    lens = Counter()
    over30 = sum(1 for v in d.values() if v.get('duration', 0) > 30)
    sil_mid = sum(1 for v in d.values() if 'sil' in v['perceived_train_target'].split()[1:-1])
    print(f'      30초 초과 {over30:,} | perceived 중간 sil 포함 {sil_mid:,}')
# v2 ⊂ v1 ?
in_v1 = sum(1 for k in v2 if k in v1)
print(f'v2 키 중 v1에 있는 것: {in_v1:,} / {len(v2):,}')
# v1에서 v2로 살아남은 것과 탈락한 것의 특성 비교
kept = [v1[k] for k in v2 if k in v1]; dropped = [v for k, v in v1.items() if k not in v2]
def summarize(lst, label):
    if not lst: print(label, '없음'); return
    n = len(lst)
    print(f'{label}: {n:,}건 | 평균 길이 {sum(v["duration"] for v in lst)/n:.1f}s | 30초 초과 {sum(v["duration"]>30 for v in lst):,}'
          f' | perceived 중간 sil {sum("sil" in v["perceived_train_target"].split()[1:-1] for v in lst):,}'
          f' | canonical 중간 sil {sum("sil" in v["canonical_train_target"].split()[1:-1] for v in lst):,}'
          f' | _sN 키 {sum(bool(re.search(r"_s\d+$", v.get("wav",""))) for v in lst):,}')
summarize(kept, 'v1→v2 보존'); summarize(dropped, 'v1→v2 탈락')
