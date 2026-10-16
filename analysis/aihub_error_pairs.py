"""AI-Hub 원본 라벨의 tagging 기준 오류 쌍 집계 (졸업논문 표 11용)
사용법: python aihub_error_pairs.py <prn_label_train 폴더> <aihub_clean_v2.json>
출력: v2 포함 문장 / 전체 문장 각각의 상위 대치 쌍(정발음→실제 발음), 삭제 음소, 삽입 음소
"""
import json, os, re, sys
from collections import Counter

label_dir, v2_path = sys.argv[1], sys.argv[2]

IPA_TO_ARPA = {
    'ə':'ah','ɑ':'aa','ɔ':'ao','ɛ':'eh','æ':'ae','ɪ':'ih','ʊ':'uh','i':'iy','u':'uw',
    'ð':'dh','θ':'th','ŋ':'ng','ʃ':'sh','ʒ':'zh','ʤ':'jh','ʧ':'ch','h':'hh','j':'y',
    'b':'b','d':'d','f':'f','g':'g','k':'k','l':'l','m':'m','n':'n','p':'p','r':'r',
    's':'s','t':'t','v':'v','w':'w','z':'z',
}
def arpa(t): return IPA_TO_ARPA.get(t, t)   # 대응표 밖 기호(a, o 등)는 원본 그대로

def norm_key(k):
    k = re.split(r'[\\/]', k)[-1]
    k = re.sub(r'^ah_', '', k)
    k = re.sub(r'\.(wav|json)$', '', k)
    return re.sub(r'_s\d+$', '', k)

v2_keys = {norm_key(k) for k in json.load(open(v2_path, encoding='utf-8'))}

def new(): return {'S': Counter(), 'D': Counter(), 'I': Counter(), 'sent': 0, 'phones': 0}
agg = {'v2 포함 문장': new(), '전체 문장': new()}

files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
for i, f in enumerate(files):
    if (i + 1) % 20000 == 0: print(f'  {i+1}/{len(files)} 처리 중...')
    d = json.load(open(os.path.join(label_dir, f), encoding='utf-8-sig'))
    stem = f[:-5]
    contents = d.get('itemScript', {}).get('contents', []) or []
    for idx, c in enumerate(contents):
        t = (c.get('transAnsw') or '').split(); p = (c.get('phonemeText') or '').split(); g = (c.get('tagging') or '').split()
        if not (t and p and g and len(t) == len(p) == len(g)): continue
        targets = ['전체 문장']
        if idx == 0 and stem in v2_keys: targets.append('v2 포함 문장')
        for name in targets:
            a = agg[name]; a['sent'] += 1
            for x, y, tag in zip(t, p, g):
                if x != '***': a['phones'] += 1
                if tag == 'S': a['S'][(arpa(x), arpa(y))] += 1
                elif tag == 'D': a['D'][arpa(x)] += 1
                elif tag == 'I': a['I'][arpa(y)] += 1

for name, a in agg.items():
    print(f'\n=== {name}: {a["sent"]:,}문장, 정발음 음소 {a["phones"]:,} ===')
    print('상위 대치 쌍(정발음→실제 발음):')
    for (x, y), n in a['S'].most_common(15): print(f'   {x}→{y}  {n:,}')
    print('상위 삭제 음소:', [(x, n) for x, n in a['D'].most_common(8)])
    print('상위 삽입 음소:', [(y, n) for y, n in a['I'].most_common(8)])
