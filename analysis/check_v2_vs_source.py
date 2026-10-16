"""AI-Hub 원본 라벨 vs 서버 v2 / 로컬 산출 대조 (졸업논문 5.3·6.1용)
사용법:
  python check_v2_vs_source.py <prn_label_train 폴더> <aihub_clean_v2.json> <aihub_train.json> <aihub_val.json>
(a) 원본 tagging 기준 오류 분포: 전체 / v2 대응 문장 / 로컬 산출 대응 문장
(b) 서버 v2 규칙 역추적: 병합 변형별 정확 일치율, 원본 *** 위치와 v2 묵음(sil) 개수 대응
(c) 로컬 산출: 문단 파일(문장 2개 이상) 항목 수와 시간, train/val 중복
"""
import json, os, re, sys
from collections import Counter

label_dir, v2_path, local_train, local_val = sys.argv[1:5]

VALID = {'sil','aa','ae','ah','ao','aw','ay','b','ch','d','dh','eh','er','ey','f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh'}
IPA_TO_ARPA = {
    'ə':'ah','ɑ':'aa','ɔ':'ao','ɛ':'eh','æ':'ae','ɪ':'ih','ʊ':'uh','i':'iy','u':'uw',
    'ð':'dh','θ':'th','ŋ':'ng','ʃ':'sh','ʒ':'zh','ʤ':'jh','ʧ':'ch',
    'b':'b','d':'d','f':'f','g':'g','h':'hh','j':'y','k':'k','l':'l','m':'m','n':'n',
    'p':'p','r':'r','s':'s','t':'t','v':'v','w':'w','z':'z',
    'e':'eh',   # 서버 대조 때 관찰된 대응([UNK:e] -> eh)
}
DIPH = [("a ɪ","ay"),("a ʊ","aw"),("o ʊ","ow"),("e ɪ","ey"),("ɔ ɪ","oy")]
VARIANTS = {'V0 병합없음':(False,False), 'V1 이중모음':(True,False), 'V2 이중모음+ə r':(True,True), 'V3 ə r만':(False,True)}

def norm_key(k):
    k = re.split(r'[\\/]', k)[-1]
    k = re.sub(r'^ah_', '', k)
    k = re.sub(r'\.(wav|json)$', '', k)
    return re.sub(r'_s\d+$', '', k)

def expected(tokens, md, me):
    s = ' '.join(tokens)
    if md:
        for a, b in DIPH: s = s.replace(a, b)
    if me: s = s.replace('ə r', 'er')
    out = []
    for t in s.split():
        if t == '***': out.append('sil')
        elif t in VALID: out.append(t)
        elif t in IPA_TO_ARPA: out.append(IPA_TO_ARPA[t])
        else: out.append(f'<UNK:{t}>')
    while out and out[0] == 'sil': out.pop(0)
    while out and out[-1] == 'sil': out.pop()
    return out

def interior_star(tokens):
    t = list(tokens)
    while t and t[0] == '***': t.pop(0)
    while t and t[-1] == '***': t.pop()
    return t.count('***')

# ── 대상 키 로드
v2 = json.load(open(v2_path, encoding='utf-8'))
v2_by = {norm_key(k): v for k, v in v2.items()}
loc = {}
for name, p in [('train', local_train), ('val', local_val)]:
    d = json.load(open(p, encoding='utf-8'))
    loc[name] = {norm_key(k): v for k, v in d.items()}
need = set(v2_by) | set(loc['train']) | set(loc['val'])

# ── 원본 순회
def new_tag(): return {'C':0,'S':0,'D':0,'I':0,'O':0,'기타':0,'문장':0,'무오류':0,'대치만':0,'삽입삭제포함':0}
tags = {'전체': new_tag(), 'v2 대응': new_tag(), '로컬 train 대응': new_tag(), '로컬 val 대응': new_tag()}
def add_tags(bucket, g):
    b = tags[bucket]; b['문장'] += 1
    for x in g: b[x if x in b else '기타'] += 1
    if 'D' in g or 'I' in g: b['삽입삭제포함'] += 1
    elif 'S' in g: b['대치만'] += 1
    else: b['무오류'] += 1

src = {}
inv = {'transAnsw': Counter(), 'phonemeText': Counter()}
files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
for i, f in enumerate(files):
    if (i + 1) % 20000 == 0: print(f'  {i+1}/{len(files)} 처리 중...')
    d = json.load(open(os.path.join(label_dir, f), encoding='utf-8-sig'))
    contents = d.get('itemScript', {}).get('contents', []) or []
    for c in contents:
        g = (c.get('tagging') or '').split()
        if g: add_tags('전체', g)
        for fld in inv: inv[fld].update((c.get(fld) or '').split())
    keys = {f[:-5]}
    if d.get('fileNm'): keys.add(norm_key(d['fileNm']))
    hit = keys & need
    if hit and contents:
        c = contents[0]
        rec = {'t': (c.get('transAnsw') or '').split(), 'p': (c.get('phonemeText') or '').split(),
               'g': (c.get('tagging') or '').split(), 'n_sent': len(contents)}
        for k in hit: src[k] = rec

for bucket, keyset in [('v2 대응', v2_by), ('로컬 train 대응', loc['train']), ('로컬 val 대응', loc['val'])]:
    for k in keyset:
        if k in src and src[k]['g']: add_tags(bucket, src[k]['g'])

print('\n=== (a) 원본 tagging 기준 오류 분포 ===')
for name, b in tags.items():
    err = b['S'] + b['D'] + b['I']; n = max(b['문장'], 1); e = max(err, 1)
    print(f"[{name}] 문장 {b['문장']:,} | 무오류 {b['무오류']/n*100:.1f}% 대치만 {b['대치만']/n*100:.1f}% 삽입·삭제 포함 {b['삽입삭제포함']/n*100:.1f}%")
    print(f"   토큰 C {b['C']:,} S {b['S']:,} D {b['D']:,} I {b['I']:,} O {b['O']:,} 기타 {b['기타']:,} | 오류 중 S/D/I = {b['S']/e*100:.1f} / {b['D']/e*100:.1f} / {b['I']/e*100:.1f}% | 음소 오류율(오류/정발음 음소) {err/max(b['C']+b['S']+b['D'],1)*100:.1f}%")

print('\n=== 원본 기호 목록 (매핑표·*** 밖의 기호, 상위 15) ===')
for fld, cnt in inv.items():
    other = [(t, n) for t, n in cnt.most_common() if t not in IPA_TO_ARPA and t != '***']
    print(f'{fld}: 전체 기호 {len(cnt)}종 | 매핑표 밖 {len(other)}종 {other[:15]}')

print('\n=== (b) 서버 v2 규칙 역추적 ===')
matched = [k for k in v2_by if k in src]
print(f'v2 항목 {len(v2_by):,} | 원본 매칭 {len(matched):,} | 문장 2개 이상 파일 {sum(src[k]["n_sent"]>1 for k in matched):,}')
for side, fld, key in [('정발음', 'canonical_train_target', 't'), ('실제 발음', 'perceived_train_target', 'p')]:
    print(f'-- {side}')
    for vname, (md, me) in VARIANTS.items():
        ok = sum(expected(src[k][key], md, me) == v2_by[k][fld].split() for k in matched)
        print(f'   {vname:16s} 정확 일치 {ok:,} ({ok/max(len(matched),1)*100:.1f}%)')
    eq = more = less = with_star = star_tok = sil_tok = 0
    unk = Counter()
    for k in matched:
        ns = interior_star(src[k][key]); sv = v2_by[k][fld].split().count('sil')
        with_star += ns > 0; star_tok += ns; sil_tok += sv
        if sv == ns: eq += 1
        elif sv > ns: more += 1
        else: less += 1
        for t in expected(src[k][key], True, True):
            if t.startswith('<UNK'): unk[t] += 1
    n = max(len(matched), 1)
    print(f'   원본 내부 *** 있는 항목 {with_star:,} ({with_star/n*100:.1f}%) | 내부 *** 토큰 {star_tok:,} | v2 묵음 토큰 {sil_tok:,}')
    print(f'   항목별 v2 묵음 수 = 원본 내부 *** 수: {eq:,} ({eq/n*100:.1f}%) | v2가 더 많음 {more:,} | 더 적음 {less:,}')
    if unk: print(f'   매핑표 밖 원본 기호 상위: {unk.most_common(8)}')
    shown = 0
    for k in matched:
        e = expected(src[k][key], True, True); a = v2_by[k][fld].split()
        if e != a and shown < 2:
            shown += 1
            print(f'   불일치 예 {k}\n     원본: {" ".join(src[k][key])}\n     V2기대: {" ".join(e)}\n     서버  : {" ".join(a)}')

print('\n=== (c) 로컬 산출 ===')
for name in ['train', 'val']:
    d = loc[name]; m = [k for k in d if k in src]
    para = [k for k in m if src[k]['n_sent'] > 1]
    print(f'aihub_{name}: 항목 {len(d):,} | 원본 매칭 {len(m):,} | 문단 파일 항목 {len(para):,} ({len(para)/max(len(d),1)*100:.1f}%), 시간 {sum(d[k]["duration"] for k in para)/3600:.1f}h / 전체 {sum(v["duration"] for v in d.values())/3600:.1f}h')
print(f'train·val 키 중복: {len(set(loc["train"]) & set(loc["val"])):,}')
