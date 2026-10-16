"""ECHO 최종 학습 데이터셋(train/val/test.json) 기술 통계 산출 스크립트 (졸업논문 5.2용)"""
import json, sys
from collections import Counter, defaultdict
from rapidfuzz.distance import Levenshtein

VALID = {'sil','aa','ae','ah','ao','aw','ay','b','ch','d','dh','eh','er','ey','f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh'}
L1 = {'ABA':'Arabic','SKA':'Arabic','YBAA':'Arabic','ZHAA':'Arabic',
      'BWC':'Mandarin','LXC':'Mandarin','NCC':'Mandarin','TXHC':'Mandarin',
      'ASI':'Hindi','RRBI':'Hindi','SVBI':'Hindi','TNI':'Hindi',
      'HJK':'Korean','HKK':'Korean','YDCK':'Korean','YKWK':'Korean',
      'EBVS':'Spanish','ERMS':'Spanish','MBMPS':'Spanish','NJS':'Spanish',
      'HQTV':'Vietnamese','PNV':'Vietnamese','THV':'Vietnamese','TLV':'Vietnamese'}
# 주의: L2-ARCTIC 한국어 화자는 HJK, HKK, YDCK, YKWK (논문 요약본의 SKA 표기는 오기 — SKA는 아랍어 화자)

src_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
data = {}
for split in ['train','val','test']:
    d = json.load(open(f'{src_dir}/{split}.json', encoding='utf-8'))
    for k,v in d.items():
        v['_split'] = split; v['_src'] = 'L2-ARCTIC' if k.startswith('l2_') else 'AI-Hub'
        data[k] = v

def strip_sil(seq): return [p for p in seq if p != 'sil']

R = {}
# A. 규모
A = defaultdict(lambda: {'n':0,'dur':0.0,'spk':set()})
for v in data.values():
    for key in [(v['_split'],v['_src']),(v['_split'],'ALL'),('ALL',v['_src']),('ALL','ALL')]:
        A[key]['n']+=1; A[key]['dur']+=v['duration']; A[key]['spk'].add(v['spk_id'])
R['scale'] = {f'{s}|{o}':{'utts':x['n'],'hours':round(x['dur']/3600,1),'speakers':len(x['spk'])} for (s,o),x in A.items()}

# B. L2-ARCTIC 화자·L1별 분할
B = defaultdict(Counter)
for v in data.values():
    if v['_src']=='L2-ARCTIC': B[v['spk_id']][v['_split']]+=1
R['l2_speaker_split'] = {spk:{'L1':L1.get(spk,'?'), **dict(c), 'total':sum(c.values())} for spk,c in sorted(B.items())}
R['l2_L1_split'] = {}
for spk,c in B.items():
    l1=L1.get(spk,'?'); R['l2_L1_split'].setdefault(l1,Counter())
    for s,n in c.items(): R['l2_L1_split'][l1][s]+=n
R['l2_L1_split'] = {k:{**dict(v),'total':sum(v.values())} for k,v in R['l2_L1_split'].items()}

# C. 발화 길이 / D. 음소 빈도·커버리지 / E. 오류 통계
C = defaultdict(lambda: {'n':0,'dur':0.0,'ph':0,'wd':0,'ph_max':0,'dur_max':0.0,'over30':0})
canon_freq = defaultdict(Counter); perc_freq = defaultdict(Counter)
E = defaultdict(lambda: {'utts':0,'correct_utts':0,'canon_phones':0,'S':0,'D':0,'I':0,'err_utts_phones':0})
subs = defaultdict(Counter); dels = defaultdict(Counter); ins = defaultdict(Counter)
bad_tokens = Counter()
for k,v in data.items():
    src=v['_src']; c=v['canonical_train_target'].split(); p=v['perceived_train_target'].split()
    for t in c+p:
        if t not in VALID: bad_tokens[t]+=1
    for g in [src,'ALL']:
        x=C[g]; x['n']+=1; x['dur']+=v['duration']; x['ph']+=len(strip_sil(c)); x['wd']+=len(v['wrd'].split())
        x['ph_max']=max(x['ph_max'],len(strip_sil(c))); x['dur_max']=max(x['dur_max'],v['duration']); x['over30']+= v['duration']>30
        canon_freq[g].update(c); perc_freq[g].update(p)
    cs, ps = strip_sil(c), strip_sil(p)
    groups=[src,'ALL']
    if src=='L2-ARCTIC' and L1.get(v['spk_id'])=='Korean': groups.append('L2-ARCTIC-Korean')
    ops = Levenshtein.editops(cs, ps)
    for g in groups:
        e=E[g]; e['utts']+=1; e['canon_phones']+=len(cs)
        if cs==ps: e['correct_utts']+=1
        for op in ops:
            if op.tag=='replace': e['S']+=1; subs[g][(cs[op.src_pos],ps[op.dest_pos])]+=1
            elif op.tag=='delete': e['D']+=1; dels[g][cs[op.src_pos]]+=1
            else: e['I']+=1; ins[g][ps[op.dest_pos]]+=1
R['bad_tokens']=dict(bad_tokens)
R['length']={g:{'utts':x['n'],'mean_dur_s':round(x['dur']/x['n'],2),'max_dur_s':round(x['dur_max'],1),'over30s':x['over30'],
                'mean_phones':round(x['ph']/x['n'],1),'max_phones':x['ph_max'],'mean_words':round(x['wd']/x['n'],1)} for g,x in C.items()}
R['phone_coverage']={g:{'canonical_types':len([t for t in canon_freq[g] if t!='sil']),'perceived_types':len([t for t in perc_freq[g] if t!='sil'])} for g in canon_freq}
R['phone_freq_canonical']={g:dict(canon_freq[g].most_common()) for g in canon_freq}
R['phone_freq_perceived']={g:dict(perc_freq[g].most_common()) for g in perc_freq}
R['errors']={}
for g,e in E.items():
    tot=e['S']+e['D']+e['I']
    R['errors'][g]={**e,'correct_utt_ratio':round(e['correct_utts']/e['utts']*100,1),'error_utts':e['utts']-e['correct_utts'],
                    'total_errors':tot,'PER_%':round(tot/e['canon_phones']*100,2),
                    'S_%':round(e['S']/tot*100,1) if tot else 0,'D_%':round(e['D']/tot*100,1) if tot else 0,'I_%':round(e['I']/tot*100,1) if tot else 0,
                    'errors_per_utt':round(tot/e['utts'],2)}
R['top_substitutions']={g:[{'canonical':a,'perceived':b,'count':n} for (a,b),n in subs[g].most_common(15)] for g in subs}
R['top_deletions']={g:[{'phone':a,'count':n} for a,n in dels[g].most_common(8)] for g in dels}
R['top_insertions']={g:[{'phone':a,'count':n} for a,n in ins[g].most_common(8)] for g in ins}
json.dump(R, open(f'{src_dir}/dataset_stats.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)

# 요약 출력
print('== 규모 (split|원천: 발화, 시간h, 화자)')
for k in ['train|L2-ARCTIC','train|AI-Hub','train|ALL','val|L2-ARCTIC','val|AI-Hub','val|ALL','test|L2-ARCTIC','test|AI-Hub','test|ALL','ALL|L2-ARCTIC','ALL|AI-Hub','ALL|ALL']:
    print(f"  {k:18s} {R['scale'][k]}")
print('== 비표준 토큰:', R['bad_tokens'])
print('== 길이:'); [print(' ',g,x) for g,x in R['length'].items()]
print('== 음소 커버리지:', R['phone_coverage'])
print('== L1별 분할:'); [print(' ',k,v) for k,v in R['l2_L1_split'].items()]
print('== 한국어 화자:'); [print(' ',s,v) for s,v in R['l2_speaker_split'].items() if v['L1']=='Korean']
print('== 오류 통계:'); [print(' ',g,{k:v for k,v in x.items()}) for g,x in R['errors'].items()]
for g in ['ALL','L2-ARCTIC','AI-Hub','L2-ARCTIC-Korean']:
    print(f'== 대치 상위 ({g}):', [(d['canonical'],d['perceived'],d['count']) for d in R['top_substitutions'][g][:12]])
    print(f'   삭제 상위: {[(d["phone"],d["count"]) for d in R["top_deletions"][g][:6]]} | 삽입 상위: {[(d["phone"],d["count"]) for d in R["top_insertions"][g][:6]]}')
