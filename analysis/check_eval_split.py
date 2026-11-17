"""성능표(2026. 6. 8. 공유)가 어느 분할(test / val / train)에서 나올 수 있는 수치인지 라벨만으로 검사 (졸업논문 5.4, 표 14)
사용법: python check_eval_split.py test.json val.json train.json

모델·GPU·음성 없이 분할 JSON만 있으면 된다.

원리
  팀 평가 코드(EchoModel, echo/evaluation/metrics.py 의 calculate_mdd_metrics)는
    - 두 정렬 필드(canonical_aligned, perceived_aligned)의 길이가 다른 발화를 건너뛰고
    - 모델 출력을 perceived_aligned 에 편집 거리로 맞춘 뒤, 자리마다
        양쪽이 모두 묵음이면 세지 않고,
        정발음 != 실제 발음이면 '오류 자리', 같으면 '정상 자리'로 본다.
    - 모델이 perceived_aligned 에 대응 자리가 없는 음소를 더 내면(삽입) 오탐(FR)으로 센다.
  오류 자리 수를 E, 정상 자리 수를 C, 모델 삽입 수를 I 라 하면
        재현율 R = TR / E,   FRR = FR / (C + I),   정밀도 P = TR / (TR + FR)
  이므로
        E / (C + I) = P * FRR / (R * (1 - P))                      ... (식 1)
  E 와 C 는 라벨만으로 셀 수 있고, 오른쪽은 표의 수치다. 따라서 분할마다
        I = E / (식 1의 오른쪽) - C
  를 구할 수 있다. I 는 모델이 낸 삽입의 개수이므로
        (조건 A) I >= 0
        (조건 B) I <= PER 을 계산한 편집 거리의 합 = PER * (묵음을 뺀 실제 발음 음소 수)
  를 만족해야 한다. 만족하지 못하는 분할에서는 표의 수치가 나올 수 없다.
  두 조건을 모두 만족하는 분할이 하나뿐이면 그 분할이 표의 출처다.

  표의 수치는 소수 넷째 자리까지라서 반올림 폭(±0.00005)을 구간으로 함께 계산한다.

전제와 한계
  - 표의 수치가 분할 하나 전체를 위 평가 코드로 평가한 결과라고 본다.
  - 어느 체크포인트를 평가했는지는 이 방법으로 알 수 없다.
"""
import itertools
import json
import sys

SILENCE = {'sil', 'sp', 'spn', 'pau', ''}          # echo/constants.py 의 SILENCE_TOKENS
HALF = 0.00005                                     # 소수 넷째 자리 반올림 폭

# 2026. 6. 8. 팀 대화방에 공유된 표 (PER, F1, 정밀도, 재현율, FRR, FAR)
TABLE = {
    'baseline': dict(per=0.1593, f1=0.7650, p=0.7861, r=0.7450, frr=0.0583, far=0.2550),
    'FiLM-128': dict(per=0.1546, f1=0.7615, p=0.8076, r=0.7205, frr=0.0494, far=0.2795),
}


def count_split(path):
    """평가 코드와 같은 규칙으로 분할 하나의 E, C 등을 센다."""
    data = json.load(open(path, encoding='utf-8'))
    st = dict(n=len(data), n_used=0, n_skipped=0, E=0, C=0, both_sil=0, n_ref=0, target_mismatch=0,
              by_source={})
    for item in data.values():
        can = (item.get('canonical_aligned') or '').split()
        per = (item.get('perceived_aligned') or '').split()
        # PER 의 기준 음소 수: 학습 목표(실제 발음)에서 묵음을 뺀 것
        ref = [t for t in (item.get('perceived_train_target') or '').split() if t.lower() not in SILENCE]
        st['n_ref'] += len(ref)
        if ref != [t for t in per if t.lower() not in SILENCE]:
            st['target_mismatch'] += 1               # 학습 목표와 정렬 필드의 음소열이 다른 발화(0 이어야 조건 B 가 엄밀)
        src = st['by_source'].setdefault(item.get('source', '?'), dict(n=0, n_used=0, E=0, C=0))
        src['n'] += 1
        if not can or not per or len(can) != len(per):
            st['n_skipped'] += 1                     # calculate_mdd_metrics 가 건너뛰는 발화
            continue
        st['n_used'] += 1
        src['n_used'] += 1
        for c, p in zip(can, per):
            if c.lower() in SILENCE and p.lower() in SILENCE:
                st['both_sil'] += 1
            elif c != p:
                st['E'] += 1
                src['E'] += 1
            else:
                st['C'] += 1
                src['C'] += 1
    return st


def rho_interval(m):
    """식 1의 오른쪽 값과, 반올림 폭을 고려한 최소·최대."""
    def rho(p, r, frr):
        return p * frr / (r * (1 - p))
    vals = [rho(m['p'] + a * HALF, m['r'] + b * HALF, m['frr'] + c * HALF)
            for a, b, c in itertools.product((-1, 1), repeat=3)]
    return rho(m['p'], m['r'], m['frr']), min(vals), max(vals)


def main(paths):
    print('표에서 계산한 E/(C+I)')
    rhos = {}
    for name, m in TABLE.items():
        mid, lo, hi = rho_interval(m)
        rhos[name] = (mid, lo, hi)
        print(f'  {name:9s} {mid:.5f}  (반올림 구간 {lo:.5f} ~ {hi:.5f})')
    print()

    for path in paths:
        st = count_split(path)
        E, C = st['E'], st['C']
        print(f'[{path}] 발화 {st["n"]:,} | 집계 대상 {st["n_used"]:,} | 길이 불일치로 제외 {st["n_skipped"]:,}')
        print(f'  오류 자리 E = {E:,} | 정상 자리 C = {C:,} | E/C = {E / C:.5f} | 양쪽 묵음(미집계) {st["both_sil"]:,}')
        print(f'  PER 기준 음소 수 = {st["n_ref"]:,} | 학습 목표와 정렬 필드가 다른 발화 = {st["target_mismatch"]:,}')
        for s, v in sorted(st['by_source'].items()):
            ec = f'{v["E"] / v["C"]:.5f}' if v['C'] else '-'
            print(f'    {s:9s} 발화 {v["n"]:,} (집계 {v["n_used"]:,}) | E {v["E"]:,} | C {v["C"]:,} | E/C {ec}')
        ok_all = True
        for name, m in TABLE.items():
            mid, lo, hi = rhos[name]
            i_mid, i_lo, i_hi = E / mid - C, E / hi - C, E / lo - C          # 모델 삽입 수와 그 구간
            per_err_max = (m['per'] + HALF) * st['n_ref']                    # PER 편집 거리 합의 상한
            cond_a = i_hi >= 0
            cond_b = i_lo <= per_err_max
            ok = cond_a and cond_b
            ok_all = ok_all and ok
            share = i_mid / (m['per'] * st['n_ref'])
            print(f'  {name:9s} 모델 삽입 수 I = {i_mid:,.0f} ({i_lo:,.0f} ~ {i_hi:,.0f})'
                  f' | 정상 자리 대비 {i_mid / C:+.2%} | PER 오류 {m["per"] * st["n_ref"]:,.0f}개 중 {share:+.1%}'
                  f' | 조건 A {"만족" if cond_a else "위반"} · 조건 B {"만족" if cond_b else "위반"}')
        print(f'  => {"이 분할에서 표의 수치가 나올 수 있음" if ok_all else "이 분할에서는 표의 수치가 나올 수 없음"}')
        print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
