# COMP 3105A A1 - 실험 실행 + 리포트용 표 출력
#
# 이 파일은 제출하지 않는다.
# synRegExperiments / runCCS / synClsExperiments / runBCW 를 실행하고
# 결과를 리포트에 바로 붙여넣을 수 있는 형태(텍스트 / Markdown / LaTeX)로 찍어준다.
# 구현 안 된 함수는 알아서 건너뛴다.
#
# 실행법 (A1files 폴더에서):
#     python A1run_experiments.py           <- 구현된 것 전부 실행
#     python A1run_experiments.py synreg    <- Q1(d) 만
#     python A1run_experiments.py ccs       <- Q1(e) 만
#     python A1run_experiments.py syncls    <- Q2(c) 만
#     python A1run_experiments.py bcw       <- Q2(d) 만

import os
import sys
import time

import numpy as np

import JaeyoonLee.A1codes as A1codes


HERE = os.path.dirname(os.path.abspath(__file__))
CCS_FOLDER = os.path.abspath(os.path.join(HERE, 'data', 'CCS'))
BCW_FOLDER = os.path.abspath(os.path.join(HERE, 'data', 'BCW'))

DEC = 8   # 리포트에 쓸 소수점 자리수


# ---------------------------------------------------------------- 표 출력 유틸

def _fmt(v):
    return f'{float(v):.{DEC}f}'


def printTable(matrix, row_labels, col_labels, title):
    """하나의 표를 텍스트 / Markdown / LaTeX 세 가지 형태로 출력."""
    M = np.asarray(matrix, dtype=float)
    rl = [str(r) for r in row_labels]
    cl = [str(c) for c in col_labels]

    print(f'\n{title}')
    print('=' * len(title))

    # --- 1) 콘솔에서 읽기 좋은 정렬 텍스트
    w0 = max(len(s) for s in rl + ['Model'])
    ws = [max(len(cl[j]), DEC + 6) for j in range(len(cl))]
    header = 'Model'.ljust(w0) + '  ' + '  '.join(
        cl[j].rjust(ws[j]) for j in range(len(cl)))
    print('\n' + header)
    print('-' * len(header))
    for i, name in enumerate(rl):
        print(name.ljust(w0) + '  ' + '  '.join(
            _fmt(M[i, j]).rjust(ws[j]) for j in range(len(cl))))

    # --- 2) Markdown (Word / Notion 에 붙여넣기)
    print('\n--- Markdown ---')
    print('| Model | ' + ' | '.join(cl) + ' |')
    print('|' + '---|' * (len(cl) + 1))
    for i, name in enumerate(rl):
        print(f'| {name} | ' + ' | '.join(_fmt(M[i, j])
                                          for j in range(len(cl))) + ' |')

    # --- 3) LaTeX (Overleaf 에 붙여넣기)
    print('\n--- LaTeX ---')
    print('\\begin{table}[h]')
    print('\\centering')
    print('\\begin{tabular}{l' + 'r' * len(cl) + '}')
    print('\\hline')
    print('Model & ' + ' & '.join(cl) + ' \\\\')
    print('\\hline')
    for i, name in enumerate(rl):
        print(f'{name} & ' + ' & '.join(_fmt(M[i, j])
                                        for j in range(len(cl))) + ' \\\\')
    print('\\hline')
    print('\\end{tabular}')
    print(f'\\caption{{{title}}}')
    print('\\end{table}')


def printAccTable(M, title):
    """Q2(c) 전용 4x3 정확도 표 (열마다 하이퍼파라미터 값이 다름)."""
    M = np.asarray(M, dtype=float)
    ms = (10, 50, 100, 200)
    d1 = (1, 2, 4, 8)
    d2 = (1, 2, 4, 8)

    print(f'\n{title}')
    print('=' * len(title))

    print('\n     m   Accuracy |  dim1   Accuracy |  dim2   Accuracy')
    print('-' * 56)
    for i in range(4):
        print(f'  {ms[i]:>4}   {_fmt(M[i,0]):>8} | '
              f'{d1[i]:>5}   {_fmt(M[i,1]):>8} | '
              f'{d2[i]:>5}   {_fmt(M[i,2]):>8}')

    print('\n--- Markdown ---')
    print('| m | Accuracy | dim1 | Accuracy | dim2 | Accuracy |')
    print('|---|---|---|---|---|---|')
    for i in range(4):
        print(f'| {ms[i]} | {_fmt(M[i,0])} | {d1[i]} | {_fmt(M[i,1])} '
              f'| {d2[i]} | {_fmt(M[i,2])} |')

    print('\n--- LaTeX ---')
    print('\\begin{tabular}{rrrrrr}')
    print('\\hline')
    print('$m$ & Accuracy & dim1 & Accuracy & dim2 & Accuracy \\\\')
    print('\\hline')
    for i in range(4):
        print(f'{ms[i]} & {_fmt(M[i,0])} & {d1[i]} & {_fmt(M[i,1])} '
              f'& {d2[i]} & {_fmt(M[i,2])} \\\\')
    print('\\hline')
    print('\\end{tabular}')


# ---------------------------------------------------------------- 자동 점검

def sanityCheckLossTables(train, test, tag):
    """손실표가 말이 되는지 자동 점검. 실패하면 구현을 다시 봐야 한다."""
    print(f'\n### {tag} 자동 점검 ###')
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    names = ['L2', 'L1', 'Linf']
    ok = True

    if train.shape != (3, 3) or test.shape != (3, 3):
        print(f'  [FAIL] shape 이 (3,3) 이 아니다: '
              f'train={train.shape}, test={test.shape}')
        return False

    if not (np.all(np.isfinite(train)) and np.all(np.isfinite(test))):
        print('  [FAIL] NaN 또는 inf 가 들어 있다.')
        ok = False

    if np.any(train < 0) or np.any(test < 0):
        print('  [FAIL] 음수 손실이 있다. 손실은 항상 >= 0 이다.')
        ok = False

    # 핵심 점검: 학습 손실표의 대각선은 각 열의 최솟값이어야 한다.
    # (각 모델이 바로 그 손실을 최소화하도록 학습했으므로 수학적으로 보장된다)
    for j in range(3):
        col = train[:, j]
        if abs(col[j] - col.min()) > 1e-9:
            print(f'  [FAIL] 학습 {names[j]} loss 열의 최솟값이 '
                  f'{names[j]} 모델이 아니다 (최소는 {names[col.argmin()]} 모델).')
            print(f'         -> minimize{names[j]} 구현 또는 {names[j]} 손실 '
                  f'계산식을 다시 볼 것.')
            ok = False
    if ok:
        print('  [OK  ] 학습 손실표 대각선이 각 열의 최솟값이다.')

    # 참고 정보 (실패가 아니라 관찰거리)
    diag_test_ok = all(abs(test[j, j] - test[:, j].min()) < 1e-9
                       for j in range(3))
    if diag_test_ok:
        print('  [info] 테스트 손실표도 대각선이 최소다.')
    else:
        print('  [info] 테스트 손실표는 대각선이 최소가 아니다. '
              '-> (d.2) 분석에서 다룰 좋은 소재다.')

    if np.all(train[:, 2] > train[:, 1]):
        print('  [OK  ] Linf loss 가 L1 loss 보다 크다 (최댓값 vs 평균, 정상).')

    gap = test.mean() - train.mean()
    print(f'  [info] 평균 테스트손실 - 평균 학습손실 = {gap:+.4f}')
    return ok


def sanityCheckAccTables(train, test, tag):
    print(f'\n### {tag} 자동 점검 ###')
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    ok = True

    if train.shape != (4, 3) or test.shape != (4, 3):
        print(f'  [FAIL] shape 이 (4,3) 이 아니다: '
              f'train={train.shape}, test={test.shape}')
        return False

    if np.any(train < 0) or np.any(train > 1) or np.any(test < 0) or np.any(test > 1):
        print('  [FAIL] 정확도가 [0, 1] 범위를 벗어난다. '
              '-> 라벨 shape 브로드캐스팅 사고를 의심할 것.')
        ok = False

    if np.all(test < 0.6):
        print('  [FAIL] 테스트 정확도가 전부 0.6 미만이다. '
              '-> 예측 라벨 계산 또는 find_opt 를 의심할 것.')
        ok = False

    if np.all(train >= test - 1e-9):
        print('  [OK  ] 모든 설정에서 학습 정확도 >= 테스트 정확도.')
    else:
        print('  [info] 일부 설정에서 테스트 정확도가 학습보다 높다 '
              '(작은 m 에서는 가능하다).')

    print(f'  [info] m 열 테스트 정확도 추이: '
          f'{[round(float(v), 4) for v in test[:, 0]]}')
    print(f'  [info] dim1 열 테스트 정확도 추이: '
          f'{[round(float(v), 4) for v in test[:, 1]]}')
    print(f'  [info] dim2 열 테스트 정확도 추이: '
          f'{[round(float(v), 4) for v in test[:, 2]]}')
    return ok


# ---------------------------------------------------------------- 실험 실행

ROW_LABELS = ['L2 model', 'L1 model', 'Linf model']
COL_LABELS = ['L2 loss', 'L1 loss', 'Linf loss']


def _run(name, fn, *args):
    print(f'\n{"#" * 70}')
    print(f'# {name}')
    print(f'{"#" * 70}')
    t0 = time.time()
    try:
        out = fn(*args)
    except NotImplementedError:
        print('  [SKIP] 아직 구현되지 않았다.')
        return None
    except FileNotFoundError as e:
        print(f'  [SKIP] 데이터 파일을 찾을 수 없다: {e}')
        return None
    except Exception as e:
        import traceback
        print(f'  [ERR ] {type(e).__name__}: {e}')
        traceback.print_exc()
        return None
    print(f'  (소요 시간 {time.time() - t0:.1f}초)')
    return out


def runSynReg():
    out = _run('Q1(d) synRegExperiments  ->  Table 1, Table 2',
               A1codes.synRegExperiments)
    if out is None:
        return
    train, test = out
    printTable(train, ROW_LABELS, COL_LABELS,
               'Table 1: Different training losses for different models')
    printTable(test, ROW_LABELS, COL_LABELS,
               'Table 2: Different test losses for different models')
    sanityCheckLossTables(train, test, 'Q1(d)')


def runCCSExp():
    out = _run(f'Q1(e) runCCS  ->  Table 3, Table 4\n#   data: {CCS_FOLDER}',
               A1codes.runCCS, CCS_FOLDER)
    if out is None:
        return
    train, test = out
    printTable(train, ROW_LABELS, COL_LABELS,
               'Table 3: Training losses on the CCS dataset')
    printTable(test, ROW_LABELS, COL_LABELS,
               'Table 4: Test losses on the CCS dataset')
    sanityCheckLossTables(train, test, 'Q1(e)')


def runSynCls():
    out = _run('Q2(c) synClsExperiments  ->  Table 5, Table 6',
               A1codes.synClsExperiments)
    if out is None:
        return
    train, test = out
    printAccTable(train, 'Table 5: Training accuracies with different hyper-parameters')
    printAccTable(test, 'Table 6: Test accuracies with different hyper-parameters')
    sanityCheckAccTables(train, test, 'Q2(c)')


def runBCWExp():
    out = _run(f'Q2(d) runBCW\n#   data: {BCW_FOLDER}',
               A1codes.runBCW, BCW_FOLDER)
    if out is None:
        return
    train_acc, test_acc = out
    print(f'\n  Average training accuracy : {float(train_acc):.{DEC}f}')
    print(f'  Average test accuracy     : {float(test_acc):.{DEC}f}')
    if not (0 <= float(test_acc) <= 1):
        print('  [FAIL] 정확도가 [0,1] 범위 밖이다.')
    elif float(test_acc) < 0.85:
        print('  [warn] BCW 는 선형분리가 잘 되는 데이터라 보통 0.9 이상 나온다. '
              '전처리를 다시 확인해볼 것.')
    else:
        print('  [OK  ] 합리적인 범위의 정확도다.')


TASKS = {
    'synreg': runSynReg,
    'ccs':    runCCSExp,
    'syncls': runSynCls,
    'bcw':    runBCWExp,
}


if __name__ == '__main__':
    np.set_printoptions(precision=DEC, suppress=True)
    which = sys.argv[1:] or list(TASKS)
    for key in which:
        if key not in TASKS:
            print(f'알 수 없는 인자: {key}  (가능: {", ".join(TASKS)})')
            continue
        TASKS[key]()
    print('\n끝. 위 표를 리포트에 옮기면 된다.')
