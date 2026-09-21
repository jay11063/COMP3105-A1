# COMP 3105A A1 - 로컬 검증용 스크립트
#
# 이 파일은 제출하지 않는다. A1codes.py 의 함수들이 과제 PDF에 적힌
# 기대값과 맞는지 확인하는 용도.
#
# 실행법:  python A1local_test.py
# (A1files 폴더 안, JaeyoonLee/ 폴더와 toy_data/ 폴더가 보이는 위치에서 실행)

import os
import numpy as np
import pandas as pd

import JaeyoonLee.A1codes as A1codes


HERE = os.path.dirname(os.path.abspath(__file__))
TOY = os.path.join(HERE, 'toy_data')


def _augment(X):
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def _loadReg(name):
    df = pd.read_csv(os.path.join(TOY, name))
    X = df[['x']].values
    y = df[['y']].values
    return _augment(X), y


def _loadCls(name):
    df = pd.read_csv(os.path.join(TOY, name))
    X = df[['x1', 'x2']].values
    y = df[['y']].values
    return _augment(X), y


def _losses(X, y, w):
    """(L2, L1, Linf) 손실을 반환. 정의는 과제 명세를 따를 것."""
    r = X @ w - y
    n = X.shape[0]
    return (
        float(np.sum(r ** 2) / (2 * n)),
        float(np.sum(np.abs(r)) / n),
        float(np.max(np.abs(r))),
    )


def _check(label, got, expected, tol=1e-3):
    ok = np.allclose(np.asarray(got, dtype=float).ravel(),
                     np.asarray(expected, dtype=float).ravel(), atol=tol)
    mark = 'OK  ' if ok else 'FAIL'
    print(f'  [{mark}] {label}')
    if not ok:
        print(f'         got      = {np.asarray(got).ravel()}')
        print(f'         expected = {np.asarray(expected).ravel()}')
    return ok


def testRegression():
    """Q1(a)(b)(c) 검증 - 과제 PDF의 Table 3, 4 기대값과 비교."""
    print('\n=== Q1: Regression (toy_data) ===')
    Xtr, ytr = _loadReg('regression_train.csv')
    Xte, yte = _loadReg('regression_test.csv')

    models = {
        'L2':   (A1codes.minimizeL2,   [-26.72481802, -1.1663904]),
        'L1':   (A1codes.minimizeL1,   [-26.88576639, -1.16315391]),
        'Linf': (A1codes.minimizeLinf, [-25.3365398,  -1.39556938]),
    }
    expected_train = {
        'L2':   [2.00305568, 1.56314076, 7.31711154],
        'L1':   [2.01278477, 1.55829245, 7.4682775],
        'Linf': [2.36347873, 1.72061687, 6.6215357],
    }
    expected_test = {
        'L2':   [1.66071948, 1.42055149, 5.58874693],
        'L1':   [1.62892803, 1.40563309, 5.45534197],
        'Linf': [1.9868478,  1.58480749, 5.02664782],
    }

    for name, (fn, w_exp) in models.items():
        try:
            w = fn(Xtr, ytr)
        except NotImplementedError:
            print(f'  [SKIP] minimize{name} - 아직 미구현')
            continue
        except Exception as e:
            print(f'  [ERR ] minimize{name} - {type(e).__name__}: {e}')
            continue

        print(f'  -- {name} model')
        if np.asarray(w).shape != (Xtr.shape[1], 1):
            print(f'         !! shape 경고: {np.asarray(w).shape}, '
                  f'기대 ({Xtr.shape[1]}, 1)')
        _check(f'{name}: w', w, w_exp)
        _check(f'{name}: train loss', _losses(Xtr, ytr, w), expected_train[name])
        _check(f'{name}: test loss',  _losses(Xte, yte, w), expected_test[name])


def testFindOpt():
    """Q2(a) 검증 - find_opt 결과가 minimizeL2 해석해와 일치해야 한다."""
    print('\n=== Q2(a): find_opt vs 해석해 ===')
    Xtr, ytr = _loadReg('regression_train.csv')
    try:
        w_analytic = A1codes.minimizeL2(Xtr, ytr)
        w_opt = A1codes.find_opt(A1codes.linearRegL2Obj,
                                 A1codes.linearRegL2Grad, Xtr, ytr)
    except NotImplementedError:
        print('  [SKIP] 아직 미구현')
        return
    _check('find_opt == minimizeL2', w_opt, w_analytic, tol=1e-2)


def testGradients():
    """해석적 그래디언트를 수치 미분과 비교 (autograd 없이도 동작)."""
    print('\n=== 그래디언트 수치 검증 ===')
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    X = _augment(X)
    d = X.shape[1]

    pairs = [
        ('linearReg', A1codes.linearRegL2Obj, A1codes.linearRegL2Grad,
         rng.standard_normal((40, 1))),
        ('logisticReg', A1codes.logisticRegObj, A1codes.logisticRegGrad,
         (rng.random((40, 1)) > 0.5).astype(float)),
    ]

    for name, obj, grad, y in pairs:
        w = rng.standard_normal((d, 1)) * 0.3
        try:
            g_analytic = np.asarray(grad(w, X, y), dtype=float).ravel()
            eps = 1e-6
            g_numeric = np.zeros(d)
            for i in range(d):
                wp, wm = w.copy(), w.copy()
                wp[i] += eps
                wm[i] -= eps
                g_numeric[i] = (float(obj(wp, X, y)) - float(obj(wm, X, y))) / (2 * eps)
        except NotImplementedError:
            print(f'  [SKIP] {name} - 아직 미구현')
            continue
        _check(f'{name} gradient', g_analytic, g_numeric, tol=1e-5)


def testClassification():
    """Q2(b)(c) 검증 - 과제 PDF의 기대 모델/정확도와 비교."""
    print('\n=== Q2: Classification (toy_data) ===')
    Xtr, ytr = _loadCls('classification_train.csv')
    Xte, yte = _loadCls('classification_test.csv')

    try:
        w = A1codes.find_opt(A1codes.logisticRegObj,
                             A1codes.logisticRegGrad, Xtr, ytr)
    except NotImplementedError:
        print('  [SKIP] 아직 미구현')
        return

    _check('w', w, [0.0318, -2.47, -2.45], tol=1e-2)

    acc_tr = float(np.mean(((Xtr @ w) > 0).astype(float) == ytr))
    acc_te = float(np.mean(((Xte @ w) > 0).astype(float) == yte))
    _check('train accuracy', acc_tr, 0.93, tol=5e-3)
    _check('test accuracy',  acc_te, 0.925, tol=5e-3)


if __name__ == '__main__':
    testRegression()
    testFindOpt()
    testGradients()
    testClassification()
    print('\n완료. FAIL 이 하나도 없으면 다음 단계로.')
