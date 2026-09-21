# COMP 3105A Introduction to Machine Learning - Fall 2026
# Assignment 1
# Name: Jaeyoon Lee
# Student ID:  <학번>
#
# ============================================================================
# 중요: 이 파일에는 import / 상수 할당 / 함수 정의 외에 top-level 문장을
#       절대 넣지 말 것. 함수 호출, 데이터 로딩, 실험 실행, print 금지.
#       테스트는 별도 파일(A1local_test.py)에서 할 것.
# ============================================================================

import os

import numpy as np
import cvxopt
from cvxopt import solvers
from scipy.optimize import minimize

# 솔버 출력 억제 (상수 할당이므로 top-level 허용됨)
solvers.options['show_progress'] = False


# ============================================================================
# Question 1 - Linear Regression
# ============================================================================

def minimizeL2(X, y):
    """
    w = argmin_w  (1 / 2n) * ||Xw - y||_2^2  =  (X^T X)^{-1} X^T y
    Args:
        X: (n, d)
        y: (n, 1)
    Returns:
        w: (d, 1)
    """
    # TODO: 구현
    XT = X.T @ X # XT: (d, d)
    yT = X.T @ y # yT: (d, 1)
    w = np.linalg.solve(XT, yT) # w: (d, 1)
    return w

def minimizeL1(X, y):
    """
    w = argmin_w  (1/n) * ||Xw - y||_1

    LP (joint variable u = [w; delta] in R^{d+n}):
        min_{w,delta}  delta^T 1_n
        s.t.           -delta <= 0_n
                       Xw - y <= delta
                       y - Xw <= delta

    Args:
        X: (n, d)
        y: (n, 1)
    Returns:
        w: (d, 1)
    """
    # TODO: 구현
    # first d variables are w, second n variables are delta
    n, d = X.shape

    c = np.concatenate([np.zeros(d), np.ones(n)]) # c: (d+n, 1) / LINE 5: [BUG...], because [REASON...].

    G = np.concatenate([np.concatenate([np.zeros_like(X), -np.eye(n)], axis=1), # -delta ⪯ 0
                        np.concatenate([X, -np.eye(n)], axis=1), # Xw - y ⪯ delta
                        np.concatenate([-X, -np.eye(n)], axis=1) # y - Xw ⪯ delta / LINE 9: [BUG...], because [REASON...].
                       ]) # G: (3n, d+n)

    h = np.concatenate([np.zeros_like(y), y, -y])

    cvxopt.solvers.options["show_progress"] = False

    c = cvxopt.matrix(c.astype(float)) # cvxopt.matrix requires float64 dtype
    G = cvxopt.matrix(G.astype(float))
    h = cvxopt.matrix(h.astype(float))
    res = cvxopt.solvers.lp(c, G, h) # LINE 15: [BUG...], because [REASON...].
    w = res['x'][:d] # LINE 16: [BUG...], because [REASON...].
    return np.array(w)


def minimizeLinf(X, y):
    """
    Q1(c.5) [1%]  L-infinity 손실 회귀를 LP로 푼다.

        w = argmin_w  ||Xw - y||_inf

    LP 형태 (joint variable u = [w; delta] in R^{d+1}):
        min_{w,delta}  delta
        s.t.           delta >= 0
                       Xw - y <= delta * 1_n
                       y - Xw <= delta * 1_n

    Args:
        X: (n, d) 입력 행렬
        y: (n, 1) 타깃 벡터
    Returns:
        w: (d, 1) 가중치 벡터

    구현 메모:
      - 반드시 리포트의 (c.1)~(c.4) 유도를 먼저 끝내고 그 결과를 옮길 것.
        유도가 없으면 이 부분 점수도 받지 못한다.
      - 블록 shape:
            c:     (d+1, 1)
            G^(1): (1, d+1)     h^(1): (1, 1)
            G^(2): (n, d+1)     h^(2): (n, 1)
            G^(3): (n, d+1)     h^(3): (n, 1)
        -> G: (2n+1, d+1),  h: (2n+1, 1)
      - np.zeros / np.ones / np.concatenate 조합으로 블록을 쌓을 것.
      - cvxopt.matrix 변환 잊지 말 것.
    """
    # TODO: 구현
    raise NotImplementedError


def synRegExperiments():
    """
    Q1(d.1) [0.5%]  합성 데이터에서 세 모델을 50회 반복 평가.

    Returns:
        train_loss: (3, 3) 평균 학습 손실   [모델 3종] x [손실 3종]
        test_loss:  (3, 3) 평균 테스트 손실

        행 순서: L2 model, L1 model, Linf model
        열 순서: L2 loss, L1 loss, Linf loss

    구현 메모:
      - L2 손실 = (1/2n)||Xw - y||^2,  L1 손실 = (1/n)||Xw - y||_1
        Linf 손실 = max|Xw - y|  (Linf 는 n 으로 나누지 않음)
      - genData 의 y[0] *= -0.1 은 학습 데이터에만 적용되는 의도적 이상치다.
        (d.2) 분석에서 이게 핵심 논점이 된다.
    """

    def genData(n_points, is_training=False):
        """합성 데이터 생성 (과제 PDF 템플릿 그대로)."""
        X = np.random.randn(n_points, d)                              # 입력 행렬
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1)       # 절편항 추가
        y = X @ w_true + np.random.randn(n_points, 1) * noise         # 정답 라벨
        if is_training:
            y[0] *= -0.1                                              # 이상치 주입
        return X, y

    n_runs = 50
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 3, 3])   # n_runs x n_models x n_metrics
    test_loss = np.zeros([n_runs, 3, 3])    # n_runs x n_models x n_metrics

    # TODO: 아래 시드를 본인 학번으로 바꿀 것
    np.random.seed(42)

    for r in range(n_runs):

        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train, is_training=True)
        Xtest, ytest = genData(n_test, is_training=False)

        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # TODO: 세 모델의 학습 데이터 성능 평가 (각 모델마다 L2, L1, Linf 손실)
        #       -> train_loss[r] 에 저장

        # TODO: 세 모델의 테스트 데이터 성능 평가 (각 모델마다 L2, L1, Linf 손실)
        #       -> test_loss[r] 에 저장

    # TODO: 50회 평균 계산 (어느 축으로 평균낼지 주의)
    # TODO: (3, 3) train_loss 와 (3, 3) test_loss 반환
    raise NotImplementedError


def preprocessCCS(dataset_folder):
    """
    Q1(e.1) [1%]  Concrete Compressive Strength 데이터 전처리.

    Args:
        dataset_folder: Concrete_Data.xls 가 들어 있는 폴더의 절대 경로
                        (os.path.abspath 의 결과)
    Returns:
        X: (n, d) 원시 특징(raw feature) 행렬 — 절편항은 붙이지 않는다
        y: (n, 1) 라벨 벡터 (compressive strength)

    구현 메모:
      - pandas.read_excel 로 읽는다. .xls 라서 xlrd 패키지가 필요함.
      - os.path.join(dataset_folder, 'Concrete_Data.xls') 로 경로 조립.
        (절대 하드코딩된 내 컴퓨터 경로를 넣지 말 것 — 채점 환경에서 깨진다)
      - 마지막 열이 타깃(강도), 앞의 8개 열이 특징이다.
      - y 의 shape 이 (n, 1) 인지 확인. .values 는 (n,) 를 줄 수 있다.
      - 절편항 추가는 runCCS 에서 하므로 여기서는 하지 않는다.
    """
    # TODO: 구현
    raise NotImplementedError


def runCCS(dataset_folder):
    """
    Q1(e.2) [0.5%]  CCS 데이터에서 세 모델을 50회 반복 평가.

    Returns:
        train_loss: (3, 3) 평균 학습 손실
        test_loss:  (3, 3) 평균 테스트 손실
    """
    X, y = preprocessCCS(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)   # 절편항 추가

    n_runs = 50
    train_loss = np.zeros([n_runs, 3, 3])   # n_runs x n_models x n_metrics
    test_loss = np.zeros([n_runs, 3, 3])    # n_runs x n_models x n_metrics

    # TODO: 아래 시드를 본인 학번으로 바꿀 것
    np.random.seed(42)

    for r in range(n_runs):

        # TODO: 데이터를 무작위로 50% 학습 / 50% 테스트로 분할
        #       (sklearn 금지. np.random.permutation 으로 인덱스를 섞을 것)

        # TODO: L2, L1, Linf 손실로 세 모델 학습

        # TODO: 세 모델의 학습 데이터 성능 평가 -> train_loss[r]

        # TODO: 세 모델의 테스트 데이터 성능 평가 -> test_loss[r]

        pass

    # TODO: 50회 평균 계산
    # TODO: (3, 3) train_loss 와 (3, 3) test_loss 반환
    raise NotImplementedError


# ============================================================================
# Question 2 - Logistic Regression
# ============================================================================

def linearRegL2Obj(w, X, y):
    """
    Q2(a.1) [1%]  선형회귀 L2 목적함수 값.

        J(w) = (1 / 2n) * ||Xw - y||_2^2

    Args:
        w: (d, 1) 파라미터
        X: (n, d) 입력 행렬
        y: (n, 1) 라벨
    Returns:
        obj_val: 스칼라 (배열이 아니라 실수여야 scipy.optimize.minimize 가 잘 동작)
    """
    # TODO: 구현
    raise NotImplementedError


def linearRegL2Grad(w, X, y):
    """
    Q2(a.1) [1%]  위 목적함수의 해석적 그래디언트.

    Args:
        w: (d, 1) 파라미터
        X: (n, d) 입력 행렬
        y: (n, 1) 라벨
    Returns:
        gradient: (d, 1)

    구현 메모:
      - 손으로 미분한 뒤, autograd 의 grad 함수로 계산한 값과 비교해 검증할 것.
        (autograd 는 requirements.txt 에 이미 들어 있음)
    """
    # TODO: 구현
    raise NotImplementedError


def find_opt(obj_func, grad_func, X, y):
    """
    Q2(a.2) [1%]  scipy.optimize.minimize 로 볼록 최적화 문제를 푼다.

    Args:
        obj_func:  (w, X, y) -> 스칼라   (w 는 (d, 1))
        grad_func: (w, X, y) -> (d, 1)
        X: (n, d), y: (n, 1)
    Returns:
        w: (d, 1) 최적 파라미터

    구현 메모:
      - minimize 는 1-D 배열만 받는다. 우리 obj_func/grad_func 는 열벡터를
        받으므로 래퍼 안에서 reshape 변환이 필요하다.
      - grad 역시 (d, 1) 로 나오므로 minimize 에 넘기기 전에 1-D 로 펴야 한다.
      - 검증법: find_opt(linearRegL2Obj, linearRegL2Grad, X, y) 의 결과가
        minimizeL2(X, y) 와 거의 같아야 한다. 같으면 이 함수는 맞은 것.
    """
    d = X.shape[1]
    # TODO: 크기 d 의 1-D 랜덤 초기 파라미터 생성
    w_0 = None

    # TODO: w 하나만 인자로 받는 목적함수 `func` 정의
    # TODO: w 하나만 인자로 받는 그래디언트 함수 `gd` 정의

    # return minimize(func, w_0, jac=gd)['x'][:, None]
    raise NotImplementedError


def logisticRegObj(w, X, y):
    """
    Q2(b.3) [1%]  로지스틱 회귀 목적함수 (cross-entropy loss).

        J(w) = (1/n) * [ -y^T log(sigma(Xw)) - (1 - y)^T log(1 - sigma(Xw)) ]

    Args:
        w: (d, 1), X: (n, d), y: (n, 1)
    Returns:
        obj_val: 스칼라

    구현 메모:
      - sigmoid 를 직접 계산한 뒤 log 를 취하면 underflow 로 log(0) -> NaN.
      - (b.1), (b.2) 에서 유도한 항등식을 써서 np.logaddexp 로 바꿔 쓸 것.
            -log(sigma(z))     = np.logaddexp(0, -z)
            -log(1 - sigma(z)) = np.logaddexp(z, 0)
        이렇게 하면 sigmoid 자체를 계산하지 않고도 손실을 구할 수 있다.
    """
    # TODO: 구현
    raise NotImplementedError


def logisticRegGrad(w, X, y):
    """
    Q2(b.3) [1%]  로지스틱 회귀의 해석적 그래디언트.

        grad J(w) = (1/n) * X^T (sigma(Xw) - y)

    Args:
        w: (d, 1), X: (n, d), y: (n, 1)
    Returns:
        gradient: (d, 1)

    구현 메모:
      - 여기서는 sigma 를 실제로 계산해야 한다. 수치적으로 안정한 sigmoid
        구현을 쓸 것 (z 가 큰 음수일 때 exp(-z) 가 overflow 한다).
      - autograd 로 검증할 것.
    """
    # TODO: 구현
    raise NotImplementedError


def synClsExperiments():
    """
    Q2(c.1) [1%]  합성 이진분류 데이터에서 하이퍼파라미터별 정확도를 50회 평균.

    Returns:
        train_acc: (4, 3) 평균 학습 정확도
        test_acc:  (4, 3) 평균 테스트 정확도

        열 0: m     을 (10, 50, 100, 200) 으로 변화
        열 1: dim1  을 (1, 2, 4, 8) 로 변화      <- 판별에 유용한 차원
        열 2: dim2  를 (1, 2, 4, 8) 로 변화      <- 순수 노이즈 차원
    """

    def genData(n_points, dim1, dim2):
        """합성 데이터 생성 (과제 PDF 템플릿 그대로)."""
        c0 = np.ones([1, dim1])                              # class 0 중심
        c1 = -np.ones([1, dim1])                             # class 1 중심
        X0 = np.random.randn(n_points, dim1 + dim2)          # class 0 입력
        X0[:, :dim1] += c0
        X1 = np.random.randn(n_points, dim1 + dim2)          # class 1 입력
        X1[:, :dim1] += c1
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1)   # 절편항
        y = np.concatenate([np.zeros([n_points, 1]),
                            np.ones([n_points, 1])], axis=0)
        return X, y

    def runClsExp(m=100, dim1=2, dim2=2):
        """주어진 하이퍼파라미터로 분류 실험 1회 실행."""
        n_test = 1000
        Xtrain, ytrain = genData(m, dim1, dim2)
        Xtest, ytest = genData(n_test, dim1, dim2)

        w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # TODO: 학습 데이터의 예측 라벨 계산
        #       (Xw > 0 이면 1, 아니면 0. sigmoid(Xw) > 0.5 와 동치)
        ytrain_hat = None
        # TODO: 학습 정확도 계산
        train_acc = None

        # TODO: 테스트 데이터의 예측 라벨 계산
        ytest_hat = None
        # TODO: 테스트 정확도 계산
        test_acc = None

        return train_acc, test_acc

    n_runs = 50
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])

    # TODO: 아래 시드를 본인 학번으로 바꿀 것
    np.random.seed(42)

    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, dim1 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)
        for i, dim2 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)

    # TODO: 50회 평균 계산
    # TODO: (4, 3) train_acc 와 (4, 3) test_acc 반환
    raise NotImplementedError


def preprocessBCW(dataset_folder):
    """
    Q2(d.1) [1%]  Breast Cancer Wisconsin (Diagnostic) 데이터 전처리.

    Args:
        dataset_folder: wdbc.data 가 들어 있는 폴더의 절대 경로
    Returns:
        X: (n, d) 특징 행렬 — 절편항은 붙이지 않는다
        y: (n, 1) 라벨 벡터 (B -> 0, M -> 1)

    구현 메모:
      - wdbc.data 는 헤더가 없는 CSV 다. pandas.read_csv(..., header=None).
      - 열 0 은 ID -> 반드시 제거. 열 1 이 진단(B/M) -> 타깃.
        열 2 이후 30개가 특징.
      - y shape 이 (n, 1) 인지 확인.
    """
    # TODO: 구현
    raise NotImplementedError


def runBCW(dataset_folder):
    """
    Q2(d.2) [0.5%]  BCW 데이터에서 로지스틱 회귀를 50회 반복 평가.

    Returns:
        train_acc: 평균 학습 정확도 (스칼라)
        test_acc:  평균 테스트 정확도 (스칼라)
    """
    X, y = preprocessBCW(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)   # 절편항 추가

    n_runs = 50
    train_acc = np.zeros([n_runs])
    test_acc = np.zeros([n_runs])

    # TODO: 아래 시드를 본인 학번으로 바꿀 것
    np.random.seed(42)

    for r in range(n_runs):

        # TODO: 데이터를 무작위로 50% 학습 / 50% 테스트로 분할
        #       (홀수 개면 최대한 균등하게)
        Xtrain, ytrain, Xtest, ytest = None, None, None, None

        w = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # TODO: 학습 데이터 정확도 -> train_acc[r]

        # TODO: 테스트 데이터 정확도 -> test_acc[r]

    # TODO: 50회 평균 계산
    # TODO: 평균 학습 정확도와 평균 테스트 정확도 두 개를 반환
    raise NotImplementedError
