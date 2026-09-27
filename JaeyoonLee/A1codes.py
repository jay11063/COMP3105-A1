# COMP 3105A Introduction to Machine Learning - Fall 2026
# Assignment 1
# Name: Jaeyoon Lee
# Student ID:  <ID>
#
# ============================================================================
# 중요: 이 파일에는 import / 상수 할당 / 함수 정의 외에 top-level 문장을
#       절대 넣지 말 것. 함수 호출, 데이터 로딩, 실험 실행, print 금지.
#       테스트는 별도 파일(A1local_test.py)에서 할 것.
# ============================================================================

import os

import numpy as np
import pandas as pd
import cvxopt
from scipy.optimize import minimize

from StudentID import ID



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
    # X_T = X.T
    # A = X_T @ X # A = X^T X : (d, d)
    # b = X_T @ y # b = X^T y : (d, 1)
    # Aw = b 
    w = np.linalg.solve(X.T @ X, X.T @ y) # w: (d, 1)
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
    # first d variables are w, second n variables are delta
    n, d = X.shape

    # c^T = [0_d 1_n] : (d+n, 1)
    c = np.concatenate([np.zeros(d), np.ones(n)]) # LINE 5: [BUG...], because [REASON...].

    # G = [G^(1) = [0_(n*d) -I_n]
    #      G^(2) = [   X    -I_n]
    #      G^(3) = [  -X    -I_n]]
    # G : (3n, d+n)
    G = np.concatenate([np.concatenate([np.zeros_like(X), -np.eye(n)], axis=1), # -delta ⪯ 0
                        np.concatenate([X, -np.eye(n)], axis=1), # Xw - y ⪯ delta
                        np.concatenate([-X, -np.eye(n)], axis=1) # y - Xw ⪯ delta / LINE 9: [BUG...], because [REASON...].
                       ])

    # h = [h^(1) = 0_n
    #      h^(2) = y
    #      h^(3) = -y ]
    # h : (3n, 1)
    h = np.concatenate([np.zeros_like(y), y, -y])

    cvxopt.solvers.options["show_progress"] = False

    c = cvxopt.matrix(c)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    res = cvxopt.solvers.lp(c, G, h) # LINE 15: [BUG...], because [REASON...].
    w = res['x'][:d] # LINE 16: [BUG...], because [REASON...].
    return np.array(w)


def minimizeLinf(X, y):
    """
    w = argmin_w ||Xw - y||_inf

    LP (joint variable u = [w; delta] in R^{d+1}):
        min_{w,delta}  delta
        s.t.           delta >= 0
                       Xw - y <= delta * 1_n
                       y - Xw <= delta * 1_n

    Args:
        X: (n, d)
        y: (n, 1)
    Returns:
        w: (d, 1)
    """
    n, d = X.shape

    # c = [0_d 1] : (d+1, 1)
    c = np.concatenate([np.zeros(d), np.ones(1)])

    # G = [ G^(1) = [0_d^T, -1] : (1, d+1)
    #       G^(2) = [X, -1_n]   : (n, d+1)
    #       G^(3) = [-X, -1_n]  : (n, d+1) ]
    # G : (2n+1, d+1)
    G = np.concatenate([np.concatenate([np.zeros((1, d)), -np.ones((1, 1))], axis=1),   
                        np.concatenate([X, -np.ones((n, 1))], axis=1),                  
                        np.concatenate([-X, -np.ones((n, 1))], axis=1)                  
                       ])

    # h = [ h^(1) = 0
    #       h^(2) = y
    #       h^(3) = -y ]
    # h : (2n+1, 1)
    h = np.concatenate([np.zeros((1, 1)), y, -y]) # 

    cvxopt.solvers.options["show_progress"] = False

    c = cvxopt.matrix(c)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    res = cvxopt.solvers.lp(c, G, h)
    w = res['x'][:d]
    return np.array(w)


def synRegExperiments():
    def genData(n_points, is_training=False):
        '''
        This function generates synthetic data
        '''
        X = np.random.randn(n_points, d) # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
        if is_training:
            y[0] *= -0.1
        return X, y
    n_runs = 50
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics

    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(ID)
    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train, is_training=True)
        Xtest, ytest = genData(n_test, is_training=False)
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        weights = [w_L2, w_L1, w_Linf]
        
        for model_idx, w in enumerate(weights):
            train_err = Xtrain @ w - ytrain
            test_err = Xtest @ w - ytest
            # TODO: Evaluate the three models' performance (for each model,
            #       calculate the L2, L1 and L infinity losses on the training
            #       data). Save them to `train_loss`

            train_loss[r, model_idx, 0] = 0.5 * np.mean(train_err ** 2) # L2
            train_loss[r, model_idx, 1] = np.mean(np.abs(train_err))    # L1
            train_loss[r, model_idx, 2] = np.max(np.abs(train_err))     # Linf

            # TODO: Evaluate the three models' performance (for each model,
            #       calculate the L2, L1 and L infinity losses on the test
            #       data). Save them to `test_loss`
            test_loss[r, model_idx, 0] = 0.5 * np.mean(test_err ** 2) # L2
            test_loss[r, model_idx, 1] = np.mean(np.abs(test_err))    # L1
            test_loss[r, model_idx, 2] = np.max(np.abs(test_err))     # Linf

    # TODO: compute the average losses over runs
    train_loss_avg = np.mean(train_loss, axis=0)
    test_loss_avg = np.mean(test_loss, axis=0)

    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    return train_loss_avg, test_loss_avg
    


def preprocessCCS(dataset_folder):
    file_path = os.path.join(dataset_folder, 'Concrete_Data.xls')

    df = pd.read_excel(file_path)
    
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:,-1:].to_numpy()

    return X, y

def runCCS(dataset_folder):
    X, y = preprocessCCS(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment

    n_runs = 50
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics

    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(ID)
    for r in range(n_runs):
        # TODO: Randomly partition the dataset into two parts (50%
        #       training and 50% test) 
        permuted_indices = np.random.permutation(n)
        n_train = n//2

        train_idx = permuted_indices[:n_train]
        test_idx = permuted_indices[n_train:]

        Xtrain, ytrain = X[train_idx], y[train_idx]
        Xtest, ytest =  X[test_idx], y[test_idx]

        # TODO: Learn three different models from the training data
        #       using L2, L1 and L infinity losses
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        weights = [w_L2, w_L1, w_Linf]

        for model_idx, w in enumerate(weights):
            train_err = Xtrain @ w - ytrain
            test_err = Xtest @ w - ytest

            # TODO: Evaluate the three models' performance (for each model,
            #       calculate the L2, L1 and L infinity losses on the training
            #       data). Save them to `train_loss`
            train_loss[r, model_idx, 0] = 0.5 * np.mean(train_err ** 2) # L2
            train_loss[r, model_idx, 1] = np.mean(np.abs(train_err))    # L1
            train_loss[r, model_idx, 2] = np.max(np.abs(train_err))     # Linf

            # TODO: Evaluate the three models' performance (for each model,
            #       calculate the L2, L1 and L infinity losses on the test
            #       data). Save them to `test_loss`
            abs_test_err = np.abs(test_err)
            test_loss[r, model_idx, 0] = 0.5 * np.mean(test_err ** 2) # L2
            test_loss[r, model_idx, 1] = np.mean(abs_test_err)        # L1
            test_loss[r, model_idx, 2] = np.max(abs_test_err)         # Linf

    # TODO: compute the average losses over runs
    train_loss_avg = np.mean(train_loss, axis=0)
    test_loss_avg = np.mean(test_loss, axis=0)
    
    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    return train_loss_avg, test_loss_avg



# ============================================================================
# Question 2 - Logistic Regression
# ============================================================================

def linearRegL2Obj(w, X, y):
    """
    J(w) = (1 / 2n) * ||Xw - y||_2^2

    Args:
        w: (d, 1)
        X: (n, d)
        y: (n, 1)
    Returns:
        obj_val: scalar
    """
    obj_val = 0.5 * np.mean((X @ w - y)**2)
    return obj_val


def linearRegL2Grad(w, X, y):
    """
    ∇J(w) = (1/n) X^T (Xw - y)

    Args:
        w: (d, 1)
        X: (n, d)
        y: (n, 1)
    Returns:
        gradient: (d, 1)
    """
    n = X.shape[0]
    gradient = (1.0 / n) * X.T @ (X @ w - y)
    return gradient


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
    w_0 = np.random.randn(d)

    # TODO: w 하나만 인자로 받는 목적함수 `func` 정의
    def func(w):
        return obj_func(w[:, None], X, y)

    # TODO: w 하나만 인자로 받는 그래디언트 함수 `gd` 정의
    def gd(w):
        gradient = grad_func(w[:,None], X, y)
        return gradient.flatten()

    return minimize(func, w_0, jac=gd)['x'][:, None]


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
    n = X.shape[0]
    z = X @ w
    obj_val = (1.0 / n) * (y.T @ np.logaddexp(0,-z) + (np.ones((n, 1)) - y).T @ np.logaddexp(z, 0))
    return obj_val.item()


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
    n = X.shape[0]
    z = X @ w
    sigmoid_Xw = np.exp(-np.logaddexp(0,-z))
    grad = (1.0 / n) * X.T @ (sigmoid_Xw - y)
    return grad


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
        '''
        This function generates synthetic data
        '''
        c0 = np.ones([1, dim1]) # class 0 center
        c1 = -np.ones([1, dim1]) # class 1 center
        X0 = np.random.randn(n_points, dim1 + dim2) # class 0 input
        X0[:, :dim1] += c0
        X1 = np.random.randn(n_points, dim1 + dim2) # class 1 input
        X1[:, :dim1] += c1
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
        y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
        return X, y

    def runClsExp(m=100, dim1=2, dim2=2):
        '''
        Run classification experiment with the specified arguments
        '''
        n_test = 1000
        Xtrain, ytrain = genData(m, dim1, dim2)
        Xtest, ytest = genData(n_test, dim1, dim2)

        w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)
        ytrain_hat = (Xtrain @ w_logit > 0).astype(int) # TODO: Compute predicted labels of the training points
        train_acc = np.mean(ytrain_hat == ytrain) # TODO: Compute the accuracy of the training set
    
        ytest_hat = (Xtest @ w_logit > 0).astype(int) # TODO: Compute predicted labels of the test points
        test_acc = np.mean(ytest_hat == ytest) # TODO: Compute the accuracy of the test set

        return train_acc, test_acc

    n_runs = 50
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])

    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(ID)

    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, dim1 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)
        for i, dim2 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)


    # TODO: compute the average accuracies over runs
    # TODO: return a 4-by-3 training accuracy variable and a 4-by-3 test accuracy variable
    avg_train_acc = np.mean(train_acc, axis=0)
    avg_test_acc = np.mean(test_acc, axis=0)

    return avg_train_acc, avg_test_acc


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
    file_path = os.path.join(dataset_folder, 'wdbc.data')

    df = pd.read_csv(file_path, header=None)
    
    X = df.iloc[:, 2:].to_numpy()
    y = df.iloc[:,1:2].replace({'M' : 1, 'B' : 0}).to_numpy()

    return X, y

def runBCW(dataset_folder):
    """
    Q2(d.2) [0.5%]  BCW 데이터에서 로지스틱 회귀를 50회 반복 평가.

    Returns:
        train_acc: 평균 학습 정확도 (스칼라)
        test_acc:  평균 테스트 정확도 (스칼라)
    """
    X, y = preprocessBCW(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment

    n_runs = 50
    train_acc = np.zeros([n_runs])
    test_acc = np.zeros([n_runs])

    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(ID)

    for r in range(n_runs):
        # TODO: Randomly partition the dataset into two parts (50%
        #       training and 50% test; or as evenly as possible)
        permuted_indices = np.random.permutation(n)
        n_train = n//2

        train_idx = permuted_indices[:n_train]
        test_idx = permuted_indices[n_train:]

        Xtrain, ytrain = X[train_idx], y[train_idx]
        Xtest, ytest =  X[test_idx], y[test_idx]

        w = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # TODO: Evaluate the model's accuracy on the training
        # data. Save it to `train_acc`
        ytrain_hat = (Xtrain @ w > 0).astype(int)
        train_acc[r] = np.mean(ytrain_hat == ytrain)

        # TODO: Evaluate the model's accuracy on the test
        #       data. Save it to `test_acc`
        ytest_hat = (Xtest @ w > 0).astype(int)
        test_acc[r] = np.mean(ytest_hat == ytest)

    # TODO: compute the average accuracies over runs
    train_acc_avg = np.mean(train_acc, axis=0)
    test_acc_avg = np.mean(test_acc, axis=0)

    # TODO: return two variables: the average training accuracy and average test accuracy
    return train_acc_avg, test_acc_avg
