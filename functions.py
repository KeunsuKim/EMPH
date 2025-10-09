import numpy as np
import gudhi as gd  
import gudhi.representations
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.utils import shuffle
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from gtda.homology import VietorisRipsPersistence
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning



def run_one_seed(seed, X, Y, Ntrain):
    if seed is None:
        idx = np.arange(len(Y))                     # ★ 셔플 안 함
    else:
        rng = np.random.default_rng(int(seed))      # ★ 결정적 셔플
        idx = rng.permutation(len(Y))
    Xs, Ys = X[idx], Y[idx]
    return float(train_eval_linear_svm(Xs, Ys, Ntrain, gamma_auto=True))


def global_range_from_PDs(PDs, eps=1e-6):
    vals = [D for D in PDs if D is not None and D.size > 0]
    if not vals:
        xmin = ymin = 0.0
        xmax = ymax = 1.0
    else:
        all_xy = np.vstack(vals)          # (sum_k, 2)
        xmin, xmax = float(all_xy[:,0].min()), float(all_xy[:,0].max())
        ymin, ymax = float(all_xy[:,1].min()), float(all_xy[:,1].max())

    if not np.isfinite(xmin) or not np.isfinite(xmax): xmin, xmax = 0.0, 1.0
    if not np.isfinite(ymin) or not np.isfinite(ymax): ymin, ymax = 0.0, 1.0
    if xmax - xmin < eps:
        mid = 0.5 * (xmin + xmax); xmin, xmax = mid - eps/2, mid + eps/2
    if ymax - ymin < eps:
        mid = 0.5 * (ymin + ymax); ymin, ymax = mid - eps/2, mid + eps/2
    return (xmin, xmax, ymin, ymax)

# -----------------------------
# 벡터화: BettiCurve (전역 1D 범위)
# -----------------------------
def betticurve_vectors_with_global_range(PDs, resolution, eps=1e-6):
    xmin, xmax, ymin, ymax = global_range_from_PDs(PDs, eps=eps)
    gmin, gmax = min(xmin, ymin), max(xmax, ymax)
    BC = gd.representations.vector_methods.BettiCurve(
        sample_range=(gmin, gmax), resolution=resolution
    )
    out = np.zeros((len(PDs), resolution))
    for i, D in enumerate(PDs):
        out[i] = 0.0 if D is None or D.size == 0 else BC(D)
    return out

# -----------------------------
# 벡터화: Persistence Image (전역 2D 범위)
# -----------------------------
def persistence_image_vectors_with_global_range(PDs, resolution, bandwidth=0.05, weight=lambda x: x[1], eps=1e-6):
    im_range = global_range_from_PDs(PDs, eps=eps)
    PI = gd.representations.vector_methods.PersistenceImage(
        bandwidth=bandwidth, weight=weight,
        resolution=[resolution, resolution], im_range=list(im_range)
    )
    out = np.zeros((len(PDs), resolution * resolution))
    for i, D in enumerate(PDs):
        out[i] = 0.0 if D is None or D.size == 0 else PI(D)
    return out

# -----------------------------
# 헬퍼: 분할·스케일·학습·평가
# -----------------------------
def train_eval_linear_svm(X, Y, Ntrain, *, gamma_auto=True, max_iter=-1):
    mms = MinMaxScaler()
    Xtr = mms.fit_transform(X[:Ntrain])
    Xte = mms.transform(X[Ntrain:])
    Ytr, Yte = Y[:Ntrain], Y[Ntrain:]
    clf = svm.SVC(kernel='linear', gamma=('auto' if gamma_auto else 'scale'), max_iter=max_iter)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Xtr, Ytr)
    
    return float(np.mean(clf.predict(Xte) == Yte) * 100.0)

def SW_method(
    data, label, *, Ntrain,
    resolution, dim, tau, method, norm, bandwidth=0.05,
    seeds=None,            # e.g. range(10) 또는 [0,1,...,9]
    n_jobs_seeds=1,        # 씨드별 병렬화만 사용
    skip=1,                # 필요시 2나 4로 올리면 VR 속도 크게 ↑
    dtype=np.float32
):
    """
    data: (N, T)
    method: 'betti' | 'image'
    norm:   'chebyshev' 등 (VietorisRipsPersistence metric)

    변경점
    ------
    - 임베딩 결과가 모두 같은 길이 → 그대로 stack
    - VR.fit_transform()을 배치로 한 번만 호출
    - coeff=11 유지
    - classifier, ripser는 손대지 않음
    """

    if seeds is None:
        seeds = [None]
    else:
        seeds = list(seeds)

    delay = tau

    # 1) 임베딩 (전 샘플 배치)
    tde = TimeDelayEmbedding(dim=dim, delay=delay, skip=skip)
    emb_all = np.stack([tde(ts) for ts in data], axis=0).astype(dtype, copy=False)  # (N, L, dim)

    # 2) VR 배치 처리
    VR = VietorisRipsPersistence(
        metric=norm,
        coeff=11,                      # 유지
        homology_dimensions=[1],
        n_jobs=-1
    )
    PDs_raw = VR.fit_transform(emb_all)
    PDs = [D[:, :2] if (D is not None and D.size) else np.empty((0, 2), dtype=dtype)
           for D in PDs_raw]

    # 3) 특징 생성
    if method == 'betti':
        X = betticurve_vectors_with_global_range(PDs, resolution=resolution**2, eps=1e-6)
    elif method == 'image':
        X = persistence_image_vectors_with_global_range(PDs, resolution=resolution,
                                                        bandwidth=bandwidth, eps=1e-6)
    else:
        raise ValueError("method must be 'betti' or 'image'")
    if isinstance(X, np.ndarray) and X.dtype != dtype:
        X = X.astype(dtype, copy=False)

    Y = label


    accs = Parallel(n_jobs=n_jobs_seeds, backend="threading")(
        delayed(run_one_seed)(s, X, Y, Ntrain) for s in seeds
    )

    return accs



def _knn_dtw_one_seed(data, label, train_size, seed):
    """
    단일 seed에 대해:
      - seed=None이면 셔플하지 않고 앞에서부터 분할
      - seed가 정수면 해당 random_state로 셔플 분할
      - z-정규화 (train으로 fit 후 test에도 적용)
      - 1-NN + DTW 학습/예측
    """
    if seed is None:
        # 셔플 없이 앞부분 train_size개를 학습으로, 나머지를 테스트로
        X_train, X_test = data[:train_size], data[train_size:]
        y_train, y_test = label[:train_size], label[train_size:]
    else:
        # 시드 기반 셔플 분할
        X_train, X_test, y_train, y_test = train_test_split(
            data, label,
            train_size=train_size,
            shuffle=True,
            random_state=seed
        )

    # z-정규화
    scaler = TimeSeriesScalerMeanVariance()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 1-NN + DTW
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "seed": seed,
        "accuracy": float(acc),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run_knn_dtw_parallel(data, label, Ntrain, seeds=None, n_jobs=-1, verbose=0):
    """
    data: (N, T)
    label: (N,)
    Ntrain: 학습 데이터 개수(정수) 또는 비율(0<train_size<1)
    seeds: 반복 실행할 시드들의 iterable (예: range(10) 또는 [0,1,...]) 
           → None 포함 가능 (shuffle 안 함)
    n_jobs: 병렬 잡 수 (joblib)
    verbose: joblib verbose
    """
    # train_size 해석
    if isinstance(Ntrain, (int, np.integer)):
        train_size = int(Ntrain)
    else:
        train_size = float(Ntrain)  # 비율

    if seeds is None:
        seeds = [None]  # 기본 1회 (shuffle 없음)

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(_knn_dtw_one_seed)(data, label, train_size, seed) for seed in seeds
    )
    accs = np.array([r["accuracy"] for r in results], dtype=float)
    return results, accs

def EMPH_method(tr_seed, resolution, fmax, base_amp, label, *, Ntrain,
                     bandwidth=0.05, method='betti', seeds=None, dtype=np.float32):
    """
    drop-in replacement: 반환값과 인터페이스 동일
    """
    rng = np.random.default_rng(tr_seed)
    N = base_amp.shape[0]

    # 1) direction / endpoint (float32로 한 번에 캐스팅)
    direction = 1.0 + np.round(rng.random(fmax).astype(np.float64), 1)
    endpoint  = np.round((5.0 * rng.random(fmax)).astype(np.float64), 1)

    # 2) scale 계산 (분자/분모를 분리해 곱셈 위주로)
    dir_norm = np.sqrt(np.dot(direction, direction)).astype(dtype)
    inv_sqrt_fmax = (1.0 / np.sqrt(np.float32(fmax))).astype(dtype)
    scale = (dir_norm * inv_sqrt_fmax) / direction                                     # (fmax,)

    # 3) birth/death 벡터화 (배열 1회 생성)
    # birth_time = -endpoint * scale (N개로 브로드캐스트)
    birth_time = (-endpoint * scale).astype(dtype)                                     # (fmax,)
    death_time = ((base_amp.astype(dtype, copy=False) - endpoint) * scale).astype(dtype, copy=False)  # (N,fmax)

    # (N,fmax,2)로 한 번에 스택 (리스트/루프 제거)
    # 모든 샘플 동일한 birth_time을 브로드캐스트
    PDs = np.empty((N, fmax, 2), dtype=dtype)
    PDs[..., 0] = birth_time     # birth
    PDs[..., 1] = death_time     # death

    # 4) 특징 변환 (배치 입력 지원 가정)
    if method == 'image':
        X = persistence_image_vectors_with_global_range(
            PDs, resolution=resolution, bandwidth=bandwidth, eps=1e-6
        )
    elif method == 'betti':
        # Betti는 길이 맞추기 위해 resolution**2 사용 (기존 로직 유지)
        X = betticurve_vectors_with_global_range(
            PDs, resolution=resolution**2, eps=1e-6
        )
    else:
        raise ValueError("method must be 'image' or 'betti'")

    # 5) 시드별 학습/평가: 인덱스 셔플로 복사 최소화
    acc_list = []
    if seeds is None:
        acc = train_eval_linear_svm(X, label, Ntrain, gamma_auto=True)
        acc_list.append(float(acc))
    else:
        Nidx = np.arange(N)
        for s in seeds:
            rng_s = np.random.default_rng(int(s))
            idx = rng_s.permutation(Nidx)  # 인덱스만 섞음
            acc = train_eval_linear_svm(X[idx], label[idx], Ntrain, gamma_auto=True)
            acc_list.append(float(acc))

    acc_arr = np.asarray(acc_list, dtype=float)
    return acc_arr, direction.tolist(), endpoint.tolist()





def EMPH_method_original(tr_seed, resolution, fmax, base_amp, label, *, Ntrain,
                          bandwidth=0.05, method='betti', max_iter=-1, seeds=None, dtype=np.float32):
    """
    Drop-in replacement for EMPH_method_original (same I/O).
    Vectorized; removes exponential permutation cost for n=1 case.
    """
    rng = np.random.default_rng(tr_seed)
    N = base_amp.shape[0]               # #samples
    lefr = int(fmax)                    # number of frequencies (L)
    assert base_amp.shape[1] >= lefr, "base_amp's 2nd dim must be >= fmax"

    # 1) direction / endpoint
    direction = 1.0 + np.round(rng.random(lefr).astype(np.float64), 1)     # [1.0, 2.0] step 0.1
    endpoint  = np.round((5.0 * rng.random(lefr)).astype(np.float64), 1)   # [0.0, 5.0) step 0.1
    direction_normal = direction / np.linalg.norm(direction)               # (lefr,)

    # 공통 분모: sqrt(L) * â_L
    denom = np.sqrt(lefr) * direction_normal                               # (lefr,)

    # ##### 핵심 수식 (n=1 바코드) #####
    # 전체 birth는 각 축 L에 대해 (−b_L / denom_L)의 최댓값으로 고정 (샘플 무관)
    birth_global = np.max(-endpoint / denom)                               # scalar

    # 각 샘플×주파수 L에 대한 death: (sqrt(3)*base_amp[:,L] − b_L)/denom_L
    deaths = (np.sqrt(3.0) * base_amp[:, :lefr] - endpoint[np.newaxis, :]) / denom[np.newaxis, :]   # (N, lefr)

    # 유효 interval만 필터링: death - birth >= 1e-6
    mask = (deaths - birth_global) >= 1e-6                                 # (N, lefr)

    # PDs: 각 샘플 i에 대해 유효한 (birth, death_ij) 쌍들
    PDs = []
    birth_col = np.full((lefr,), birth_global, dtype=np.float64)           # for stacking
    for i in range(N):
        if mask[i].any():
            d = deaths[i, mask[i]].astype(np.float64, copy=False)
            b = birth_col[mask[i]]
            PDs.append(np.asarray(np.stack([b, d], axis=1), dtype=dtype))
        else:
            PDs.append(np.asarray([], dtype=dtype).reshape(0, 2))

    # 4) 특징 변환
    if method == 'image':
        X = persistence_image_vectors_with_global_range(
            PDs, resolution=resolution, bandwidth=bandwidth, eps=1e-6
        )
    elif method == 'betti':
        X = betticurve_vectors_with_global_range(
            PDs, resolution=resolution**2, eps=1e-6
        )
    else:
        raise ValueError("method must be 'image' or 'betti'")

    # 5) 시드별 학습/평가 (메모리 복사 최소화)
    acc_list = []
    if seeds is None:
        acc = train_eval_linear_svm(X, label, Ntrain, gamma_auto=True, max_iter=max_iter)
        acc_list.append(float(acc))
    else:
        Nidx = np.arange(N)
        for s in seeds:
            rng_s = np.random.default_rng(int(s))
            idx = rng_s.permutation(Nidx)
            acc = train_eval_linear_svm(X[idx], label[idx], Ntrain, gamma_auto=True, max_iter=max_iter)
            acc_list.append(float(acc))

    acc_arr = np.asarray(acc_list, dtype=float)
    return acc_arr, direction.tolist(), endpoint.tolist()





