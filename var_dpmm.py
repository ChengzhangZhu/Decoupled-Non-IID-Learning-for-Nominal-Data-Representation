import numpy as np
import scipy.sparse as sp
from scipy.special import psi, gammaln
from tqdm import tqdm

def transform_data(data):
    num_attribute = data.shape[1]
    num_object = data.shape[0]
    X = list()
    for attribute in range(num_attribute):
        data_slice = data[:,attribute]
        values = np.unique(data_slice)
        num_values = len(values)
        X_slice = list()
        for i, obj in enumerate(data_slice):
            onehot = np.zeros(num_values)
            onehot[obj-1] = 1
            X_slice.append(onehot.tolist())
        X_slice = np.asarray(X_slice)
        X_slice = sp.csr_matrix(X_slice)
        X.append(X_slice)

    return X


def get_cat(phi):
    label = []
    N, M = phi.shape
    for m in range(M):
        list_phi = phi[:,m].tolist()
        label.append(list_phi.index(max(list_phi)))
    return label


def load_ap_data(ap_data, ap_vocab):
    n = len(open(ap_data).readlines())
    m = len(open(ap_vocab).readlines())

    X = sp.lil_matrix((n,m))

    for i, line in enumerate(open(ap_data)):
        words = line.split()
        idxs = []
        vals = []
        for w in words[1:]:
            idx, val = map(int, w.split(':'))
            idxs.append(idx)
            vals.append(val)
        
        X[i,idxs] = vals

    X = X.tocsr()
    return X

def load_bow_data(bow_data):
    f = open(bow_data)
    n = int(f.readline().strip())
    m = int(f.readline().strip())
    _ = f.readline()

    d = np.array([map(int, line.split()) for line in f])

    X = sp.csr_matrix((d[:,2], d[:,:2].T - 1), shape=(n,m))
    return X

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis)
    try:
        return a_max + np.log(np.sum(np.exp(a - a_max), axis=axis))
    except:
        return a_max + np.log(np.sum(np.exp(a - a_max[:,np.newaxis]), axis=axis))


def var_dpmm_multinomial(X, alpha, T=50, n_iter=100, Xtest=None, verbose=True):
    '''
    runs variational inference on a DP mixture model where each
    mixture component is a multinomial distribution.

    X: observed data, (N,M) matrix, can be sparse
    alpha: concentration parameter
    base_dirichlet: base measure (Dirichlet (1,M) in this case)
    '''
    #N, M = X.shape
    N = X[0].shape[0]
    M = len(X)
    # attribute J is a variable

    base_dirichlet = []
    for m in range(M):
        base_dirichlet.append(np.ones((X[m].shape[1])))

    # variational multinomial parameters for z_n
    phi = np.matrix(np.random.uniform(size=(T,N)))
    phi = np.divide(phi, np.sum(phi, axis=0))

    # variational beta parameters for V_t
    gamma1 = np.matrix(np.zeros((T-1,1)))
    gamma2 = np.matrix(np.zeros((T-1,1)))

    # variational dirichlet parameters for \eta_t
    tau = []
    for m in range(M):
        tau.append(np.matrix(np.zeros((T, X[m].shape[1]))))
    #tau = np.matrix(np.zeros((T,M,J)))

    temp1 = np.zeros((T, N))
    temp2 = np.zeros((N, N))

    ll = []
    held_out = []
    if verbose is True:
        bar = tqdm(range(n_iter))
    else:
        bar = range(n_iter)
    for it in bar:
        # sys.stdout.write('.'); sys.stdout.flush()
        gamma1 = 1. + np.sum(phi[:T-1,:], axis=1)
        phi_cum = np.cumsum(phi[:0:-1,:], axis=0)[::-1,:]
        gamma2 = alpha + np.sum(phi_cum, axis=1)

        for m in range(M):
            tau[m] = base_dirichlet[m] + phi * X[m]
        #tau = base_dirichlet + phi * X

        lV1 = psi(gamma1) - psi(gamma1 + gamma2)  # E_q[log V_t]
        lV1 = np.vstack((lV1, 0.))
        lV2 = psi(gamma2) - psi(gamma1 + gamma2)  # E_q[log (1-V_t)]
        lV2 = np.cumsum(np.vstack((0., lV2)), axis=0)  # \sum_{i=1}^{t-1} E_q[log (1-V_i)]

        #eta = psi(tau) - psi(np.sum(tau, axis=1)) # E_q[eta_t]
        eta = []
        for m in range(M):
            eta.append(psi(tau[m] - psi(np.sum(tau[m], axis=1))))
        #eta = psi(tau) - psi(np.sum(tau, axis=2))

        #S = lV1 + lV2 + eta * X.T

        for m in range(M):
            temp1 += eta[m]* X[m].T
        S = lV1 + lV2 + temp1
        S = S - logsumexp(S, axis=0)
        phi = np.exp(S)

        ll.append(log_likelihood(X, gamma1, gamma2, tau,
            alpha, base_dirichlet, phi=phi, eta=eta))
        if Xtest is not None:
            held_out.append(mean_log_predictive(Xtest, gamma1, gamma2, tau,
                alpha, base_dirichlet, temp2, eta=eta))

    return gamma1, gamma2, tau, phi, ll, held_out

def log_likelihood(X, gamma1, gamma2, tau, alpha, base_dirichlet, phi, eta=None):
    '''computes lower bound on log marginal likelihood'''
    lV1 = psi(gamma1) - psi(gamma1 + gamma2)  # E_q[log V_t]
    lV11 = np.vstack((lV1, 0.))
    lV2 = psi(gamma2) - psi(gamma1 + gamma2)  # E_q[log (1-V_t)]
    lV22 = np.cumsum(np.vstack((0., lV2)), axis=0)  # \sum_{i=1}^{t-1} E_q[log (1-V_i)]
    #lambda1 = np.matrix(base_dirichlet).T  # M*1
    N = X[0].shape[0]
    M = len(X)
    # attribute J is a variable
    lambda1 = []
    for m in range(M):
        lambda1.append(np.matrix(base_dirichlet[m]).T)
    #lambda1 = np.matrix(base_dirichlet)     # M*J

    T = tau[0].shape[0]

    if eta is None:
        eta = []
        for m in range(M):
            eta.append(psi(tau[m] - psi(np.sum(tau[m], axis=1))))
        #eta = psi(tau) - psi(np.sum(tau, axis=1))

    phi_cum = np.cumsum(phi[:0:-1,:], axis=0)[::-1,:]

    # E_q[log p(V|alpha)]
    ll = np.sum((alpha - 1) * lV2) - \
            (T-1) * (gammaln(alpha) - gammaln(1.+alpha))

    # E_q[log p(eta|lambda)]
    #ll += np.sum(eta * (lambda1 - 1)) - \
            #T * (np.sum(gammaln(lambda1)) - gammaln(np.sum(lambda1)))
    for m in range(M):
        ll += np.sum(eta[m]*(lambda1[m]-1))-T*(np.sum(gammaln(lambda1[m]))-gammaln(np.sum(lambda1[m])))

    # \sum_n E_q[log p(Z_n|V)]
    ll += np.sum(np.multiply(phi[:-1,:], lV1) + np.multiply(phi_cum, lV2))

    # \sum_n E_q[log p(x_n | Z_n)]
    #ll += np.sum(np.multiply(phi.T, X * eta.T))
    for m in range(M):
        ll += np.sum(np.multiply(phi.T, X[m] * eta[m].T))

    # - E_q[log q(V)]
    ll -= ((gamma1 - 1).T * lV1 + (gamma2 - 1).T * lV2).item() - \
            np.sum(gammaln(gamma1) + gammaln(gamma2) - gammaln(gamma1 + gamma2))

    # - E_q[log q(eta)]
    # ll -= np.sum(np.multiply(tau - 1, eta)) - \
    #         np.sum(np.sum(gammaln(tau), axis=1) - gammaln(np.sum(tau, axis=1)))
    for m in range(M):
        ll -= np.sum(np.multiply(tau[m] - 1, eta[m])) - \
              np.sum(np.sum(gammaln(tau[m]), axis=1) - gammaln(np.sum(tau[m], axis=1)))

    # - E_q[log q(z)]
    ll -= np.sum(np.nan_to_num(np.multiply(phi, np.log(phi))))

    return ll

def mean_log_predictive(X, gamma1, gamma2, tau, alpha, base_dirichlet, temp2, eta=None):
    '''Computes the mean of the log predictive distribution over sample X,
    typically held out data.'''
    lV1 = psi(gamma1) - psi(gamma1 + gamma2)  # E_q[log V_t]
    lV11 = np.vstack((lV1, 0.))
    lV2 = psi(gamma2) - psi(gamma1 + gamma2)  # E_q[log (1-V_t)]
    lV22 = np.cumsum(np.vstack((0., lV2)), axis=0)  # \sum_{i=1}^{t-1} E_q[log (1-V_i)]

    if eta is None:
        #eta = psi(tau) - psi(np.sum(tau, axis=1))
        M = len(X)
        for m in range(M):
            eta[m] = psi(tau[m]) - psi(np.sum(tau[m], axis=1))

    # E_q[log pi(V)]
    lPi = lV11 + lV22

    # p(x_N|x) = \sum_t E_q[pi_t(V)] E_q[p(x_N|eta_t)]
    #lPred = logsumexp(lPi.T + X * eta.T, axis=1)

    for m in range(M):
        temp2 += X[m] * eta[m].T
    lPred = logsumexp(lPi.T + temp2, axis=1)
    return lPred.mean()

