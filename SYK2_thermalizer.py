import numpy as np
import functools as ft
import more_itertools
from more_itertools import distinct_permutations as idp
from scipy.linalg import expm
import random

# down -- ground state of a single qubit
# up -- excited state of a single qubit

down = np.array([1j, 1]) / np.sqrt(2)
up = np.array([-1j, 1]) / np.sqrt(2)

# 2 x 2 matrices acting in a single qubit space:
one = np.array([[1, 0], [0, 1]])
sy = np.array([[0, - 1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]]) 
sx = np.array([[0, 1], [1, 0]])
plus = np.array([[0, 1], [0, 0]])
minus = np.array([[0, 0], [1, 0]])
mplus = plus @ sz
mminus = sz @ minus
#low = 0.5 * ( - sz + 1j * sx )

#def gs(n, m): 
#    v = [1] * int(m) + [0] * int(n - m) 
#    for j in range(len(v)):
#        if v[j] == 0:
#            v[j] = down        
#        else:
#            v[j] = up     
#    f = ft.reduce(np.kron, v)
#    return f

def psi(n): 
    f = []
    for k in range(n+1):
        listvec = [0] * (n - k) + [1] * k
        for i in idp(listvec):
            v = list(i)
            for j in range(len(v)):
                if v[j] == 0:
                    v[j] = down        
                else:
                    v[j] = up     
            vec = ft.reduce(np.kron, v)
            f.extend([vec])
    return f

def energ_sp(n, omega_min, omega_max):
    f = np.zeros(n)
    for j in range(n):
        f[j] = random.uniform(omega_min, omega_max)
    return f  

def perm_diag(n): 
    listvec = ['Y']  + ['I'] * (n - 1) 
    L1 = []
    for i in idp(listvec):
        v = list(i)   
        L1.extend([v])
    return L1  

def H_0(n, omegas):
    dim = (2 ** n, 2 ** n)
    f_sum = np.zeros(dim)
    L1 = perm_diag(n)
    for i in range(len(L1)):
        v = L1[i]
        for j in range(len(v)):
            if v[j] == 'Y':
                v[j] = sy  
            else:
                v[j] = one        
        f = 0.5 * omegas[i] * ft.reduce(np.kron, v) 
        f_sum = f_sum + f
    return f_sum

def energ_mb(n, omegas):
    fs = psi(n)
    ls = []
    es = []
    H = H_0(n, omegas)
    for l in range(len(fs)):
        ls.append(l)
        energ = np.conjugate(fs[l]) @ H @ fs[l]
        if np.abs(np.imag(energ)) > 1e-15:
            print('error: complex energies')
        es.append(np.real(energ))
    return ls, es
        
def perm_SYK2(n):
    listvec = ['P', 'M']  + ['I'] * (n - 2)  
    #listvec =  ['M']  +['I'] * (n - 2) + ['P']  
    L1 = [] 
    L2 = []
    k = 0
    for i in idp(listvec):
        k += 1
        v = list(i)
        vconj = [0] * len(listvec)        
        for j in range(len(listvec)):
            if v[j] == 'P':
                vconj[j] = 'M'        
            elif v[j] == 'M':
                vconj[j] = 'P'   
            else:
                vconj[j] = 'I'    
        L1.extend([v])
        L2.extend([vconj])         
        if k > 1:
            for m in range(2, k+1):
                if v == L2[m-2]:
                    L1.remove(v)
    l = []    
    for k in range(len(L1)):
        for m in range(len(L1[k])):
            if L1[k][m] == 'P':
                L1[k][m] = 'PZ'
                l.extend([m])
                break
            elif L1[k][m] == 'M':
                L1[k][m] = 'ZM'
                l.extend([m])
                break  
    for k in range(len(L1)):
        for m in range(l[k]+1,len(L1[k])):
            if L1[k][m] == 'I':
                L1[k][m] = 'Z'
            elif L1[k][m] == 'P':
                break
            elif L1[k][m] == 'M':
                break
    return L1

def test_perm(n):
    listvec = []
    listvec.append(['P', 'ZM', 'I', 'I'])
    listvec.append(['P', 'Z', 'ZM', 'I'])
    listvec.append(['P', 'Z', 'Z', 'ZM'])
    listvec.append(['I', 'P', 'ZM', 'I'])
    listvec.append(['I', 'P', 'Z', 'ZM'])
    listvec.append(['I', 'I', 'P', 'ZM']) 
    return listvec
 
def J_SYK2(n, J):
    varJ = J ** 2 / n 
    Js = [] 
    L1 = perm_SYK2(n)
    for i in range(len(L1)):    
        v = np.random.normal(loc = 0.0, scale = np.sqrt(0.5 * varJ), size = (1, 2)).view(np.complex)[0][0]
        Js.extend([v])
    return Js  

def H_SYK2(n, couplings):
    dim = (2 ** n, 2 ** n)
    H_sum = np.zeros(dim) 
    if n == 4:
        L1 = test_perm(n)
    else:
        L1 = perm_SYK2(n)
    #L1 = test_perm(n)
    Js = couplings
    k = 0
    for i in range(len(L1)):
        k += 1
        v = L1[i]        
        vconj = [0] * len(v)        
        for j in range(len(v)):
            if v[j] == 'ZM':
                v[j] = mminus
                vconj[j] = mplus
            elif v[j] == 'P':
                v[j] = plus
                vconj[j] = minus
            elif v[j] == 'PZ':
                v[j] = mplus 
                vconj[j] = mminus
            elif v[j] == 'M':
                v[j] = minus
                vconj[j] = plus
            elif v[j] == 'Z':
                v[j] = sz
                vconj[j] = sz
            else:
                v[j] = one
                vconj[j] = one                 
        #H =  Js[k-1] * ft.reduce(np.kron, v) + np.conjugate(Js[k-1]) * ft.reduce(np.kron, vconj)
        H =  Js[k-1] * ft.reduce(np.kron, vconj) + np.conjugate(Js[k-1]) * ft.reduce(np.kron, v)
        H_sum = H_sum + H
        #for i in range(0,dim[0]):
        #    for j in range(0,dim[1]):
        #        if H_sum[i][j] - np.conjugate(H_sum[j][i]) != 0.:
        #            print('Hermicity check fails!')     
    return H_sum

def ev_state(n, m, t, H_0, V):
    H_tot = H_0 + V
    U = expm(- 1j * t * H_tot)
    f0 = psi(n)
    e0 = np.conjugate(f0[m]) @ H_0 @ f0[m]
    if np.abs(np.imag(e0)) > 1e-15:
            print('error: complex energies')
    f = np.dot(U, f0[m])
    return e0, f

def p(n, m, t, H_0, V): 
    fs = psi(n)
    es = []
    ps = []
    #H_tot = H_0 + V
    E_in, psi_t = ev_state(n, m, t, H_0, V)
    for l in range(len(fs)):
        energ = np.conjugate(fs[l]) @ H_0 @ fs[l]
        if np.abs(np.imag(energ)) > 1e-15:
            print('error: complex energies')
        es.append(np.real(energ))
        pop = np.abs(np.dot(np.conjugate(fs[l]), psi_t)) ** 2
        ps.append(pop)
    es_order = []
    ps_order = []    
    for l in range(len(fs)):    
        es_order.append(sorted(zip(es, ps))[l][0])
        ps_order.append(sorted(zip(es, ps))[l][1])
    return es_order, ps_order, E_in

def density_matrix(n, m, t, H_0, V): 
    fs = psi(n)
    es = []
    ps = []
    #H_tot = H_0 + V
    psi_t = ev_state(n, m, t, H_0, V)[1]
    E_in = ev_state(n, m, t, H_0, V)[0]
    for l in range(len(fs)):
        energ = np.conjugate(fs[l]) @ H_0 @ fs[l]
        if np.abs(np.imag(energ)) > 1e-15:
            print('error: complex energies')
        es.append(np.real(energ))
        pop = np.dot(np.conjugate(fs[l]), psi_t)
        ps.append(pop)
    es_order = []
    ps_order = []    
    for l in range(len(fs)):    
        es_order.append(sorted(zip(es, ps))[l][0])
        ps_order.append(sorted(zip(es, ps))[l][1])
    dens_mat = np.zeros((len(fs), len(fs)), dtype = 'complex_')    
    for l1 in range(len(fs)):
        for l2 in range(len(fs)):
            dens_mat[l1][l2] = ps_order[l1] * np.conj(ps_order[l2])
    return dens_mat, E_in

def p_th(n, b, omegas):
    es = energ_mb(n, omegas)[1]
    ps = []
    for l in range(len(es)):
        pop = np.exp(- b * es[l])
        ps.append(pop)
    Z = np.sum(ps)
    for l in range(len(es)):
        ps[l] = ps[l] / Z 
    es_order = []
    ps_order = []
    for l in range(len(es)):
        es_order.append(sorted(zip(es, ps))[l][0])
        ps_order.append(sorted(zip(es, ps))[l][1])
    return es_order, ps_order

# the parameters: 
# num -- number of qubits
# num_ex -- number of the initial state. It ranges from 0 to 2^N - 1.
# Jc -- square root of the variance of random couplings J_{ij}
# nr -- number of realizations
# [omega_min, omega_max] -- defines the interval for the single-particle energies
# do -- levels' spacing
# t_min -- mininum time
# t_max -- maximum time
# nt -- number of time points

def gen_coupl(num, nr, Jc):
    Js = [0] * nr
    for j in range(nr):
        Js[j] = J_SYK2(num, Jc)
    np.save('data/couplings_N={}_nr={}_J={}.npy'.format(num, nr, Jc), Js, allow_pickle = True)  
    
def load_coupl(num, nr, Jc):
    Js = [0] * nr
    cc_mat = [0] * nr
    for j in range(nr):
        #print(j)
        cc_mat[j] = np.load('input/random_interaction/random_interaction_{}.npy'.format(j+1), allow_pickle = True)
        cc_list = []
        for i1 in range(num-1, 0, -1):
            for i2 in range(i1-1, -1, -1): 
                print(i2, i1)
                #cc_list.append(cc_mat[j][i1][i2])
                cc_list.append(cc_mat[j][i2][i1])
        Js[j] = np.array(cc_list)
        #print(cc_list)
        #Js[j] = cc_mat[j]
    np.save('data/couplings_QC_N={}_nr={}_J={}.npy'.format(num, nr, Jc), Js, allow_pickle = True)  
        
def gen_energ(num, omega_min, omega_max):
    sp_levels = energ_sp(num, omega_min, omega_max)
    energ = sorted(energ_mb(num, sp_levels)[1])
    print(np.round(sp_levels, 3))
    print(np.round(energ, 3))
    np.save('data/sp_levels_N={}.npy'.format(num), sp_levels, allow_pickle = True)  
    np.save('data/energ_N={}.npy'.format(num), energ, allow_pickle = True)
    
def gen_energ4(num):
    sp_levels = [0.28, 0.38, 0.63, 0.86]
    #sp_levels = [0.86, 0.63, 0.38, 0.28]
    #sp_levels = [0.28, 0.63, 0.38, 0.86]
    #sp_levels = [0.86, 0.28, 0.38, 0.63]
    #sp_levels = [0.63, 0.86, 0.28, 0.38]
    #sp_levels = [0.38, 0.63, 0.86, 0.28]
    energ = sorted(energ_mb(num, sp_levels)[1])
    print(np.round(sp_levels, 3))
    print(np.round(energ, 3))
    np.save('data/sp_levels_N={}.npy'.format(num), sp_levels, allow_pickle = True)  
    np.save('data/energ_N={}.npy'.format(num), energ, allow_pickle = True)    
         
def SYK2(num, num_ex, Jc, nr, t_min, t_max, nt): 
    time = np.linspace(t_min, t_max, nt) 
    #sp_levels = energ_sp(num, omega_min, omega_max)    #[0.28, 0.38, 0.63, 0.86] 
    sp_levels = np.load('data/sp_levels_N={}.npy'.format(num), allow_pickle = True)  
    H_qubits = H_0(num, sp_levels)
    energ = np.load('data/energ_N={}.npy'.format(num), allow_pickle = True) #sorted(energ_mb(num, sp_levels)[1])
    #print(np.round(sp_levels, 3))
    #print(np.round(energ, 3))

    Le = [[0] * nr for t in range(len(time))]
    Le2 = [[0] * nr for t in range(len(time))]
    Lp = [[0] * nr for t in range(len(time))] 
    E = [[0] * nr for t in range(len(time))] 
    E2 = [[0] * nr for t in range(len(time))] 
    EE2 = [[0] * nr for t in range(len(time))]
    S = [[0] * nr for t in range(len(time))]
    rho = [[0] * nr for t in range(len(time))]
     
    coupling_consts = np.load('data/couplings_N={}_nr={}_J={}.npy'.format(num, nr, Jc), allow_pickle = True)
    #coupling_consts = np.load('data/couplings_QC_N={}_nr={}_J={}.npy'.format(num, nr, Jc), allow_pickle = True)
    
    E_in = np.real(p(num, num_ex, 0., H_qubits, - H_qubits)[2])
    print(E_in)
    
    for j in range(nr):
        print(j)
        #Js = J_SYK2(num, Jc)
        Js = coupling_consts[j]
        V = H_SYK2(num, Js)
        for t in range(len(time)):
            Le[t][j], Lp[t][j], ein = p(num, num_ex, time[t], H_qubits, V)
            delta = np.array(Le[t][j]) - np.array(energ)
            if delta.any() != 0:
                print('error: the energies do not coincide')
            Le2[t][j] = [q ** 2 for i, q in enumerate(Le[t][j])]    
            E[t][j] = np.array(Le[t][j]) @ np.array(Lp[t][j])
            E2[t][j] = np.array(Le2[t][j]) @ np.array(Lp[t][j]) 
            EE2[t][j] =  E2[t][j] - E[t][j] ** 2
            S[t][j] = - np.array(Lp[t][j]) @ np.log(np.array(Lp[t][j]))
            #rho[t][j] = density_matrix(num, num_ex, time[t], H_qubits, V)[0] 
 
    pav = np.sum(np.array(Lp), 1) / nr
    err_av = np.std(np.array(Lp), 1, ddof = 1) / np.sqrt(nr)
    Eav = np.sum(np.array(E), 1)  / nr
    E2av = np.sum(np.array(E2), 1)  / nr
    VarE = E2av  - Eav ** 2
    Sav = np.sum(np.array(S), 1) / nr
    #rhoav = np.sum(np.array(rho), 1) / nr

    np.save('data/time.npy', time, allow_pickle = True)
    np.save('data/es_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), Le, allow_pickle = True)
    np.save('data/p_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), Lp, allow_pickle = True)
    np.save('data/pav_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), pav, allow_pickle = True)
    np.save('data/err_av_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), err_av, allow_pickle = True)
    np.save('data/E_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), E, allow_pickle = True)
    np.save('data/Eav_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), Eav, allow_pickle = True)
    np.save('data/E2_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), E2, allow_pickle = True)
    np.save('data/EE2_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), EE2, allow_pickle = True)
    np.save('data/S_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), S, allow_pickle = True)
    np.save('data/E2av_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), E2av, allow_pickle = True)
    np.save('data/VarE_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), VarE, allow_pickle = True)
    np.save('data/Sav_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), Sav, allow_pickle = True)
    #np.save('data/rhoav_N={}_M={}_nr={}.npy'.format(num, num_ex, nr), rhoav, allow_pickle = True)
    # the data is saved into 'data/...'
    
    #return E_in, S, Sav

def entropy(num, Jc, nr, t_min, t_max, nt):
    for m in range(0, 2 ** num):
        SYK2(num, m, Jc, nr, t_min, t_max, nt)
        
def initial_energies(num):
    sp_levels = np.load('data/sp_levels_N={}.npy'.format(num), allow_pickle = True)  
    H_qubits = H_0(num, sp_levels)
    M = int(2 ** num)
    E_in = [0] * M
    
    for m in range(M):
        E_in[m] = np.real(p(num, m, 0., H_qubits, - H_qubits)[2])
        
    np.save('data/initial_N={}.npy'.format(num), E_in, allow_pickle = True)
