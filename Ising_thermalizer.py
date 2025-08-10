import numpy as np
import functools as ft
import more_itertools
from more_itertools import distinct_permutations as idp
from scipy.linalg import expm
import random
from numpy.linalg import eig
from numpy.linalg import eigh
from numpy.linalg import eigvalsh

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
mplus = - plus 
mminus = - minus 

def psi(n): 
    f = []
    for k in range(n+1):
        listvec = [0] * (n - k) + [1] * k
        for i in idp(listvec):
            v = list(i)
            for j in range(len(v)):
                if v[j] == 0:
                    v[j] = up        
                else:
                    v[j] = down     
            vec = ft.reduce(np.kron, v)
            f.extend([vec])
    return f

def perm_diag(n): 
    listvec = ['S']  + ['I'] * (n - 1) 
    L1 = []
    for i in idp(listvec):
        v = list(i)   
        L1.extend([v])
    return L1  

def H_X(n, omega):
    dim = (2 ** n, 2 ** n)
    f_sum = np.zeros(dim)
    L1 = perm_diag(n)
    for i in range(len(L1)):
        v = L1[i]
        for j in range(len(v)):
            if v[j] == 'S':
                v[j] = sx  
            else:
                v[j] = one        
        f = - omega * ft.reduce(np.kron, v) 
        f_sum = f_sum + f
    return f_sum

def H_Z(n, h):
    dim = (2 ** n, 2 ** n)
    f_sum = np.zeros(dim)
    L1 = perm_diag(n)
    for i in range(len(L1)):
        v = L1[i]
        for j in range(len(v)):
            if v[j] == 'S':
                v[j] = sy  
            else:
                v[j] = one        
        f = - h * ft.reduce(np.kron, v) 
        f_sum = f_sum + f
    return f_sum
        
def perm_SYK2(n): 
    listvec = ['P', 'M']  + ['I'] * (n - 2) 
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

def perm_NN(n, pbc):
    listvec = ['P', 'M']  + ['I'] * (n - 2) 
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
    ll = []            
    for k in range(len(L1)):
        for m in range(l[k]+1,len(L1[k])):
            if L1[k][m] == 'I':
                ll.extend([k])
                break
            elif L1[k][m] == 'P':
                break
            elif L1[k][m] == 'M':
                break
    L10 = []
    for kk in ll:
        L10.extend([L1[kk]])
    for kk in range(len(L10)):
        L1.remove(L10[kk])
    if pbc == 1:
        L1_pbc = [['P']  + ['I'] * (n - 2) + ['ZM']]
        #print(L1_pbc)
        L1.extend(L1_pbc)
    return L1

def H_ZZ(n, g, pbc):
    dim = (2 ** n, 2 ** n)
    H_sum = np.zeros(dim)
    L1 = perm_NN(n, pbc) 
    k = 0
    for i in range(len(L1)):
        k += 1
        v = L1[i]        
        vconj = [0] * len(v)        
        for j in range(len(v)):
            if v[j] == 'ZM':
                v[j] = sy
                vconj[j] = sy
            elif v[j] == 'P':
                v[j] = sy
                vconj[j] = sy
            elif v[j] == 'PZ':
                v[j] = sy 
                vconj[j] = sy
            elif v[j] == 'M':
                v[j] = sy
                vconj[j] = sy
            #elif v[j] == 'Z':
            #    v[j] = sz
            #    vconj[j] = sz
            else:
                v[j] = one
                vconj[j] = one                 
        H = - g * ft.reduce(np.kron, v)
        H_sum = H_sum + H
        #for i in range(0,dim[0]):
        #    for j in range(0,dim[1]):
        #        if H_sum[i][j] - np.conjugate(H_sum[j][i]) != 0.:
        #            print('Hermicity check fails!')     
    return H_sum

def spectrum_Ising(n, omega, h, g, pbc):
    H = H_X(n, omega) + H_Z(n, h) + H_ZZ(n, g, pbc)
    es, vs = eigh(H)
    np.save('data/energ_N={}_omega={}_h={}_g={}_pbc={}.npy'.format(n, omega, h, g, pbc), es, allow_pickle = True)
    np.save('data/psi_N={}_omega={}_h={}_g={}_pbc={}.npy'.format(n, omega, h, g, pbc), vs, allow_pickle = True) 
    return es, vs
 
def J_SYK2(n, J): #:J_SYK2(n, J, pbc):
    varJ = J ** 2 / n 
    Js = [] 
    L1 = perm_SYK2(n) #, pbc) #perm_NN(n, pbc) 
    for i in range(len(L1)):    
        v = np.random.normal(loc = 0.0, scale = np.sqrt(0.5 * varJ), size = (1, 2)).view(np.complex)[0][0]
        Js.extend([v])
    return Js  

def H_SYK2(n, couplings): 
    dim = (2 ** n, 2 ** n)
    H_sum = np.zeros(dim)
    L1 =  perm_SYK2(n)
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
        H =  Js[k-1] * ft.reduce(np.kron, v) + np.conjugate(Js[k-1]) * ft.reduce(np.kron, vconj)
        H_sum = H_sum + H
        #for i in range(0,dim[0]):
        #    for j in range(0,dim[1]):
        #        if H_sum[i][j] - np.conjugate(H_sum[j][i]) != 0.:
        #            print('Hermicity check fails!')     
    return H_sum

def ev_state_Ising(n, m, t, H_0, V): #, omega, h, g, pbc):
    H_tot = H_0 + V
    U = expm(- 1j * t * H_tot)
    #f0 = np.load('data/psi_N={}_omega={}_h={}_g={}_pbc={}.npy'.format(n, omega, h, g, pbc), allow_pickle = True)
    #e0 = np.conjugate(f0[:,m]) @ H_0 @ f0[:,m]
    f0 = psi(n)
    e0 = np.conjugate(f0[m]) @ H_0 @ f0[m]
    if np.abs(np.imag(e0)) > 1e-15:
            print('error: complex energies')
    #f = np.dot(U, f0[:,m])
    f = np.dot(U, f0[m])
    return e0, f

#def Loschmidt_echo(n, m, t, H_0, V):
#    f_left = np.conjugate(ev_state_Ising(n, m, t, H_0, V)[1]) 
#    f_right = ev_state_Ising(n, m, t, H_0, 0.)[1]
#    ff = f_left @ f_right
#    M = np.abs(ff) ** 2
#    return M

def p_Ising(n, m, t, H_0, V, omega, h, g, pbc): 
    fs = np.load('data/psi_N={}_omega={}_h={}_g={}_pbc={}.npy'.format(n, omega, h, g, pbc), allow_pickle = True)
    es = []
    ps = []
    E_in, psi_t = ev_state_Ising(n, m, t, H_0, V) #, omega, h, g, pbc)
    for l in range(len(fs)):
        energ = np.conjugate(fs[:,l]) @ H_0 @ fs[:,l]
        if np.abs(np.imag(energ)) > 1e-14:
            print('error: complex energies')
        es.append(np.real(energ))
        pop = np.abs(np.dot(np.conjugate(fs[:,l]), psi_t)) ** 2
        ps.append(pop)
    #es_order = []
    #ps_order = []    
    #for l in range(len(fs)):    
    #    es_order.append(sorted(zip(es, ps))[l][0])
    #    ps_order.append(sorted(zip(es, ps))[l][1])
    #return es_order, ps_order, E_in
    #spin_x = np.conjugate(psi_t) @ Y(n, 'x') @ psi_t
    #spin_y  = np.conjugate(psi_t) @ Y(n, 'y') @ psi_t
    #spin_z = np.conjugate(psi_t) @ Y(n, 'z') @ psi_t
    return es, ps #, spin_x, spin_y, spin_z

# the parameters: 
# num -- number of qubits
# num_ex = 0 (we use the ground state of sigma^x).
# Jc -- square root of the variance of random couplings J_{ij}
# nr -- number of realizations
# t_min -- mininum time
# t_max -- maximum time
# nt -- number of time points
# pbc - (1) periodic boundary conditions; (0) open boundary conditions

# generates and saves random couplings
def gen_coupl(num, nr, Jc):
    Js = [0] * nr
    for j in range(nr):
        Js[j] = J_SYK2(num, Jc)
    np.save('data/couplings_N={}_nr={}_J={}.npy'.format(num, nr, Jc), Js, allow_pickle = True) 

# runs the quench protocol for a chosen initial state num_ex
#def SYK2(num, num_ex, Jc, nr, t_min, t_max, nt, omega, h, g, pbc): 
def SYK2(num, num_ex, Jc, nr, omega, h, g, pbc): 
    #time = np.linspace(t_min, t_max, nt) 
    #arr_t_short = np.linspace(0, 10, 1001)[:1000]
    #arr_t_middle = np.linspace(10, 999, 990) 
    #arr_t_long = np.linspace(1000, 10000, 901)
    #time = np.concatenate((arr_t_short, arr_t_middle, arr_t_long))
    time = np.linspace(0, 500, 501)  #np.linspace(0, 100, 1001) 
    H_Ising = H_X(num, omega) + H_Z(num, h) + H_ZZ(num, g, pbc)
    energ = np.load('data/energ_N={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, omega, h, g, pbc), allow_pickle = True)

    Le = [[0] * nr for t in range(len(time))]
    Le2 = [[0] * nr for t in range(len(time))]
    Lp = [[0] * nr for t in range(len(time))] 
    E = [[0] * nr for t in range(len(time))] 
    
    M = [[0] * nr for t in range(len(time))] 
    E2 = [[0] * nr for t in range(len(time))] 
    EE2 = [[0] * nr for t in range(len(time))]
    #S = [[0] * nr for t in range(len(time))]
    
    #Spin_x = [[0] * nr for t in range(len(time))]
    #Spin_y = [[0] * nr for t in range(len(time))]
    #Spin_z = [[0] * nr for t in range(len(time))]
    #Spin2_x = [[0] * nr for t in range(len(time))]
    #Spin2_y = [[0] * nr for t in range(len(time))]
    #Spin2_z = [[0] * nr for t in range(len(time))]
     
    #coupling_consts = np.load('data_NN/couplings_N={}_nr={}_J={}.npy'.format(num, nr, Jc), allow_pickle = True)
    #coupling_consts = np.load('data/couplings_N={}_nr={}_J={}_pbc={}.npy'.format(num, nr, Jc, pbc), allow_pickle = True)
    coupling_consts = np.load('data/couplings_N={}_nr={}_J={}.npy'.format(num, nr, Jc), allow_pickle = True)
    
    E_in = np.real(ev_state_Ising(num, num_ex, 0., H_Ising, -H_Ising)[0])# , omega, g, pbc)[0])
    print(E_in)
    
    for j in range(nr):
        print(j)
        Js = coupling_consts[j]
        V = H_SYK2(num, Js) 
        for t in range(len(time)):
            Le[t][j], Lp[t][j] = p_Ising(num, num_ex, time[t], H_Ising, V, omega, h, g, pbc)
            delta = np.abs(np.array(Le[t][j]) - np.array(energ))
            if delta.any() > 1e-14:
                for aa in range(len(delta)):
                    if delta[aa] > 1e-14:
                        print(aa, delta[aa])
                        print('error: the energies do not coincide')
            Le2[t][j] = [q ** 2 for i, q in enumerate(Le[t][j])]    
            E[t][j] = np.array(Le[t][j]) @ np.array(Lp[t][j])
            
            #M[t][j] = Loschmidt_echo(num, num_ex, time[t], H_Ising, V)
            E2[t][j] = np.array(Le2[t][j]) @ np.array(Lp[t][j]) 
            EE2[t][j] =  E2[t][j] - E[t][j] ** 2
            #S[t][j] = - np.array(Lp[t][j]) @ np.log(np.array(Lp[t][j]))
            
            #Spin2_x[t][j] = Spin_x[t][j] ** 2
            #Spin2_y[t][j] = Spin_y[t][j] ** 2
            #Spin2_z[t][j] = Spin_z[t][j] ** 2
 
    pav = np.sum(np.array(Lp), 1) / nr
    err_av = np.std(np.array(Lp), 1, ddof = 1) / np.sqrt(nr)
    Eav = np.sum(np.array(E), 1)  / nr
    
    #Mav = np.sum(np.array(M), 1)  / nr
    E2av = np.sum(np.array(E2), 1)  / nr
    VarE = E2av  - Eav ** 2
    #Sav = np.sum(np.array(S), 1) / nr
    
    #Spin_xav = np.sum(np.array(Spin_x), 1) / nr
    #Spin_yav = np.sum(np.array(Spin_y), 1) / nr
    #Spin_zav = np.sum(np.array(Spin_z), 1) / nr
    #Spin2_xav = np.sum(np.array(Spin2_x), 1) / nr
    #Spin2_yav = np.sum(np.array(Spin2_y), 1) / nr
    #Spin2_zav = np.sum(np.array(Spin2_z), 1) / nr
    

    np.save('data/time_N={}.npy'.format(num), time, allow_pickle = True)
    np.save('data/es_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), Le, allow_pickle = True)
    np.save('data/p_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), Lp, allow_pickle = True)
    np.save('data/pav_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), pav, allow_pickle = True)
    np.save('data/err_av_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), err_av, allow_pickle = True)
    np.save('data/E_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), E, allow_pickle = True)
    np.save('data/Eav_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), Eav, allow_pickle = True)
    
    #np.save('data/M_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), M, allow_pickle = True)
    #np.save('data/Mav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Mav, allow_pickle = True)
    np.save('data/E2_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), E2, allow_pickle = True)
    np.save('data/EE2_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), EE2, allow_pickle = True)
    #np.save('data/S_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), S, allow_pickle = True)
    np.save('data/E2av_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), E2av, allow_pickle = True)
    np.save('data/VarE_N={}_M={}_nr={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, num_ex, nr, omega, h, g, pbc), VarE, allow_pickle = True)
    #np.save('data/Sav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Sav, allow_pickle = True)
    
    #np.save('data/Spin_xav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Spin_xav, allow_pickle = True)
    #np.save('data/Spin_yav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Spin_yav, allow_pickle = True)
    #np.save('data/Spin_zav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Spin_zav, allow_pickle = True)
    #np.save('data/Spin2_xav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Spin2_xav, allow_pickle = True)
    #np.save('data/Spin2_yav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Spin2_yav, allow_pickle = True)
    #np.save('data/Spin2_zav_N={}_M={}_nr={}_pbc={}.npy'.format(num, num_ex, nr, pbc), Spin2_zav, allow_pickle = True)
    
    # the data is saved into 'data_NN/...'

# runs the quench protocol for all initial states    
#def entropy(num, Jc, nr, t_min, t_max, nt, omega, g, pbc):
#    for m in range(0, 2 ** num):
#        SYK2(num, m, Jc, nr, t_min, t_max, nt, omega, g, pbc)

# saves the array of initial energies
def initial_energies(num, omega, h, g, pbc): 
    H_Ising = H = H_X(num, omega) + H_Z(num, h) + H_ZZ(num, g, pbc)
    M = int(2 ** num)
    E_in = [0] * M
    
    for m in range(M):
        E_in[m] = np.real(ev_state_Ising(num, m, 0., H_Ising, -H_Ising)[0]) #, omega, h, g, pbc)[0])
        #np.real(p_Ising(num, m, 0., H_Ising, - H_Ising)[2]) #, omega, h, g pbc)[2])
        
    np.save('data/initial_N={}_omega={}_h={}_g={}_pbc={}.npy'.format(num, omega, h, g, pbc), E_in, allow_pickle = True)