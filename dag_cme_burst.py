import numpy as np
from numpy import matlib

import scipy 
import scipy.integrate
import scipy.fft

import networkx as nx
import random   
import time

#https://ipyparallel.readthedocs.io/en/latest/dag_dependencies.html
def random_dag(nodes, edges, seed):
	"""Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
	n = nodes
	e = edges
	random.seed(seed)
	while True:
		nodes = n
		edges = e
		G = nx.DiGraph()
		for i in range(nodes):
			G.add_node(i)
		while edges > 0:
			a = random.randint(0,nodes-1)
			b=a
			while b==a:
				b = random.randint(0,nodes-1)
			G.add_edge(a,b)
			if nx.is_directed_acyclic_graph(G):
				edges -= 1
			else:
				# we closed a loop!
				G.remove_edge(a,b)
		if sum([G.in_degree[i]==0 for i in range(nodes)])==1 and nx.is_weakly_connected(G) and len(G.edges()) == e:
			break
	rt = np.where([G.in_degree[i]==0 for i in range(nnod)])[0][0]
	G = nx.relabel_nodes(G,{0:rt,rt:0},copy=True)

	return G


def construct_S(nrxn,nnod,G,n_deg):
	S = np.zeros((nrxn,nnod))
	spl_rxns = list(G.edges())
	c = 1
	for ed in spl_rxns:
		S[c][ed[0]] = -1
		S[c][ed[1]] = 1
		c+=1
	rt = np.where([G.in_degree[i]==0 for i in range(nnod)])[0][0]
	term = np.where([G.out_degree[i]==0 for i in range(nnod)])[0]
	deg_dom = np.arange(nnod)
	deg_dom = np.delete(deg_dom,term)
	deg_dom = np.delete(deg_dom,rt)
	deg = np.random.choice(deg_dom,n_deg-len(term))
	deg = np.append(deg,term)

	for i in range(len(deg)):
		S[c,deg[i]] = -1
		c+= 1

	S[:,[rt,0]]=S[:,[0,rt]]
	S[0,0] = -5
	return S

def construct_C(nnod,nedg,n_deg,G,k,S):
	C = np.zeros((nnod,nnod))
	spl_rxns = list(G.edges())
	for i in range(nedg):
		C[spl_rxns[i][0],spl_rxns[i][1]] = k[i+1]
	C-= np.diag(np.sum(C,1))
	for i in range(n_deg):
		degind = np.where(S[-(1+i)]==-1)[0][0]
		C[degind,degind] -= k[-(1+i)]    
	return C

def gillvec(k,t_matrix,S,nCells):
	k = np.matlib.repmat(k,nCells,1)
	n_species = S.shape[1]

	num_t_pts = t_matrix.shape[1]
	X_mesh = np.empty((nCells,num_t_pts,n_species),dtype=int) #change to float if storing floats!!!!!!! 
	X_mesh[:] = np.nan

	t = np.zeros(nCells,dtype=float)
	tindex = np.zeros(nCells,dtype=int)

	#initialize state: gene,integer unspliced, integer spliced 
	X = np.zeros((nCells,n_species))

	#initialize list of cells that are being simulated
	simindices = np.arange(nCells)
	activecells = np.ones(nCells,dtype=bool)

	while any(activecells):
		mu = np.zeros(nCells,dtype=int);
		n_active_cells = np.sum(activecells)
		
		(t_upd,mu_upd) = rxn_calculator( \
			X[activecells,:], \
			t[activecells], \
			k[activecells,:], \
			S, \
			n_active_cells)

		t[activecells] = t_upd
		mu[activecells] = mu_upd
		
		tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
		update = np.zeros(nCells,dtype=bool)
		update[activecells] = t[activecells]>tvec_time
		
		while any(update):
			tobeupdated = np.where(update)[0]
			for i in range(len(tobeupdated)):
				X_mesh[simindices[tobeupdated[i]],tindex[tobeupdated[i]],:] = \
					X[tobeupdated[i],:]
			
			tindex = tindex + update;
			ended_in_update = tindex[update]>=num_t_pts;

			if any(ended_in_update):
				ended = tobeupdated[ended_in_update];
				
				activecells[ended] = False;
				mu[ended] = 0;

				if ~any(activecells):
					break			
			
			tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
			update = np.zeros(nCells,dtype=bool)
			update[activecells] = t[activecells]>tvec_time
		
		z_ = np.where(activecells)[0]
		not_burst = mu[z_] > 1
		burst = mu[z_] == 1
		if any(not_burst):
			X[z_[not_burst]] += S[mu[z_[not_burst]]-1]
		if any(burst):				 
			bs = np.random.geometric(1/(1+S[0][0]),size=(sum(burst),1))-1
			X[z_[burst]] += np.matlib.hstack((bs,np.zeros((sum(burst),n_species-1),dtype=int)))
	return X_mesh

def rxn_calculator(X,t,k,S,nCells):
	nRxn = S.shape[0]

	kinit = k[:,0]

	a = np.zeros((nCells,nRxn),dtype=float)
	a[:,0] = kinit
	for i in range(1,nRxn):
		ind = np.where(S[i,:]==-1)[0][0]
		a[:,i] = X[:,ind]*k[:,i]

	a0 = np.sum(a,1)
	t += np.log(1./np.random.rand(nCells)) / a0
	r2ao = a0 * np.random.rand(nCells)
	mu = np.sum(np.matlib.repmat(r2ao,nRxn+1,1).T >= np.cumsum(np.matlib.hstack((np.zeros((nCells,1)),a)),1) ,1)
	return (t,mu)

# def compute_exp(S,k):
# 	q = S.shape[0]-1
# 	n = S.shape[1]
# 	r = np.zeros(n)
# 	for i in range(n):
# 		r[i] = np.sum(k[np.where(S[:,i]==-1)[0]])	
# 	return r



def cme_integrator(L,V,Vinv,k,mx,bs,t,TIME=False):
	t1 = time.time()
	u = []
	for i in range(len(mx)):
		l = np.arange(mx[i])
		u_ = np.exp(-2j*np.pi*l/mx[i])-1
		u.append(u_)
	g = np.meshgrid(*[u_ for u_ in u])
	for i in range(len(mx)):
		g[i] = g[i].flatten()
	g = np.asarray(g)
 
	coeff = compute_coeff(L,V,Vinv,g)
	t_coeff = time.time()-t1

	t1 = time.time()
	fun = lambda x: INTFUNC(x,L,coeff,bs)  
	I = scipy.integrate.quad_vec(fun,0,t)[0]
	t_integral = time.time()-t1

	t1 = time.time()
	I = np.exp(I*k[0])
	I = np.reshape(I,mx)
	FFTRES = np.fft.ifftn(I)
	t_fft = time.time()-t1
	FFTRES = np.squeeze(np.real(FFTRES))
	if not TIME:
		return FFTRES
	if TIME:
		return FFTRES, t_coeff, t_integral, t_fft

def INTFUNC(x,L,coeff,bs):
	Ufun = bs*(np.exp(L*x)@coeff)
	return Ufun/(1-Ufun)

def compute_mean(L,V,Vinv,k,bs,i):
    n = len(L)
    g = np.zeros(n) 
    g[i] = 1
    g = g[:,np.newaxis]
    a = compute_coeff(L,V,Vinv,g)
    L = L[:,np.newaxis]
    return k[0]*bs*np.sum(a/-L)

def compute_cov(L,V,Vinv,k,bs,i,j):
    n = len(L)
    g = np.zeros((n,2))
    g[i,0] = 1
    g[j,1] = 1
    a = compute_coeff(L,V,Vinv,g)
    v = 0
    for i_ in range(n):
        for j_ in range(n):
            v += -a[i_,0]*a[j_,1]/(L[i_]+L[j_])
    v *= 2*k[0]*bs**2
    if i==j:
        v += compute_mean(L,V,Vinv,k,bs,i)
    return v

def compute_coeff(L,V,Vinv,u):
    n_u = u.shape[1]
    a = np.asarray([(V@np.diag( Vinv @ u[:,i]))[0] for i in range(n_u)]).T
    return a

def compute_eigs(C):
    L,V = np.linalg.eig(C)
    Vinv = np.linalg.inv(V)
    return L,V,Vinv