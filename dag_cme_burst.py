import numpy as np
from numpy import matlib

import scipy 
import scipy.integrate
import scipy.fft

import networkx as nx
import random   

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
	return G


def construct_S(nrxn,nnod,G,n_deg):
	S = np.zeros((nrxn,nnod))
	adj = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
	c = 1
	for i in range(nnod):
		for j in range(nnod):
			if adj[i,j] == 1:
				S[c,i] = -1
				S[c,j] = 1
				c += 1
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

def compute_exp(S,k):
	q = S.shape[0]-1
	n = S.shape[1]
	r = np.zeros(n)
	for i in range(n):
		r[i] = np.sum(k[np.where(S[:,i]==-1)[0]])	
	return r

def compute_coeff(S,k,g,r):
	# q = S.shape[0]-1
	n = S.shape[1]
	M = len(g[0])
	A = np.zeros((n,n,M),dtype=np.complex128)
	deg_rxn = np.where(np.sum(S,1)==-1)[0]
	deg_spec = np.array([np.where(S[i,:]==-1)[0][0] for i in deg_rxn])

	#determine terminal nodes
	term_spec = []
	for s in deg_spec:
		r_ = np.where(S[:,s] == -1)[0]
		no_iso = True
		for rx in r_:
			if np.any(S[rx,:]==1):
				no_iso = False
		if no_iso:
			term_spec.append(s)
	term_spec = np.array(term_spec)

	for t in term_spec:
		A[t,t,:] = g[t] #set terminal

	#initialize set of computed species
	C = list(term_spec)
	species_set = np.arange(n)

	while len(C)<n:
		st = np.delete(species_set,C)
		for s in st:
			r_i = np.where(S[:,s] == -1)[0]
			r_i = np.array(list(set(r_i).difference(set(deg_rxn)))) #delete the degradation reactions. this list will not have any of the terminal species!
			
			P = np.array([np.where(S[i,:]==1)[0][0] for i in r_i]) #identify the products
			if set(P).issubset(set(C)):
				for j in range(len(P)):
					f =  k[r_i[j]] / (r[s] - r)
					
					f[s] = 0
					A[s,:,:] += A[P[j],:,:] * np.matlib.repmat(f,M,1).T
				A[s,s,:] = g[s] - np.sum(A[s,:,:],0)
				C.append(s)
	return np.squeeze(A[0,:,:]).T


def cme_integrator(k,mx,S,t):
	u = []
	for i in range(len(mx)):
		l = np.arange(mx[i])
		u_ = np.exp(-2j*np.pi*l/mx[i])-1
		u.append(u_)
	g = np.meshgrid(*[u_ for u_ in u])
	for i in range(len(mx)):
		g[i] = g[i].flatten()
	r = compute_exp(S,k)
	coeff = compute_coeff(S,k,g,r)

	b = S[0,0]
	fun = lambda x: INTFUNC_(x,k,r,coeff,b)  
	I = scipy.integrate.quad_vec(fun,0,t)[0]
	
	I = np.exp(I*k[0])
	I = np.reshape(I,mx)
	return np.squeeze(np.real(np.fft.ifftn(I)))
	
def INTFUNC_(x,k,r,coeff,b):
	M = coeff.shape[0]
	Ufun = b*np.sum(np.matlib.repmat(np.exp(-r*x),M,1)*coeff,1)
	return Ufun/(1-Ufun)

def compute_mean(S,k,i):
	r = compute_exp(S,k)
	n = S.shape[1]
	g = [np.array([0])]*n 
	g[i] = np.array([1])
	a = np.real(compute_coeff(S,k,g,r))
	return k[0]*S[0,0]*sum(a/r)

def compute_cov(S,k,i,j):
	r = compute_exp(S,k)
	n = S.shape[1]
	g1 = [np.array([0])]*n 
	g2 = g1.copy()
	g1[i] = np.array([1])
	g2[j] = np.array([1])

	a1 = np.real(compute_coeff(S,k,g1,r))
	a2 = np.real(compute_coeff(S,k,g2,r))
	
	v = 0
	for i_ in range(n):
		for j_ in range(n):
			v += a1[i_]*a2[j_]/(r[i_]+r[j_])
	v *= 2*k[0]*S[0,0]**2
	if i==j:
		v += compute_mean(S,k,i)
	return v