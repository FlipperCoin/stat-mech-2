# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Function Definitions

# %%
def get_pairs(arr):
    _, counts = np.unique(arr, return_counts=True)
    return np.sum(counts * (counts-1) / 2)

# %%
def get_l(board):
    return int(np.sqrt(board.shape[0]))

# %%
def get_total_pairs(board):
    pairs = 0
    l = get_l(board)
    for i in range(board.shape[0]):
        pairs += get_pairs(board[i])   
    for j in range(board.shape[1]):
        pairs += get_pairs(board[:,j])
    for sqi in range(l):
        for sqj in range(l):
            pairs += get_pairs(board[sqi*l:(sqi+1)*l,sqj*l:(sqj+1)*l])

    return pairs

# %%
def get_E(board):
    l = get_l(board)
    pairs = get_total_pairs(board)
    E = pairs / (l**4/2 * (l**2 + 2*l - 3))
    return E

# %%
def get_pairs_from_E(E,l):
    return E * (3/2 * (l**4) * ((l**2) - 1))

# %%
def metropolis_board(board, Tspace, Mspace):
    current_board=board.copy()
    l = int(np.sqrt(current_board.shape[0]))

    Es = []
    for idx,T in enumerate(Tspace):

        beta = 1/T if T!=0 else None

        for i in range(Mspace[idx]):

            i1,j1 = np.random.randint(0,l**2), np.random.randint(0,l**2)
            i2,j2 = np.random.randint(0,l**2), np.random.randint(0,l**2)
            val1 = current_board[i1,j1]
            val2 = current_board[i2,j2]

            E_before = get_E(current_board)

            current_board[i1,j1] = val2
            current_board[i2,j2] = val1

            E_after = get_E(current_board)

            A=1
            if E_before < E_after:
                if beta != None:
                    A=np.exp(-beta*(E_after-E_before))
                else:
                    A=0

            take = np.random.random() < A
            if not take:
                current_board[i1,j1] = val1
                current_board[i2,j2] = val2
                Es += [E_before]
            else:
                Es += [E_after]
    
    return current_board, np.array(Es)

# %%
def plot_alg(Es, Tspace, Mspace, log=False):
    plt.figure(dpi=130)
    ranges = np.cumsum([0]+Mspace)
    for i,T in enumerate(Tspace):
        plt.plot(range(ranges[i],ranges[i+1]),Es[ranges[i]:ranges[i+1]],".",label=f"T={T}")
    plt.xlabel(f"Step Number")
    plt.ylabel(r"E $\left[J\right]$")
    if log:
        plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.show()

# %%
def get_board(l):
    return np.repeat(np.arange(1,l**2+1,1).reshape((1,l**2)),l**2,0)

# %% [markdown]
# # Sanity check

# %%
l=2
b=np.zeros((l**2,l**2))
for i in range(l):
    for j in range(l):
        b[i*l:(i+1)*l,j*l:(j+1)*l]=i*l+j+1

print(b)
print(get_E(b))

# %% [markdown]
# # Board solutions

# %% [markdown]
# ## $l=2$

# %%
boardl2 = get_board(l=2)
print(boardl2)
print(get_E(boardl2))
print(get_total_pairs(boardl2))

# %%
Tspace_l2_T0 = [0]
Mspace = [300]
boardl2_ground_T0, Es_l2_T0 = metropolis_board(boardl2,Tspace_l2_T0,Mspace) 

# %%
print(f"Init state E: {get_E(boardl2)}, ground state E: {Es_l2_T0[-1]}")
print(f"Init state pairs: {get_total_pairs(boardl2)}, ground state pairs: {get_total_pairs(boardl2_ground_T0)}")
plot_alg(Es_l2_T0, Tspace_l2_T0, Mspace)
print(boardl2_ground_T0)

# %%
Tspace_l2_T = [0.05,0.01,0.005]
Mspace = [240,240,240]
boardl2_ground_T, Es_l2_T = metropolis_board(boardl2,Tspace_l2_T,Mspace) 

# %%
print(f"Init state E: {get_E(boardl2)}, ground state E: {Es_l2_T[-1]}")
print(f"Init state pairs: {get_total_pairs(boardl2)}, ground state pairs: {get_total_pairs(boardl2_ground_T)}")
plot_alg(Es_l2_T, Tspace_l2_T, Mspace)
print(boardl2_ground_T)

# %% [markdown]
# ## $l=3$

# %%
boardl3 = get_board(l=3)
print(boardl3)
print(get_E(boardl3))
print(get_total_pairs(boardl3))

# %%
Tspace_l3_T0 = [0]
Mspace = [25000]
boardl3_ground_T0, Es_l3_T0 = metropolis_board(boardl3,Tspace_l3_T0,Mspace) 

# %%
print(f"Init state E: {get_E(boardl3)}, ground state E: {Es_l3_T0[-1]}")
print(f"Init state pairs: {get_total_pairs(boardl3)}, ground state pairs: {get_total_pairs(boardl3_ground_T0)}")
plot_alg(Es_l3_T0, Tspace_l3_T0,Mspace)
print(boardl3_ground_T0)

# %%
Tspace_l3_T = [0.003,0.001,1e-4,1e-5]
Mspace =[6500,6500,6500,6500]
boardl3_ground_T, Es_l3_T = metropolis_board(boardl3,Tspace_l3_T,Mspace) 

# %%
print(f"Init state E: {get_E(boardl3)}, ground state E: {Es_l3_T[-1]}")
print(f"Init state pairs: {get_total_pairs(boardl3)}, ground state pairs: {get_total_pairs(boardl3_ground_T)}")
plot_alg(Es_l3_T, Tspace_l3_T,Mspace)
print(boardl3_ground_T)

# %% [markdown]
# ## $l=4$

# %%
boardl4 = get_board(l=4)
print(boardl4)
print(get_E(boardl4))
print(get_total_pairs(boardl4))

# %%
Tspace_l4_T0 = [0]
Mspace=[65000]
boardl4_ground_T0, Es_l4_T0 = metropolis_board(boardl4,Tspace_l4_T0,Mspace) 

# %%
print(f"Init state E: {get_E(boardl4)}, ground state E: {Es_l4_T0[-1]}")
print(f"Init state pairs: {get_total_pairs(boardl4)}, ground state pairs: {get_total_pairs(boardl4_ground_T0)}")
plot_alg(Es_l4_T0, Tspace_l4_T0,Mspace)
print(boardl4_ground_T0)

# %%
Tspace_l4_T = [1e-3,1e-4,1e-5,1e-6]
Mspace_l4_T = [5000,20000,20000,20000]
boardl4_ground_T, Es_l4_T = metropolis_board(boardl4,Tspace_l4_T,Mspace=Mspace_l4_T) 

# %%
print(f"Init state E: {get_E(boardl4)}, ground state E: {Es_l4_T[-1]}")
print(f"Init state pairs: {get_total_pairs(boardl4)}, ground state pairs: {get_total_pairs(boardl4_ground_T)}")
plot_alg(Es_l4_T, Tspace_l4_T, Mspace_l4_T)
print(boardl4_ground_T)


