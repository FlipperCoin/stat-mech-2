# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

# %%
def get_e(board, i, j):
    l = int(board.shape[0])
    
    e = 0
    for k in [-1,1]:
        if board[(i + k)%l,(j)%l] == board[i,j]:
            e -= 1
        else:
            e += 1

    for k in [-1,1]:
        if board[(i)%l,(j+k)%l] == board[i,j]:
            e -= 1
        else:
            e += 1


    return e

# %%
def get_E(board):
    l = int(board.shape[0])

    E = 0 
    for i in range(l):
        for j in range(l):
            e = 0
            # neighbour below
            if board[(i + 1)%l,(j)%l] == board[i,j]:
                e -= 1
            else:
                e += 1
            
            #neighbour to the right
            if board[(i)%l,(j+1)%l] == board[i,j]:
                e -= 1
            else:
                e += 1
            
            E += e

    return E

# %%
def metropolis_step(board, beta, E, periodic=True):
    l = int(board.shape[0])

    if periodic:
        i1,j1 = np.random.randint(0,l), np.random.randint(0,l)
    else:
        i1,j1 = np.random.randint(0,l), np.random.randint(1,l-1)
    val1 = board[i1,j1]

    e = get_e(board, i1, j1)
    
    E_after = E - 2*e

    A=1
    if E < E_after:
        if beta != None:
            A=np.exp(-beta*(E_after-E))
        else:
            A=0

    take = np.random.random() < A
    if take:
        board[i1,j1] = -val1
        E = E_after

    return E


# %%
def metropolis_board(board, Tspace, periodic=True):
    current_board=board.copy()
    l = int(current_board.shape[0])

    E_all = []
    boards = []
    E = get_E(board)
    for j,T in enumerate(tqdm(list(Tspace))):

        beta = 1/T if T!=0 else None

        req = 125000
        Et = [None]*req

        # equilibrium with T
        E_min = E
        count=0
        while count < req:
            E = metropolis_step(current_board, beta, E, periodic)
            Et[count] = E
            count += 1
            if E < E_min:
                E_min = E
                count=0
        # # statistics
        # for i in range(int(50000)):
        #     E = metropolis_step(current_board, beta, E, periodic)
        #     Et += [E]

        if j%int(len(Tspace)/5) == 0:
            boards += [current_board.copy()]

        if j%int(len(Tspace)/15) == 0:
            print(f"status - spin up: {100*np.sum(current_board == 1)/l**2:.2f}%, spin down: {100*np.sum(current_board == -1)/l**2:.2f}%")

        # if j == len(Tspace)-1 and periodic:
        #     counter = 0
        #     while not np.all(current_board == 1) and not np.all(current_board == -1):
        #         if counter%20000==0:
        #             print(f"finalizing solution... spin up: {100*np.sum(current_board == 1)/l**2:.2f}%, spin down: {100*np.sum(current_board == -1)/l**2:.2f}%")
        #         metropolis_step(current_board, beta, E)
        #         Et += [E]

        #         counter += 1

        E_all += [Et]
        # df_T = pd.DataFrame({"E": Et})
        # df_T["T"] = T
        # dfs += [df_T]
    
    # df = pd.concat(dfs,ignore_index=True)
    E_arr = np.stack(E_all[::-1])
    return current_board, E_arr, boards

# %% [markdown]
# ## Periodic

# %%
L=128
np.random.seed(42)
board = 2*np.random.randint(0,2,(L,L))-1
plt.imshow(board)
print(f"Init spin up: {np.sum(board == 1)/L**2:.2f}%, spin down: {np.sum(board == -1)/L**2:.2f}%")

# %%
Tspace = np.concatenate([np.arange(0.1,1.5,0.01),np.arange(1.5,3,0.0025),np.arange(3,4,0.01)])
# Tspace = np.arange(0.1,4,0.01)
Tspace = np.round(Tspace, 5)

# %%
np.random.seed(42)
new_board, data, boards = metropolis_board(board,Tspace[::-1],periodic=True)
# new_board, df = metropolis_board(board,np.arange(0.1,4,0.01)[::-1])

# %%
datetime_str = datetime.now().strftime(r"%d%m%y%H%M%S")
os.mkdir(f'results/{datetime_str}')
np.save(f'results/{datetime_str}/data', data)
np.save(f'results/{datetime_str}/board_init',board)
np.save(f'results/{datetime_str}/board',new_board)
np.save(f'results/{datetime_str}/Tspace',Tspace)
for i,board_i in enumerate(boards):
    np.save(f'results/{datetime_str}/board{i}',board_i)

# path = f'results/230324132113'
# df = pd.read_csv(f'{path}/data')
# board = np.load(f'{path}/board.npy')
# Tspace = np.load(f'{path}/Tspace.npy')

# %%
for board in boards:
    plt.figure()
    plt.imshow(board,vmin=-1,vmax=1)
    cbar = plt.colorbar(ticks=[-1,1])

plt.figure()
plt.imshow(new_board,vmin=-1,vmax=1)
cbar = plt.colorbar(ticks=[-1,1])

# %%
Emean = np.mean(data, 1)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$\left<E\right>$')
plt.plot(Tspace, Emean, '.')

# %%
Emean = np.mean(data, 1)
Emean_smooth = np.convolve(Emean, 1/7 * np.ones(7), 'valid')
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$\left<E\right>$')
plt.plot(Tspace[3:-3], Emean_smooth, '.')


# %%
Emean = np.mean(data,1)
Emean=Emean.reshape((len(Emean)))
Emean_smooth = np.convolve(Emean, 1/7 * np.ones(7),'valid')
Tred = Tspace[3:-3]
dE = ((Emean_smooth[2:]-Emean_smooth[:-2])/(Tred[2:]-Tred[:-2]))
plt.plot((Tred[2:] + Tred[:-2])/2, dE)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$C_v$')

# %%
Evar = np.var(data,1)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$\left< \left( E-\left< E \right> \right) ^2 \right>$')
plt.plot(Tspace, Evar, '.')
# plt.yscale('log')

# %%
Cv = np.var(data,1)/(Tspace**2)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$C_v$')
# plt.yscale('log')
plt.plot(Tspace, Cv)

# %%
S = (np.log(2) + np.cumsum((Cv[:-1] / Tspace[:-1])* (Tspace[1:] - Tspace[:-1]))) / L**2
plt.plot(Tspace[1:], S, '.')
plt.grid()
plt.ylabel('S/N')
plt.xlabel('T')

# %%
F = Emean[1:]/L**2 - Tspace[1:] * (S)
plt.plot(Tspace[1:], F, '-', markersize=2)
plt.grid()
plt.ylabel('F/N')
plt.xlabel('T')

# %% [markdown]
# ## Constant Walls

# %%
L=128
np.random.seed(42)
board = 2*np.random.randint(0,2,(L,L))-1
board[:,0]=1
board[:,-1]=-1
plt.imshow(board)
print(f"Init spin up: {np.sum(board == 1)/L**2:.2f}%, spin down: {np.sum(board == -1)/L**2:.2f}%")

# %%
Tspace = np.concatenate([np.arange(0.1,1.5,0.01),np.arange(1.5,3,0.0025),np.arange(3,4,0.01)])
# Tspace = np.arange(0.1,4,0.1)[::-1]
Tspace = np.round(Tspace, 5)

# %%
np.random.seed(42)
new_board, data, boards = metropolis_board(board,Tspace[::-1],periodic=False)
# new_board, df = metropolis_board(board,np.arange(0.1,4,0.01)[::-1])

# %%
datetime_str = datetime.now().strftime(r"%d%m%y%H%M%S")
os.mkdir(f'results/{datetime_str}')
np.save(f'results/{datetime_str}/data', data)
np.save(f'results/{datetime_str}/board_init',board)
np.save(f'results/{datetime_str}/board',new_board)
np.save(f'results/{datetime_str}/Tspace',Tspace)
for i,board_i in enumerate(boards):
    np.save(f'results/{datetime_str}/board{i}',board_i)

# path = f'results/230324132113'
# df = pd.read_csv(f'{path}/data')
# board = np.load(f'{path}/board.npy')
# Tspace = np.load(f'{path}/Tspace.npy')

# %%
for board in boards:
    plt.figure()
    plt.imshow(board,vmin=-1,vmax=1)
    cbar = plt.colorbar(ticks=[-1,1])

plt.figure()
plt.imshow(new_board,vmin=-1,vmax=1)
cbar = plt.colorbar(ticks=[-1,1])

# %%
Emean = np.mean(data, 1)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$\left<E\right>$')
plt.plot(Tspace, Emean, '.')

# %%
Emean = np.mean(data, 1)
Emean_smooth = np.convolve(Emean, 1/7 * np.ones(7), 'valid')
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$\left<E\right>$')
plt.plot(Tspace[3:-3], Emean_smooth, '.')


# %%
Emean = np.mean(data,1)
Emean=Emean.reshape((len(Emean)))
Emean_smooth = np.convolve(Emean, 1/7 * np.ones(7),'valid')
Tred = Tspace[3:-3]
dE = ((Emean_smooth[2:]-Emean_smooth[:-2])/(Tred[2:]-Tred[:-2]))
plt.plot((Tred[2:] + Tred[:-2])/2, dE)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$C_v$')

# %%
Evar = np.var(data,1)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$\left< \left( E-\left< E \right> \right) ^2 \right>$')
plt.plot(Tspace, Evar, '.')
# plt.yscale('log')

# %%
Cv = np.var(data,1)/(Tspace**2)
plt.grid()
plt.xlabel('T')
plt.ylabel(r'$C_v$')
# plt.yscale('log')
plt.plot(Tspace, Cv)

# %%
S = (np.log(2) + np.cumsum((Cv[:-1] / Tspace[:-1])* (Tspace[1:] - Tspace[:-1]))) / L**2
plt.plot(Tspace[1:], S, '.')
plt.grid()
plt.ylabel('S/N')
plt.xlabel('T')

# %%
F = Emean[1:]/L**2 - Tspace[1:] * (S)
plt.plot(Tspace[1:], F, '-', markersize=2)
plt.grid()
plt.ylabel('F/N')
plt.xlabel('T')

# %% [markdown]
# ## Final Plots

# %%
from scipy.integrate import cumtrapz

# %%
periodic='250324153635'
edge='250324171636'

data_periodic = np.load(f'results/{periodic}/data.npy')
data_edge = np.load(f'results/{edge}/data.npy')

# %%
Cv_periodic = np.var(data_periodic,1)/(Tspace**2)
S_periodic = (cumtrapz(Cv_periodic/Tspace, Tspace, initial=np.log(2))) / L**2
# S_periodic = (np.log(2) + np.cumsum((Cv_periodic[:-1] / Tspace[:-1])* (Tspace[1:] - Tspace[:-1]))) / L**2
F_periodic = np.mean(data_periodic,1)/L**2 - Tspace * (S_periodic)

# %%
Cv_edge = np.var(data_edge,1)/(Tspace**2)
S_edge = (cumtrapz((Cv_edge/Tspace), Tspace, initial=np.log(2*L)) / L**2)
F_edge = np.mean(data_edge,1)/L**2 - Tspace * (S_edge)

# %%
plt.figure(dpi=130)
plt.plot(Tspace, F_periodic, '-', label='free')
plt.plot(Tspace, F_edge, '-', label='bounds')
plt.grid()
plt.plot()
plt.ylabel(r'F/N $\left[ J \right]$')
plt.xlabel(r'T $\left[ J \right]$')
plt.legend()
plt.show()

# %%
plt.figure(dpi=130)
plt.plot(Tspace, S_periodic, '.', label='free',markersize=2)
plt.plot(Tspace, S_edge, '.', label='bounds',markersize=2)
plt.grid()
plt.plot()
plt.ylabel(r'S/N $\left[ J \right]$')
plt.xlabel(r'T $\left[ J \right]$')
plt.legend()
plt.show()

# %%
plt.figure(dpi=130)
plt.plot(Tspace, F_edge-F_periodic, '-')
plt.grid()
plt.plot()
plt.ylabel(r'$\Delta F$/N $\left[ J \right]$')
plt.xlabel(r'T $\left[ J \right]$')
plt.show()

# %%
print((F_edge-F_periodic)[0] * L**2)
print(np.mean((F_edge-F_periodic)[-20:]) * L**2)

# %%
print((Tspace*(S_edge-S_periodic) * L**2)[-1])

# %%
for i in range(5):
    boardi = np.load(f'results/{periodic}/board{i}.npy')
    plt.figure(dpi=130)
    plt.imshow(boardi,vmin=-1,vmax=1)
    plt.xticks([])
    plt.yticks([])
    # cbar = plt.colorbar(ticks=[-1,1])
    plt.savefig(f'plots/periodic_board{i}.png',bbox_inches='tight')
    plt.show()
board_end = np.load(f'results/{periodic}/board.npy')
plt.figure(dpi=130)
plt.imshow(board_end,vmin=-1,vmax=1)
# cbar = plt.colorbar(ticks=[-1,1])
plt.xticks([])
plt.yticks([])
plt.savefig(f'plots/periodic_board_end.png',bbox_inches='tight')
plt.show()

# %%
for i in range(5):
    boardi = np.load(f'results/{edge}/board{i}.npy')
    plt.figure(dpi=130)
    plt.imshow(boardi,vmin=-1,vmax=1)
    plt.xticks([])
    plt.yticks([])
    # cbar = plt.colorbar(ticks=[-1,1])
    plt.savefig(f'plots/edge_board{i}.png',bbox_inches='tight')
    plt.show()
board_end = np.load(f'results/{edge}/board.npy')
plt.figure(dpi=130)
plt.imshow(board_end,vmin=-1,vmax=1)
# cbar = plt.colorbar(ticks=[-1,1])
plt.xticks([])
plt.yticks([])
plt.savefig(f'plots/edge_board_end.png',bbox_inches='tight')
plt.show()

# %%
a=int(len(Tspace)/5)
print(Tspace[-1],Tspace[-1-a],Tspace[-1-2*a],Tspace[-1-3*a],Tspace[-1-4*a],Tspace[0])


