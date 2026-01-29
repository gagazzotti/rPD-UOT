import numpy as np

# Dimensions
n, p, m = 3, 3, 3  # A : n x p, B : n x m

# np.random.seed(0)
A = np.random.randn(n, p)
B = np.random.randn(n, m)

I_n = np.eye(n)

# Blocs nuls avec les dimensions correctes
zero_n_m = np.zeros((n, m))
zero_n_n = np.zeros((n, n))
zero_p_m = np.zeros((p, m))
zero_p_n = np.zeros((p, n))

# Construire M correctement
M = np.block(
    [
        [A, B, zero_n_n],  # n × (p+m+n)
        [zero_p_n, zero_p_m, B.T],  # p × (n+m+n)
        [I_n, zero_n_m, A.T],  # n × (n+m+n)
    ]
)

print("M =\n", M)

# Pseudo-inverse
M_inv = np.linalg.pinv(M)

print("\nPseudo-inverse de M :\n", np.round(M_inv, 3))

# Vérification
I_check = M @ M_inv
print("\nM @ M_inv =\n", I_check)
print(
    "\nErreur par rapport à l'identité :",
    np.linalg.norm(I_check - np.eye(M.shape[0])),
)
