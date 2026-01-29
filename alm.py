import matplotlib.pyplot as plt
import numpy as np
import ot  # POT


# -----------------------------
# Augmented Lagrangian OT
# -----------------------------
def ot_augmented_lagrangian(C, mu, nu, gamma=10.0, max_iter=10000, tol=1e-8):
    n, m = C.shape
    f = np.zeros(n)
    g = np.zeros(m)

    for _ in range(max_iter):
        # Primal update
        X = np.maximum(0.0, f[:, None] + g[None, :] - C) / gamma

        # Redualisation
        u = f + gamma * (mu - X.sum(axis=1))
        v = g + gamma * (nu - X.sum(axis=0))

        # Update
        f, g = u, v

        # stopping criterion
        err = max(
            np.linalg.norm(mu - X.sum(axis=1), 1),
            np.linalg.norm(nu - X.sum(axis=0), 1),
        )
        if err < tol:
            print("Err")
            break

    return X


# -----------------------------
# Problem generation
# -----------------------------
np.random.seed(0)

n = 20
m = 20

# random points in [0,1]
x = np.random.rand(n, 2)
y = np.random.rand(m, 2)

# cost matrix (squared Euclidean)
C = ot.dist(x, y, metric="euclidean") ** 2

# normalized histograms
mu = np.random.rand(n)
nu = np.random.rand(m)
mu /= mu.sum()
nu /= nu.sum()

# -----------------------------
# Solve with POT (exact OT)
# -----------------------------
X_pot = ot.emd(mu, nu, C)

# -----------------------------
# Solve with ALM
# -----------------------------
X_alm = ot_augmented_lagrangian(C, mu, nu, gamma=20000.0)


# -----------------------------
# Comparisons
# -----------------------------
def cost(X):
    return np.sum(C * X)


print("=== Costs ===")
print(f"POT cost : {cost(X_pot):.6f}")
print(f"ALM cost : {cost(X_alm):.6f}")

print("\n=== Marginal errors ===")
print("POT mu error :", np.linalg.norm(mu - X_pot.sum(axis=1), 1))
print("POT nu error :", np.linalg.norm(nu - X_pot.sum(axis=0), 1))
print("ALM mu error :", np.linalg.norm(mu - X_alm.sum(axis=1), 1))
print("ALM nu error :", np.linalg.norm(nu - X_alm.sum(axis=0), 1))

print("\n=== Plan difference ===")
print("||X_ALM - X_POT||_1 =", np.linalg.norm(X_alm - X_pot, 1))


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(X_pot)
plt.title("POT plan")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(X_alm)
plt.title("ALM plan")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(X_alm - X_pot)
plt.title("Difference")
plt.colorbar()

plt.tight_layout()
plt.show()
