import torch

def fitNthOrderEuqation(points, n, device, weights=None):
    X = points[..., 0]  # (D, 1)
    Y = points[..., 1]  # (D, 1)
    if weights is None:
        weights = torch.ones(X.shape)
    A = torch.zeros((n+1, n+1), device=device)
    b = torch.zeros((n+1, 1), device=device)
    X_power = [weights, X * weights]
    for i in range(2, n * 2 + 1):
        X_power.append(X * X_power[-1])

    for i in range(n + 1): 
        b[i] = torch.sum(X_power[i] * Y)

    for j in range(n + 1):
        for i in range(n + 1):
            index = i + j 
            A[i][j] = torch.sum(X_power[index])
            A[j][i] = A[i][j]

    x = torch.linalg.solve(A, b)
    return x


def getCurveFunction(points, device, weights=None):
    coeffs = fitNthOrderEuqation(points, 3, device, weights)

    def polynominal3rd(coeffs, X):
        X2 = X * X
        X3 = X2 * X
        return coeffs[0] + coeffs[1] * X + coeffs[2] * X2 + coeffs[3] * X3

    def poly(coeffs):
        def poly_(X):
            return polynominal3rd(coeffs, X)
        return poly_

    return poly(coeffs)

