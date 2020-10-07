import numpy as np

def SVD(X):

        # decomposition
        U, s, VT = np.linalg.svd(X)
        invD = np.zeros((len(U),len(VT)))

        for i in range(0,len(VT)):
                invD[i,i]=1/s[i]

        UT = np.transpose(U)
        V = np.transpose(VT)

        return np.matmul(V,np.matmul(invD,UT))


if __name__ == "__main__":
    
    # generates a design matrix X of size (50x7) and y of size (50x1)
    N = 50
    X = np.random.randint(low=0,high=10,size=(N,7))
    y = np.random.randint(low=0,high=10,size=(N,1))

    # calculates beta coefficents
    beta_pinv = np.linalg.pinv(X) @ y
    beta_svd = SVD(X.T @ X) @ X.T @ y

    # rounds because floating point instability can cause the test to fail
    beta_pinv = np.round(beta_pinv, 4)
    beta_svd = np.round(beta_svd, 4)

    # prints result
    print(beta_pinv)
    print("")
    print(beta_svd)
    print("")
    print((beta_pinv == beta_svd).all())

    # - np.linalg.pinv(X) - is equivalent to - SVD(X.T @ X) @ X.T -
    assert (beta_pinv == beta_svd).all() == True

    