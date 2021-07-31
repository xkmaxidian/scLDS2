
import numpy as np
import pandas as pd
def f(X,Y):
    X = X.as_matrix()
    Y = Y.as_matrix()
    X=X.T
    Y=Y.T
    print(X.shape)
    print(Y.shape)
    alpha=0.1
    I=np.eye(39,39)
    B=np.dot(Y,Y.T)+alpha*I
    B_I=np.linalg.inv(B)
    A=np.dot(X,Y.T)
    return np.dot(A,B_I)


def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    df = pd.DataFrame(list(lst))
    df.to_csv('zeisel_coef.csv', sep=',', header=True, index=True)
    # print(lst)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

