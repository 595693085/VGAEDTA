def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse