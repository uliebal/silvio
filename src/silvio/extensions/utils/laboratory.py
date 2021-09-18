


def ErrorRate(Invest, ResTotal= 5000, relKM=.005, Vmax=.9):
    '''
    Calculate the experimental error rate based on the investment to equipment. Based on inverse Michaelis-Menten equation mit max error 0.9 and min error 0.1.
    '''
    KM = relKM*ResTotal
    myMax = 1
    return myMax - (Vmax * Invest) / (KM + Invest)
