def getFibreCount(mask):
    return np.max(mask)


def getFibreFraction(mask):
    BinLabels = (mask > 0).astype(int)
    fibres = np.sum(BinLabels)
    volume = np.size(mask)
    return fibres/volume


def getOverlap(mask1, mask2):
    for i in range(1, np.max(mask1)):
        print(test)



labels = np.genfromtxt("mask.csv", dtype=float,
                     encoding=None, delimiter=",")
