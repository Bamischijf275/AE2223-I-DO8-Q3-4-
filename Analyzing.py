def getFibreCount(labels):
    return np.max(labels)


def getFibreFraction(labels):
    BinLabels = (labels > 0).astype(int)
    fibres = np.sum(BinLabels)
    volume = np.size(labels)
    return fibres/volume

labels = np.genfromtxt("labels.csv", dtype=float,
                     encoding=None, delimiter=",")
