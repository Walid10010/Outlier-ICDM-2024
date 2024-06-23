
class D2Point():


    def __init__(self, corr_, neigh_, dist_, kdist_, name, kComplete):
        self.corr = corr_
        self.neigh= neigh_
        self.knn_dist = dist_
        self.epsilon_neigh = []
        self.k_distance = kdist_
        self.name = name
        self.kComplete = kComplete
    def epsilonCalcNeigh(self, epsilon):
        index = 0
        for dist in self.knn_dist:
            if(dist <= epsilon):
                index += 1
            else:
                self.epsilon_neigh = self.neigh[:index]




