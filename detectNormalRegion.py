import traceback
delete_set = set([])


def expandNormalRegion(sigma, epsilon,  index_name, i_index):
    global delete_set
    delete_set = set([])
    start_punkt  = index_name[sigma]
    start_punkt.epsilon_neigh = set(start_punkt.neigh)
    collect_neighbor = set([])
    for nachbar in start_punkt.epsilon_neigh:
       try:
         try_set = expansionCOND(start_punkt, nachbar, epsilon, index_name, i_index)
         if try_set != None:
            collect_neighbor.update(try_set)
       except:
            pass
            print(traceback.print_exc())
    chainCluster(start_punkt,collect_neighbor,epsilon, index_name,i_index)


def chainCluster(sigma, collect_neighbor, epsilon, index_name, i_index):
    sigma.epsilon_neigh.update(collect_neighbor)
    global delete_set
    sigma.epsilon_neigh = sigma.epsilon_neigh.difference(delete_set)
    delete_set = set([])
    new_collect_neighbor = set([])
    for nachbar in collect_neighbor:
        try:
         new_collect_neighbor.update(expansionCOND(sigma, nachbar, epsilon, index_name, i_index))
        except:
            pass
            #print(traceback.print_exc())
    if (len(new_collect_neighbor)>0):
        chainCluster(sigma, new_collect_neighbor, epsilon, index_name, i_index)






def expansionCOND(start_punkt, nachbar_punkt, epsilon,  index_name, i_index):
  try:
    nachbar_punkt = index_name[i_index[nachbar_punkt]]
    nachbar_punkt.epsilonCalcNeigh(epsilon)
    for nachbar in nachbar_punkt.epsilon_neigh:
     index_nachbar = nachbar
     nachbar =   index_name[i_index[nachbar]]
     nachbar.epsilonCalcNeigh(epsilon)
     schnitt_list = [_ for _ in nachbar.epsilon_neigh if _ not in start_punkt.epsilon_neigh]
     abfall = len(nachbar.epsilon_neigh) - len(schnitt_list) < (1) * len(schnitt_list)
     # abfall = len(neigh.epsilon_neigh)<(rho)*len(schnitt_list)
     # abfall = rho <len(schnitt_list) / ( len(neigh.epsilon_neigh))
   #  abfall = rho >= len(schnitt_list) / ( len(neigh.epsilon_neigh))



     if( (abfall )):
         global delete_set
         delete_set.add(index_nachbar)
         return set([])
     else:
         return set(schnitt_list)
  except:
      pass

def start_punkt_bestimmen():
    pass



