from optparse import Values
import scipy.io as sio
import networkx as nx
import math
import matplotlib.pyplot as plt

matFile = sio.loadmat('Homo_sapiens.mat')
#matFile = nx.Graph()
#print(list(nx.connected_components(matFile)))
A = matFile["network"]
G = nx.from_scipy_sparse_matrix(A)

dc = nx.degree_centrality(G)
cc = nx.closeness_centrality(G)
bc = nx.betweenness_centrality(G)
hc = nx.harmonic_centrality(G)
ec = nx.eigenvector_centrality(G)
clust= nx.clustering(G)
pr = nx.pagerank(G)

with open('NodeClass.txt','w') as f:
    for x in list(G.nodes):
        f.write("Node " + f"{x}" + ": ")
        f.write(f"{dc[x],cc[x],bc[x],hc[x],ec[x],clust[x],pr[x]}")
        #f.write(f"{dc[x]}")
        f.write('\n')
        


dcMax = max(dc.values())
dcMin = min(dc.values())
print(dcMax)
print(dcMin)
range = dcMax - dcMin
n = len(G.nodes)
width = range/(math.sqrt(n))
print(width)



plt.hist(pr.values(), bins = 62) #did plt.hist(dc.values(),...) and for the rest of the categories
plt.show()




#G = nx.read_edgelist('Homo_sapiens-network.txt')

for G in benign_GRAPH_file:
    with file (out, 'r') as f:
        write('DC: ' + nx.degree_centrality(G))
        write('CC:' + nx.closeness_centrality(G) )



        '/n'
        







