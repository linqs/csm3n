#!/usr/local/bin


labels = {'Case_Based':1,'Genetic_Algorithms':2,'Neural_Networks':3,'Probabilistic_Methods':4,'Reinforcement_Learning':5,'Rule_Learning':6,'Theory':7}

ids = {}
feat = []
labs = []
f = open('cora.content','r')
line = f.readline()
i = 0
while line != '':
	row = line.strip().split('\t')
	i = i + 1
	ids[row[0]] = i
	feat.append(row[1:-1])
	y = row[-1]
	labs.append(labels[y])
	line = f.readline()
f.close()
print 'Num of nodes: {0}'.format(len(ids))

edge = []
f = open('cora.cites','r')
line = f.readline()
while line != '':
	row = line.strip().split('\t')
	id1 = row[0]
	id2 = row[1]
	if (id1 in ids) and (id2 in ids):
		n1 = ids[id1]
		n2 = ids[id2]
		edge.append([n1,n2])
# 	elif id1 not in ids:
# 		print id1
# 	elif id2 not in ids:
# 		print id2
	line = f.readline()
f.close()
print 'Num of edges: {0}'.format(len(edge))

f = open('cora.node','w')
for i in range(len(ids)):
	f.write('{0},{1},{2}\n'.format(i,labs[i],','.join([str(x) for x in feat[i]])))
f.close()

f = open('cora.edge','w')
for e in range(len(edge)):
	f.write('{0},{1}\n'.format(edge[e][0],edge[e][1]))
f.close()

