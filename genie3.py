import sys, numpy
sys.path.append('/home/shruti/Documents/Research/genie3/GENIE3/GENIE3_python/')
print sys.path
from GENIE3 import *

data = numpy.loadtxt('network_unsigned_knockdowns.tsv', skiprows=1)
f = open('network_unsigned_knockdowns.tsv')
gene_names = f.readline().rstrip('\n').split('\t')
f.close()
for i in range(100, 790, 100):
    print '-----------------------------------------'
    sys.stdout.flush()
    print i
    sys.stdout.flush()
    print '-----------------------------------------'
    sys.stdout.flush()
    i = min(i, 790)
    x = data[:i, :]
    print x.shape
    net = GENIE3(x)
    get_link_list(net, gene_names=gene_names, file_name='ranking_knockdowns_' + str(i) + '.txt')
    print
    sys.stdout.flush()
