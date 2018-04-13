#!/usr/bin/env python
from multiprocessing import Process
import sharedmem
import numpy

def do_work(data, start):
    data[start] = 0;

def split_work(num):
    n = 20
    width  = n//num
    shared = sharedmem.empty(n)
    shared[:] = numpy.random.rand(1, n)[0]
    print ("values are %s" % shared)

    processes = [Process(target=do_work, args=(shared, i*width)) for i in range(num)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print ("values are %s" % shared)
    print ("type is %s" % type(shared[0]))

if __name__ == '__main__':
    split_work(2)
