from multiprocessing import Process, Array
import scipy
import time

def f(a):
	a[0] = -a[0]

if __name__ == '__main__':
	__spec__ = None
	# Create the array
	N = int(10)
	unshared_arr = scipy.rand(N)
	arr = Array('d', unshared_arr)
	print ("Originally, the first two elements of arr = %s"%(arr[:2]))

	# Create, start, and finish the child processes

	p = Process(target=f, args=(arr,))
	p.daemon = True

	p.start()
	#p.join()
	time.sleep(1)
	# Printing out the changed values
	print ("Now, the first two elements of arr = %s"%arr[:2])
