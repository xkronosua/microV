#from pylab import *
import multiprocessing
import picopy
from multiprocessing import Queue, Array, Event
import numpy as np
import time

config = {	'ChA_VRange':'500mV','ChA_Offset':0,
			'ChB_VRange':'500mV','ChB_Offset':0,
			'sampleInterval':2e-6,'samplingDuration':0.4,
			'pico_pretrig':0.0004,'n_captures':1,'trigSrc':'ext',
			'threshold_V':-0.320,'direction':'RISING','pulseFreq':1624.}
q = multiprocessing.Queue()

def push_data_to_fixBuff(buf, data):
	push_len = len(data)
	assert len(buf) >= push_len
	buf[:-push_len] = buf[push_len:]
	buf[-push_len:] = data
	return buf

def ndp2p(shared_data,r,N):
	print('ndp2p')
	dataA = r[0]['A']
	dataB = r[0]['B']
	print(dataA,dataB.shape,N)
	try:
		b = np.array(np.split( dataB[:,:int(dataB.shape[1]//N*N)],N,axis=1)).mean(axis=1)
	except:
		traceback.print_exc()
	print(2)

	scanB = abs(b.max(axis=1) - b.min(axis=1))
	a = np.array(np.split( dataA[:,:int(dataA.shape[1]//N*N)],N,axis=1)).mean(axis=1)
	scanA = abs(a.max(axis=1) - a.min(axis=1))
	scanT = r[2][:int(dataB.shape[1]//N*N)]
	scanT = np.linspace(scanT.min(),scanT.max(),len(scanA))
	print(scanA)
	buf = np.frombuffer(shared_data['data'].get_obj(), dtype='d').reshape(shared_data['shape'])
	push_data_to_fixBuff(buf,np.array([scanT,scanA,scanB]).T)[:]
	print("s",buf[:,1:].sum())
	#q.put([scanA,scanB,scanT])

def search_time_range(shared_data, search_q=multiprocessing.Queue(), out_q=multiprocessing.Queue()):
	print('search')
	timeout = 10

	buf = np.frombuffer(shared_data['data'].get_obj(), dtype='d').reshape(shared_data['shape'])
	last_time = time.time()
	while True:
		if time.time()-last_time>timeout:
			break
		if search_q.qsize()>0:
			last_time = time.time()
			req = search_q.get()
			if req == 'kill':
				print('search_kill')
				break
			start, end, blocks = req
			w = (buf[:,0]>start) & (buf[:,0]<end)
			for i in range(100):
				if sum(w)==0:
					print('noData')
					continue
			dataA = np.array([np.mean(i) for i in np.array_split(buf[:,1][w],blocks)])
			dataB = np.array([np.mean(i) for i in np.array_split(buf[:,2][w],blocks)])
			t = np.linspace(start,end,len(dataA))
			out_q.put([t,dataA,dataB])
			print(t,dataA,dataB)
		time.sleep(0.01)

def nonstop_capture(config,shared_data,runEvent,q):
	ps = picopy.Pico3k()
	ps.set_config(config)
	N = int(config['samplingDuration']*config['pulseFreq'])
	r = ps.capture_prep_block(return_scaled_array=1)
	q.put('ready')
	while True:
		runEvent.wait()
		runEvent.clear()
		print('run')
		r = ps.capture_prep_block(return_scaled_array=1)
		p = multiprocessing.Process(target=ndp2p,args=[shared_data,r,N])
		p.daemon = True
		p.start()
		if q.qsize()>0:
			status = q.get()
			if status=='kill':
				print(status)
				break
	del ps

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys,os
	__spec__ = None #"ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
	sys.path.append('../hardware/')
	print(os.path.realpath(__file__))

	sa_shape = (100000,3)
	unshared_arr = np.zeros(sa_shape[0]*sa_shape[1])
	sa = Array('d', int(np.prod(sa_shape)))
	sa = {'data':sa, 'shape':sa_shape}
	q = Queue()
	runEvent = Event()
	p = multiprocessing.Process(target=nonstop_capture,args=[config,sa,runEvent,q])
	#p.daemon = True
	p.start()
	while q.qsize()==0:
		time.sleep(0.1)
	search_q = Queue()
	out_q = Queue()
	search_p = multiprocessing.Process(target=search_time_range,args=[sa,search_q,out_q])
	search_p.daemon = True
	search_p.start()
	from sim.E727 import *

	piStage = E727()
	print(piStage.ConnectUSBWithBaudRate())
	print(piStage.qSAI())
	print(piStage.SVO())
	print(piStage.qPOS(b'1'))
	print(piStage.VEL([10,100,100],b'1 2 3'))
	#print(piStage.ATZ())
	x_range=np.arange(0,101,50)
	y_range=np.arange(0,100,1)
	z_range=np.arange(30,31,1)
	piStage.MOV(x_range.min(),axis=1, waitUntilReady=True)
	piStage.MOV(y_range.min(),axis=2, waitUntilReady=True)
	piStage.MOV(z_range.min(),axis=3, waitUntilReady=True)
	time_table = np.hstack((time.time(),piStage.qPOS()))
	Z = []
	Y = []
	X = []
	status = q.get()
	print(status)
	runEvent.set()
	pos0 = piStage.qPOS()
	t0 = time.time()
	time_table = np.vstack((time_table, np.hstack((t0,pos0))))
	piStage.MOV(100,axis=1, waitUntilReady=True)

	time.sleep(0.5)
	pos1 = piStage.qPOS()
	t1 = time.time()
	time_table = np.vstack((time_table, np.hstack((t1,pos1))))
	search_q.put([t0,t1,1000])

	runEvent.set()
	q.put('kill')
	search_q.put('kill')
	#p.join()
	#search_p.join()
	out = []
	#time.sleep(10)

	while out_q.qsize()>0:
		out.append(out_q.get())


	'''
	try:
		print('move')
		for z in z_range:
			#pos = piStage.qPOS()
			#t.append([time.time(),pos)
			piStage.MOV(z,axis=3, waitUntilReady=True)
			#pos = piStage.qPOS()
			#t.append([time.time(),pos)
			#search_q.put([t[-2][0],t[-1][0],pos,1])
			#Z.
			forw = False
			for y in y_range:
				#t.append( [time.time(),[piStage.qPOS()]])
				piStage.MOV(y,axis=2, waitUntilReady=True)
				#t.append([time.time(),[piStage.qPOS()]])
				#search_q.put([t[-2][0],t[-1][0],t[-1][1],1])
				if forw:
					x_range_ = x_range
				else:
					x_range_ = x_range[::-1]
				for x in x_range_:
					runEvent.set()
					pos0 = piStage.qPOS()
					t0 = time.time()
					time_table = np.vstack((time_table, np.hstack((t0,pos0))))
					piStage.MOV(x,axis=1, waitUntilReady=True)
					time.sleep(0.08)
					pos1 = piStage.qPOS()
					t1 = time.time()
					time_table = np.vstack((time_table, np.hstack((t1,pos1))))
					search_q.put([t0,t1,50])
				forw = not forw

		out = []
		time.sleep(10)

		while out_q.qsize()>0:
			out.append(out_q.get())
	except:
		traceback.print_exc()
		search_q.put('kill')
		runEvent.set()
		q.put('kill')
		piStage.CloseConnection()
	search_q.put('kill')
	runEvent.set()
	q.put('kill')
	p.join()
	piStage.CloseConnection()
	from scipy.interpolate import interp1d

	x_i=interp1d(time_table[:,0],time_table[:,1])
	y_i=interp1d(time_table[:,0],time_table[:,2])
	z_i=interp1d(time_table[:,0],time_table[:,3])

	from pylab import *

	data = np.hstack(out).T
	X = x_i(data[:,0])
	Y = y_i(data[:,0])
	Z = z_i(data[:,0])
	XX,YY=np.meshgrid(X,Y)
	from scipy.interpolate import griddata
	#da = griddata((X, Y), data[:,1], (XX, YY),method='nearest')
	db = griddata((X, Y), data[:,2], (XX, YY),method='nearest')

	contourf(XX,YY,db)
	show()
	'''
