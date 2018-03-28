from pylab import *
import numpy as np

import time
import sys
from scipy.signal import resample
from picoscope import ps3000a
from multiprocessing import Process

get_ipython().run_line_magic('matplotlib', 'qt')



def fastScan(piStage,ps,x_range=np.arange(0,100,1),y_range=np.arange(0,100,1),z_range=np.arange(0,100,1),q=None):
	n_captures = 710
	direction = True
	piStage.MOV(x_range.min(),axis=1, waitUntilReady=True)
	piStage.MOV(y_range.min(),axis=2, waitUntilReady=True)
	piStage.MOV(z_range.min(),axis=3, waitUntilReady=True)

	t0=time.time()
	piStage.MOV(x_range.max(),axis=1, waitUntilReady=True)
	t1=time.time()
	piStage.MOV(x_range.min(),axis=1, waitUntilReady=True)
	t2=time.time()
	dt = mean([t2-t1,t1-t0])
	k = dt/0.08
	n_captures = int(n_captures*k)
	samples_per_segment = ps.memorySegments(n_captures)
	ps.setNoOfCaptures(n_captures)

	data_out = []
	start0 = time.time()
	for z in z_range:
		try:
			piStage.MOV(z,axis=3, waitUntilReady=True)
			print(z)
			t_list = []
			dataxyA = []
			dataxyB = []
			t_start = time.time()
			dataA = np.zeros((n_captures, samples_per_segment), dtype=np.int16)
			dataB = np.zeros((n_captures, samples_per_segment), dtype=np.int16)
			tmp_data = np.zeros((n_captures, samples_per_segment), dtype=np.int16)

			for y in y_range:
				t1 = time.time()
				piStage.MOV(y,axis=2, waitUntilReady=True)
				ps.runBlock()
				t2 = time.time()
				if direction:
					piStage.MOV(x_range.max(),axis=1, waitUntilReady=True)
				else:
					piStage.MOV(x_range.min(),axis=1, waitUntilReady=True)
				t3 = time.time()
				ps.waitReady()
				#print("Time to get sweep: " + str(t2 - t1))
				ps.getDataRawBulk(channel='A',data=dataA)
				ps.getDataRawBulk(channel='B',data=dataB)
				t4 = time.time()
				#tmp_data = dataA.copy()
				#print("Time to read data: " + str(t3 - t2))
				dataA1=dataA[:, 0:ps.noSamples]
				dataB1=dataB[:, 0:ps.noSamples]

				scanA = abs(dataA1.max(axis=1) - dataA1.min(axis=1))
				scanB = abs(dataB1.max(axis=1) - dataB1.min(axis=1))

				scanA = resample(scanA,len(x_range))
				scanB = resample(scanB,len(x_range))
				if direction:
					dataxyA.append(scanA)
					dataxyB.append(scanB)
					direction = False
				else:
					dataxyA.append(scanA[::-1])
					dataxyB.append(scanB[::-1])
					direction = True
				#t_list.append(time.time()-t_start)
				t5 = time.time()
				print('Y',y,time.time()-t5,t5-t4,t4-t3,t3-t2,t2-t1)
			print("xyScan:", time.time()-t_start)
			t_start = time.time()
			data_out.append([np.array(dataxyA).T[::-1][::-1],np.array(dataxyB).T[::-1][::-1]])
		except KeyboardInterrupt:
			break
			ps.close()
			piStage.CloseConnection()
	dataA_out=[]
	dataB_out=[]
	for d in data_out:
		dataA_out.append(d[0])
		dataB_out.append(d[1])
	a = np.array(dataA_out,dtype=np.int16)
	b = np.array(dataB_out,dtype=np.int16)
	if not q is None:
		q.put((a,b))
	return a,b


if __name__=='__main__':
	from skimage.external.tifffile import imsave
	from E727 import *
	import traceback
	ps = ps3000a.PS3000a(connect=False)
	ps.open()

	ps.setChannel("A", coupling="DC", VRange=0.1)
	ps.setChannel("B", coupling="DC", VRange=0.1)
	ps.setSamplingInterval(200e-9,15e-6)
	ps.setSimpleTrigger(trigSrc="External", threshold_V=0.020, direction='Rising',
							 timeout_ms=5, enabled=True)


	#dataA = np.zeros((n_captures, samples_per_segment), dtype=np.int16)
	#dataB = np.zeros((n_captures, samples_per_segment), dtype=np.int16)
	#t1 = time.time()
	#ps.runBlock()
	#ps.waitReady()
	#t4 = time.time()
	#ps.getDataRawBulk(channel='A',data=dataA)
	#ps.getDataRawBulk(channel='B',data=dataB)

	#print('pico_time',t4-t1)

	piStage = E727()
	print(piStage.ConnectUSBWithBaudRate())
	print(piStage.qSAI())
	print(piStage.SVO())

	start0=time.time()
	a=[]
	b=[]
	try:
		a,b=fastScan(piStage,ps,x_range=np.arange(20,80,1),y_range=np.arange(0,50,1),z_range= np.arange(40,70,3))
	except:
		traceback.print_exc()
		print(piStage.CloseConnection())
		ps.close()
	print('TotalTime:',time.time()-start0)
	from pylab import *
	print(piStage.CloseConnection())
	ps.close()
	imsave('dataA.tif',a)
	imsave('dataB.tif',a)

	matshow(a[0])
	show(0)


#plot(dataA)
#plot(dataB)
#show(0)
