from pylab import *
import numpy as np

import time
import sys
from E727 import *
from picoscope import ps3000a
from multiprocessing import Process

get_ipython().run_line_magic('matplotlib', 'qt')

ps = ps3000a.PS3000a(connect=False)
ps.open()
n_captures = 2000
ps.setChannel("A", coupling="DC", VRange=0.1)
ps.setChannel("B", coupling="DC", VRange=0.1)
ps.setSamplingInterval(200e-9,15e-6)
ps.setSimpleTrigger(trigSrc="External", threshold_V=0.020, direction='Rising',
						 timeout_ms=5, enabled=True)
samples_per_segment = ps.memorySegments(n_captures)
ps.setNoOfCaptures(n_captures)

piStage = E727()
print(piStage.ConnectUSBWithBaudRate())
print(piStage.qSAI())
print(piStage.SVO())


x_range = np.arange(0,100,1)
y_range = np.arange(0,100,1)
z_range = np.arange(40,41,1)

direction = True
piStage.MOV(x_range.min(),axis=1, waitUntilReady=True)
piStage.MOV(y_range.min(),axis=2, waitUntilReady=True)
piStage.MOV(z_range.min(),axis=3, waitUntilReady=True)
data_out = []

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
				piStage.MOV(x_range.max(),axis=1, waitUntilReady=0)
			else:
				piStage.MOV(x_range.min(),axis=1, waitUntilReady=0)
			t3 = time.time()
			ps.waitReady()

			#print("Time to get sweep: " + str(t2 - t1))
			ps.getDataRawBulk(channel='A',data=dataA)
			ps.getDataRawBulk(channel='B',data=dataB)
			t4 = time.time()
			tmp_data = dataA.copy()
			#print("Time to read data: " + str(t3 - t2))
			dataA1=dataA[:, 0:ps.noSamples]
			dataB1=dataB[:, 0:ps.noSamples]

			scanA = abs(dataA1.max(axis=1) - dataA1.min(axis=1))
			scanB = abs(dataB1.max(axis=1) - dataB1.min(axis=1))
			scanA = np.mean(scanA.reshape(-1, 20), 1)
			scanB = np.mean(scanB.reshape(-1, 20), 1)
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
		data_out.append([dataxyA,dataxyB])
	except KeyboardInterrupt:
		break
		ps.close()
		piStage.CloseConnection()




print(piStage.CloseConnection())
ps.close()

#plot(dataA)
#plot(dataB)
#show(0)
