


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import time
import sys
import traceback

import picopy
import picopy.pico_status as status

import copy
import math



mode = 'live'
if len(sys.argv)>1:
	if sys.argv[1] == 'scope':
		mode = 'scope'

app = QtGui.QApplication([])

p = pg.plot()
p.setWindowTitle('pyqtgraph example: PlotSpeedTest')

#p.setRange(QtCore.QRectF(0, -10, 5000, 20))
p.setLabel('bottom', 'Index', units='B')
curveA = p.plot(pen=(255,0,0))
curveB = p.plot(pen=(0,255,0))
curves=[]
n_captures = 100
for i in range(n_captures):
	curves.append(p.plot())

#curve.setFillBrush((0, 0, 100, 100))
#curve.setFillLevel(0)

#lr = pg.LinearRegionItem([100, 4900])
#p.addItem(lr)
ps = picopy.Pico3k()

n_captures = 50
ps.setChannel("A", coupling="DC", VRange='500mV')
ps.setChannel("B", coupling="DC", VRange='500mV')
(sampleInterval, noSamples, maxSamples) = ps.setSamplingInterval(0.000001,0.0035)

#trigger = picopy.EdgeTrigger(channel='B', threshold=-0.35, direction='FALLING')
#ps.set_trigger(trigger)
ps.setSimpleTrigger(trigSrc="B", threshold_V=-0.350, direction='FALLING',
						 timeout_ms=10, enabled=True,delay=300)



dataA = []
dataB = []
liveA = np.array([0])
liveB = np.array([0])

def update():
	try:
		global dataA, liveA,dataB, liveB, tmp_data, n_captures
		t0 = time.time()
		r = ps.capture_prep_block( number_of_frames=n_captures, downsample=2, downsample_mode='NONE',
			return_scaled_array=0)
		dataA1 = r[0]['A']
		dataB1 = r[0]['B']
		scanA=abs(dataA1.max(axis=1)-dataA1.min(axis=1))
		scanB=abs(dataB1.max(axis=1)-dataB1.min(axis=1))

		if mode == 'scope':
			curveA.setData(dataA1.mean(axis=0))
			curveB.setData(dataB1.mean(axis=0))
		elif mode == "live":
			liveA = np.hstack((liveA,scanA))
			curveA.setData(liveA)
			#liveB+=scanB.tolist()
			liveB = np.hstack((liveB,scanB))
			curveB.setData(liveB)
		#for i in range(len(data)):
		#	curves[i].setData(data[i,:])
		print("Time of cycle: " + str(time.time()-t0), r[-1])
		app.processEvents()  ## force complete redraw for every plot
		#time.sleep(1)
	except:
		traceback.print_exc()
		ps.close()
		timer.stop()
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)




## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	try:
		if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
			QtGui.QApplication.instance().exec_()
	except:
		ps.close()
	ps.close()
