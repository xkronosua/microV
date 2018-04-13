from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import time
import sys

from picoscope import ps3000a
import multiprocessing
from multiprocessing import Queue
import time
import traceback
import picopy
import sharedmem

def set_picoscope(config):
	ps = picopy.Pico3k()
	c = config
	ps.setChannel("A", coupling="DC", VRange=c['ChA_VRange'], VOffset=c['ChA_Offset'])
	ps.setChannel("B", coupling="DC", VRange=c['ChB_VRange'], VOffset=c['ChB_Offset'])

	ps.setSamplingInterval(c['sampleInterval'],c['samplingDuration'],pre_trigger = c['pico_pretrig'],
		number_of_frames=c['n_captures'], downsample=1, downsample_mode='NONE')

	ps.setSimpleTrigger(trigSrc=c['trigSrc'], threshold_V=c['threshold_V'],
			direction=c['direction'], timeout_ms=5, enabled=True)
	return ps

def push_data_to_fixBuff(buf, data):
	push_len = len(data)
	assert len(buf) >= push_len
	buf[:-push_len] = buf[push_len:]
	buf[-push_len:] = data
	return buf

def create_pico_reader(config,shared_data):
	ps = set_picoscope(config)
	i=0
	while True:
		print('start')
		try:
			#print(i)
			r = ps.capture_prep_block(return_scaled_array=1)
			dataA = r[0]['A']
			dataB = r[0]['B']
			scanA = abs(dataA.max(axis=1) - dataA.min(axis=1))
			scanB = abs(dataB.max(axis=1) - dataB.min(axis=1))
			scanT = r[1]
			push_data_to_fixBuff(shared_data,np.array([scanT,scanA,scanB]).T)
			#print(scanA)
			i +=1
		except:
			traceback.print_exc()
			break
	del ps

pico_shared_buf = sharedmem.full_like(np.zeros((500,3)),value=-1,dtype=np.float64)


config = {	'ChA_VRange':'500mV','ChA_Offset':0,
			'ChB_VRange':'500mV','ChB_Offset':0,
			'sampleInterval':0.0001,'samplingDuration':0.003,
			'pico_pretrig':0.001,'n_captures':10,'trigSrc':'ext',
			'threshold_V':-0.350,'direction':'RISING'}


def reader(q):
	pico = picopy.Pico3k()

	n_captures = 10
	pico.setChannel("A", coupling="DC", VRange='500mV')
	pico.setChannel("B", coupling="DC", VRange='500mV')
	(sampleInterval, noSamples, maxSamples) = pico.setSamplingInterval(0.00001,0.003,pre_trigger=0.001,
		number_of_frames=n_captures, downsample=1, downsample_mode='NONE',)

	#trigger = picopy.EdgeTrigger(channel='B', threshold=-0.35, direction='FALLING')
	#pico.set_trigger(trigger)
	pico.setSimpleTrigger(trigSrc="B", threshold_V=-0.350, direction='FALLING',
							 timeout_ms=10, enabled=True,delay=0)

	for i in range(50):

		r = pico.capture_prep_block( return_scaled_array=1)
		q.put(r)
	q.put('done')
	pico.close()

class Pico_view(QtGui.QMainWindow):
	timer = QtCore.QTimer()
	q = Queue()
	liveA = np.array([])
	liveB = np.array([])
	liveT = np.array([])
	def __init__(self, parent=None):
		QtGui.QMainWindow.__init__(self, parent)

		#QtGui.QApplication.setGraphicsSystem('raster')
		#app = QtGui.QApplication([])
		self.setWindowTitle('pyqtgraph example: PlotWidget')
		self.resize(800,800)
		cw = QtGui.QWidget()
		self.setCentralWidget(cw)
		l = QtGui.QVBoxLayout()
		cw.setLayout(l)

		pw = pg.PlotWidget(name='Plot1')  ## giving the plots names allows us to link their axes together
		l.addWidget(pw)
		pw1 = pg.PlotWidget(name='Plot2')  ## giving the plots names allows us to link their axes together
		l.addWidget(pw1)

		self.show()

		## Create an empty plot curve to be filled later, set its pen
		self.curveA = pw.plot()
		self.curveA.setPen((255,0,0))

		self.curveB = pw.plot()
		self.curveB.setPen((0,255,0))

		## Create an empty plot curve to be filled later, set its pen
		self.curveA1 = pw1.plot()
		self.curveA1.setPen((255,0,0))

		self.curveB1 = pw1.plot()
		self.curveB1.setPen((0,255,0))


		self.timer.timeout.connect(self.update)
		self.timer.start(0.5)
		#self.ps = picopy.Pico3k()
		self.p = multiprocessing.Process(target=create_pico_reader,args=[config,pico_shared_buf])
		self.p.start()
		#self.p.join()
		#self.actionExit.toggled.connect(self.closeEvent)


	def update(self):
		#print(data)
		data = pico_shared_buf

		self.curveA1.setData(x=data[:,0],y=data[:,1])
		self.curveB1.setData(x=data[:,0],y=data[:,2])
		app.processEvents()


	def closeEvent(self, evnt=None):
		print('closeEvent')
		self.p.terminate()

		#self.pico.close()
		#self.pico.alive=False
		#ex.pico.terminate()
		#data = np.array([self.liveT,self.liveA,self.liveB]).T
		#np.savetxt('signal'+str(time.time())+'.txt',data)




## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

	app = QtGui.QApplication(sys.argv)
	ex = Pico_view()

	app.exec_()
