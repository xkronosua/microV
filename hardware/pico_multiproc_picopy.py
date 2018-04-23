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
import SharedArray

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




print(SharedArray.list())
def create_pico_reader(config,shared_data_name,q=Queue()):
	ps = set_picoscope(config)
	i=0
	buf = SharedArray.attach(shared_data_name)
	while True:
		print('start')
		try:

			r = ps.capture_prep_block(return_scaled_array=1)
			dataA = r[0]['A']
			dataB = r[0]['B']
			scanA = abs(dataA.max(axis=1) - dataA.min(axis=1))
			scanB = abs(dataB.max(axis=1) - dataB.min(axis=1))
			scanT = r[1]
			push_data_to_fixBuff(buf,np.array([scanT,scanA,scanB]).T)[:]
			print("s",buf.sum())
			if q.qsize()>0:
				status = q.get()
				if status == 'kill':
					print(status)
					break
			#time.sleep(1)
			i +=1
		except:
			traceback.print_exc()
			break
	del ps
	del buf


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
		self.curveA = pw.plot([0,1])
		self.curveA.setPen((255,0,0))

		self.curveB = pw.plot([0,1])
		self.curveB.setPen((0,255,0))

		## Create an empty plot curve to be filled later, set its pen
		self.curveA1 = pw1.plot()
		self.curveA1.setPen((255,0,0))

		self.curveB1 = pw1.plot()
		self.curveB1.setPen((0,255,0))


		self.timer.timeout.connect(self.update)
		self.timer.start(1000)
		#self.ps = picopy.Pico3k()


		config = {	'ChA_VRange':'500mV','ChA_Offset':0,
					'ChB_VRange':'500mV','ChB_Offset':0,
					'sampleInterval':0.0001,'samplingDuration':0.003,
					'pico_pretrig':0.001,'n_captures':100,'trigSrc':'ext',
					'threshold_V':-0.350,'direction':'RISING'}
		'''
		config = {	'ChA_VRange':'20mV','ChA_Offset':0,
					'ChB_VRange':'20mV','ChB_Offset':0,
					'sampleInterval':2e-9,'samplingDuration':15e-9,
					'pico_pretrig':0.000,'n_captures':500,'trigSrc':'ext',
					'threshold_V':0.02,'direction':'RISING'}
		'''
		self.sa_name = 'sha1'
		try:
			pico_shared_buf = SharedArray.create(self.sa_name,(1000,3))
		except:
			traceback.print_exc()
			pico_shared_buf = SharedArray.attach(self.sa_name)
			pico_shared_buf[:] = 0


		self.p = multiprocessing.Process(target=create_pico_reader,args=[config,self.sa_name,self.q])

		self.p.start()
		#self.p.join()
		#self.actionExit.toggled.connect(self.closeEvent)


	def update(self):
		data = SharedArray.attach(self.sa_name)
		print(":",data.sum())
		w = data[:,0]==0

		self.curveA1.setData(x=data[~w,0],y=data[~w,1])
		self.curveB1.setData(x=data[~w,0],y=data[~w,2])
		app.processEvents()
		del data


	def closeEvent(self, evnt=None):
		print('closeEvent')
		self.q.put('kill')
		self.p.join()
		#self.p.terminate()




## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	__spec__ = None #"ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

	app = QtGui.QApplication(sys.argv)
	ex = Pico_view()

	app.exec_()
