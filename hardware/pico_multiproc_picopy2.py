from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import time
import sys

from picoscope import ps3000a
import multiprocessing
from multiprocessing import Queue, Array
import time
import traceback
import picopy

#import SharedArray

def set_picoscope(config):
	ps = picopy.Pico3k()
	ps.set_config(config)
	'''
	c = config
	ps.setChannel("A", coupling="DC", VRange=c['ChA_VRange'], VOffset=c['ChA_Offset'])
	ps.setChannel("B", coupling="DC", VRange=c['ChB_VRange'], VOffset=c['ChB_Offset'])

	ps.setSamplingInterval(c['sampleInterval'],c['samplingDuration'],pre_trigger = c['pico_pretrig'],
		number_of_frames=c['n_captures'], downsample=1, downsample_mode='NONE')

	ps.setSimpleTrigger(trigSrc=c['trigSrc'], threshold_V=c['threshold_V'],
			direction=c['direction'], timeout_ms=5, enabled=True)
	'''
	return ps

def push_data_to_fixBuff(buf, data):
	push_len = len(data)
	assert len(buf) >= push_len
	buf[:-push_len] = buf[push_len:]
	buf[-push_len:] = data
	return buf

def search_time_range(shared_data,search_q=Queue(), out_q=Queue()):
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
				break
			start, end, blocks = req
			w = (buf[:,0]>start) & (buf[:,0]<end)
			dataA = np.array([np.mean(i) for i in np.array_split(buf[:,1][w],blocks)])
			dataB = np.array([np.mean(i) for i in np.array_split(buf[:,2][w],blocks)])
			t = np.linspace(start,end,len(dataA))
			out_q.put([t,dataA,dataB])
			print(t,dataA,dataB)
		time.sleep(0.01)




#print(SharedArray.list())
def create_pico_reader(config,shared_data,q=Queue()):
	ps = set_picoscope(config)
	i=0
	buf = np.frombuffer(shared_data['data'].get_obj(), dtype='d').reshape(shared_data['shape'])
	q.put('ready')
	while True:
		#print('start')
		try:

			r = ps.capture_prep_block(return_scaled_array=1)
			dataA = r[0]['A']
			dataB = r[0]['B']
			scanA = abs(dataA.max(axis=1) - dataA.min(axis=1))
			scanB = abs(dataB.max(axis=1) - dataB.min(axis=1))
			scanT = r[1]
			push_data_to_fixBuff(buf,np.array([scanT,scanA,scanB]).T)[:]
			#print("s",buf[:,1:].sum())


			if q.qsize()>0:
				status = q.get()
				if status == 'pause':
					while status == 'pause':
						time.sleep(0.1)
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
	#del buf


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
		self.timer.start(10)
		#self.ps = picopy.Pico3k()

		'''
		config = {	'ChA_VRange':'500mV','ChA_Offset':0,
					'ChB_VRange':'500mV','ChB_Offset':0,
					'sampleInterval':0.0001,'samplingDuration':0.003,
					'pico_pretrig':0.001,'n_captures':10,'trigSrc':'ext',
					'threshold_V':-0.350,'direction':'RISING'}
		'''
		config = {	'ChA_VRange':'20mV','ChA_Offset':0,
					'ChB_VRange':'20mV','ChB_Offset':0,
					'sampleInterval':2e-9,'samplingDuration':15e-9,
					'pico_pretrig':0.000,'n_captures':5000,'trigSrc':'ext',
					'threshold_V':0.02,'direction':'RISING'}
		#'''
		self.sa_shape = (10000,3)
		unshared_arr = np.zeros(self.sa_shape[0]*self.sa_shape[1])
		sa = Array('d', int(np.prod(self.sa_shape)))
		self.sa = {'data':sa, 'shape':self.sa_shape}
		self.p = multiprocessing.Process(target=create_pico_reader,args=[config,self.sa,self.q])
		self.p.daemon = True
		self.p.start()
		while self.q.qsize()==0:
			time.sleep(0.1)
		self.search_q = Queue()
		self.out_q = Queue()
		search_p = multiprocessing.Process(target=search_time_range,args=[self.sa,self.search_q,self.out_q])
		search_p.daemon = True
		search_p.start()
		self.x = []
		self.pmtA = []
		self.pmtB = []
		#self.p.join()
		#self.actionExit.toggled.connect(self.closeEvent)


	def update(self):
		t0 = time.time()
		time.sleep(0.03)
		t1 = time.time()
		self.search_q.put([t0,t1,time.perf_counter()])

		#data = np.frombuffer(self.sa['data'].get_obj(), dtype='d').reshape(self.sa['shape'])

		#print(":",data.sum())
		#w = data[:,0]==0
		if self.out_q.qsize()>0:
			x,pmtA,pmtB = self.out_q.get()
			self.x.append(x)
			self.pmtA.append(pmtA)
			self.pmtB.append(pmtB)
			self.curveA1.setData(x=self.x, y=self.pmtA)
			self.curveB1.setData(x=self.x, y=self.pmtB)
		app.processEvents()
		#del data


	def closeEvent(self, evnt=None):
		print('closeEvent')
		self.q.put('kill')
		self.p.join()
		#self.p.terminate()




## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	__spec__ = None #"ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

	#app = QtGui.QApplication(sys.argv)
	#ex = Pico_view()

	#app.exec_()
	'''
	config = {	'ChA_VRange':'500mV','ChA_Offset':0,
				'ChB_VRange':'500mV','ChB_Offset':0,
				'sampleInterval':0.0001,'samplingDuration':0.003,
				'pico_pretrig':0.001,'n_captures':10,'trigSrc':'ext',
				'threshold_V':-0.350,'direction':'RISING'}
	'''
	config = {	'ChA_VRange':'20mV','ChA_Offset':0,
				'ChB_VRange':'20mV','ChB_Offset':0,
				'sampleInterval':2e-9,'samplingDuration':15e-9,
				'pico_pretrig':0.000,'n_captures':500,'trigSrc':'ext',
				'threshold_V':0.02,'direction':'RISING'}
	#'''

	sa_shape = (100000,3)
	unshared_arr = np.zeros(sa_shape[0]*sa_shape[1])
	sa = Array('d', int(np.prod(sa_shape)))
	sa = {'data':sa, 'shape':sa_shape}
	q = Queue()
	p = multiprocessing.Process(target=create_pico_reader,args=[config,sa,q])
	p.daemon = True
	p.start()
	while q.qsize()==0:
		time.sleep(0.1)
	search_q = Queue()
	out_q = Queue()
	search_p = multiprocessing.Process(target=search_time_range,args=[sa,search_q,out_q])
	search_p.daemon = True
	search_p.start()
	from hardware.E727 import *

	piStage = E727()
	print(piStage.ConnectUSBWithBaudRate())
	print(piStage.qSAI())
	print(piStage.SVO())
	print(piStage.qPOS(b'1'))
	#print(piStage.ATZ())
	x_range=np.arange(0,100,10)
	y_range=np.arange(0,100,1)
	z_range=np.arange(30,31,1)
	piStage.MOV(x_range.min(),axis=1, waitUntilReady=True)
	piStage.MOV(y_range.min(),axis=2, waitUntilReady=True)
	piStage.MOV(z_range.min(),axis=3, waitUntilReady=True)
	time_table = np.hstack((time.time(),piStage.qPOS()))
	Z = []
	Y = []
	X = []
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
					pos0 = piStage.qPOS()
					t0 = time.time()
					time_table = np.vstack((time_table, np.hstack((t0,pos0))))
					piStage.MOV(x,axis=1, waitUntilReady=True)
					pos1 = piStage.qPOS()
					t1 = time.time()
					time_table = np.vstack((time_table, np.hstack((t1,pos1))))
					search_q.put([t0,t1,10])
				forw = not forw

		out = []
		while not out_q.empty():
			out.append(out_q.get())
	except:
		traceback.print_exc()
		search_q.put('kill')
		q.put('kill')
		piStage.CloseConnection()
	search_q.put('kill')
	q.put('kill')
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
	show(0)
