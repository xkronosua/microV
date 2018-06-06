import sys,time,os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
import qdarkstyle
from pyqtgraph.Qt import QtGui, QtCore, uic
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt
from scipy.signal import argrelextrema
import scipy.misc
import matplotlib
from skimage.external.tifffile import imsave
import sharedmem
from scipy.signal import resample
from scipy.ndimage.measurements import center_of_mass
import pandas as pd
from scipy.interpolate import interp1d
import threading
#import SharedArray
from scipy.ndimage import gaussian_filter
from skimage.feature.peak import peak_local_max
from lmfit import Model
import csv
MODE = 'lab'

if len(sys.argv)>1:
	if sys.argv[1] == 'sim':
		MODE = 'sim'
		from hardware.sim.CCS200 import *
		from hardware.sim.ni import *
		from hardware.sim.E727 import *
		from hardware.sim.TDC001 import *
		from hardware.sim.picoscope import ps3000a
		from hardware.sim.AG_UC2 import AG_UC2
		import picopy
		from hardware.sim.andor.Shamrock import *
		from hardware.sim.andor.AndorCamera import *
		#from hardware.sim.PM100 import visa
		from hardware.sim.moco import MOCO
		from hardware.sim.HWP_stepper import HWP_stepper
else:
	from hardware.ni1 import *
	from hardware.CCS200 import *
	from hardware.E727 import *
	from hardware.TDC001 import APTMotor
	from nidaqmx.constants import AcquisitionType, TaskMode
	import nidaqmx
	from picoscope import ps3000a
	from hardware.AG_UC2 import AG_UC2
	import visa
	from ThorlabsPM100 import ThorlabsPM100
	import picopy
	from hardware.andor.Shamrock import *
	from hardware.andor.AndorCamera import *
	from hardware.HWP_stepper import HWP_stepper
	from hardware.moco import MOCO

from hardware.pico_radar import fastScan
from hardware.pico_multiproc_picopy import *

#get_ipython().run_line_magic('matplotlib', 'qt')

import traceback
from multiprocessing import Process, Queue, Array

from gui_save_restore import save_gui, restore_gui
from PyQt5.QtCore import QFileInfo, QSettings, QThread, pyqtSignal

def gaussian(x, amp, cen, wid,bg=0):
	"""1-d gaussian: gaussian(x, amp, cen, wid, bg)"""
	return bg + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def generate2Dtoolpath(Range_x, Range_y, mask=None,skip=2):
	'''zigzag toolpath by X axis with higher desity by mask'''
	Y,X = np.meshgrid(Range_y,Range_x)
	print("gen:",X.shape,Y.shape)
	Yi,Xi = np.meshgrid(np.arange(len(Range_y)),np.arange(len(Range_x)))

	if not mask is None and mask.shape == X.shape:
		newGrid = np.full(X.shape,np.nan)
		newGrid[::skip,::skip] = X[::skip,::skip]
		newGrid[skip//2::skip,skip//2::skip] = X[skip//2::skip,skip//2::skip]
		newGrid[mask]=X[mask]
	else:
		newGrid = X.copy()
	newGrid = newGrid.T
	newGrid[::2]=newGrid[::2].T[::-1].T
	newGrid = newGrid.T
	Xi = Xi.T
	Xi[::2]=Xi[::2].T[::-1].T
	Xi = Xi.T

	n,m = X.shape
	for i in range(m):
		yield b'2', i, Range_y[i]
		for j in range(n):
			val = newGrid[j,i]
			xi = Xi[j,i]
			if not np.isnan(val):
				yield b'1', xi, val
			else:
				continue


def zCompensation(ex_wl,zShift=0):
	"""
	Z position shift compensation
	"""
	cal = np.loadtxt('beamWaistZPosition.txt',skiprows=1)[:,1:]
	cal[:,1] = medfilt(cal[:,1],3)

	cal = cal[np.unique(cal[:,0],return_index = True)[-1]]
	cal[:,0] -= cal[:,0].min()
	fit = interp1d(cal[:,0],cal[:,1],kind='slinear', bounds_error=False,fill_value='extrapolate')
	return fit(ex_wl)-zShift

def photon_count(signal,t,dt_window,threshold):
	n = int(len(signal)//10)#int((t.max()-t.min())//dt_window)
	w = signal < threshold
	if sum(w) == 0:
		return np.zeros(len(t)), t, (t.max()-t.min())/n
	s = abs(signal*w)
	t = t[w]
	#s = signal.cumsum()

	print("N",n,len(signal),(t.max()-t.min()), dt_window)
	s = np.array([i.sum() for i in np.array_split(s,n)])
	print(s.shape)
	t_new = np.linspace(t.min(),t.max(),len(s))
	return s, t_new, (t.max()-t.min())/n

class TimerThread(QThread):
	update = pyqtSignal()

	def __init__(self, event):
		QThread.__init__(self)
		self.stopped = event

	def run(self):
		while not self.stopped.wait(0.1):
			self.update.emit()

class microV(QtGui.QMainWindow):
	settings = QSettings("gui.ini", QSettings.IniFormat)
	data = []
	alive = True
	piStage = E727()
	spectrometer = CCS200()
	try:
		HWP = APTMotor(83854487, HWTYPE=31)
	except:
		traceback.print_exc()
		HWP = None
	rotPiezoStage = AG_UC2()
	moco = None
	mocoSections = [0,30000,650000,1450000]
	#ps = ps3000a.PS3000a(connect=False)
	ps = None
	n_captures = None

	shamrock = ShamRockController()
	andorCCD = AndorCamera()
	HWP_stepper = None
	power_meter = None
	power_meter_instr = None#ThorlabsPM100(inst=inst)
	#DAQmx = nidaqmx.Task()
	#DAQmx = MultiChannelAnalogInput(["Dev1/ai0,Dev1/ai2"])
	live_pmtA = []
	live_pmtB = []
	live_x = []
	live_y = []
	live_integr_spectra = []
	pico_VRange_dict = {"Auto":20.0,'20mV':0.02,'50mV':0.05,'100mV':0.1,
			'200mV':0.2,'500mV':0.5,'1V': 1.0, '2V': 2.0,
			'5V': 5.0, '10V': 10.0, '20V': 20.0,}

	pico_shared_buf_shape = (10000,3)
	pico_shared_buf = None

	pico_control_queue = Queue()
	pico_config = {}
	data2D_A = np.array([])
	data2D_B = np.array([])
	dataMask = None
	NPsCenters = []
	nMeasLaserSpectra_probe_currentRow = 0

	processOut = Queue()
	shamrockConnect_thread = None
	andorCameraConnect_thread = None
	andorCCD_prev_centr_wavelength = -1

	def __init__(self, parent=None):
		QtGui.QMainWindow.__init__(self, parent)
		#from mainwindow import Ui_mw
		self.ui = uic.loadUi("microV.ui")#Ui_mw()
		self.ui.closeEvent = self.closeEvent
		self.ui.show()
		self._want_to_close = False

		self.calibrTimer = QtCore.QTimer()
		self.laserStatus = QtCore.QTimer()
		self.andorCameraLiveTimer = QtCore.QTimer()

		self.initUI()

		self.initPiStage()
		#self.initSpectrometer()
		try:
			self.initHWP()
		except:
			traceback.print_exc()
		#self.initPico()
		#self.initDAQmx()

		#self.scan_image()

		unshared_arr = np.zeros(self.pico_shared_buf_shape[0]*self.pico_shared_buf_shape[1])
		sa = Array('d', int(np.prod(self.pico_shared_buf_shape)))
		self.pico_shared_buf = {'data':sa, 'shape':self.pico_shared_buf_shape}
		self.ui.setLocale(QtCore.QLocale(QtCore.QLocale.C))

	###############################   Laser	####################################
	############################################################################

	def laserSetShutter(self):
		self.laserStatus.stop()
		state = self.ui.laserShutter.isChecked()
		if state==0:
			state = '1'
		else:
			state = '0'

		with open('laserIn','w+') as f:
			f.write('SHUTter '+state+'\n')
		self.laserStatus.start(500)

	def laserSetShutter_(self,state):
		self.laserStatus.stop()
		if state:
			state = '1'
		else:
			state = '0'
		with open('laserIn','w+') as f:
			f.write('SHUTter '+state+'\n')
		self.laserStatus.start(500)

	def laserSetWavelength(self,status=None,wavelength=None):
		self.laserStatus.stop()
		if wavelength is None or wavelength == False:
			wavelength = self.ui.laserWavelength_to_set.value()
		print(wavelength)
		with open('laserIn','w+') as f:
			f.write('WAVelength '+str(wavelength)+'\n')
		self.laserStatus.start(500)

	def laserSetWavelength_(self,status=None,wavelength=None):
		self.laserStatus.stop()
		self.ui.laserWavelength_to_set.setValue(wavelength)
		print(wavelength)
		with open('laserIn','w+') as f:
			f.write('WAVelength '+str(wavelength)+'\n')
		self.laserStatus.start(500)

	def onLaserStatus(self):
		status = ''
		with open('laserOut','r') as f:
			status = f.read()
		#print(status)
		try:
			status = status.split('\t')
			t = float(status[0])
			statusCode = status[1]
			shutter = int(status[2])
			power_int = float(status[3])
			wavelength = int(status[4])
			wavelength_ready = status[5] == 'True'
			self.ui.laserPower_internal.setValue(power_int)
			self.ui.laserWavelength.setValue(wavelength)
			self.statusBar_ExWavelength.setText("[Ex: %d nm]" % wavelength)
			self.ui.laserShutter.setChecked(shutter!=0)
			self.ui.actionShutter.blockSignals(True)
			self.ui.actionShutter.setChecked(shutter!=0)
			self.ui.actionShutter.blockSignals(False)
			if shutter!=0:
				self.statusBar_Shutter.setStyleSheet('color:red;')
			else:
				self.statusBar_Shutter.setStyleSheet('color:gray;')
			self.ui.laserWavelength_ready.setChecked(wavelength_ready)


		except:
			traceback.print_exc()
			print('Laser: noData')

	############################################################################
	###############################   DAQmx	#################################
	def connect_DAQmx(self,state):
		if state:
			self.initDAQmx()
		else:
			try:
				self.DAQmx.close()
			except:
				traceback.print_exc()

	def initDAQmx(self):
		self.DAQmx.ai_channels.add_ai_voltage_chan("Dev1/ai0,Dev1/ai2,Dev1/ai3", max_val=10, min_val=-10)
		self.DAQmx.timing.cfg_samp_clk_timing(10000, sample_mode=AcquisitionType.CONTINUOUS)
		self.DAQmx.control(TaskMode.TASK_COMMIT)
		self.DAQmx.triggers.start_trigger.cfg_anlg_edge_start_trig("Dev1/ai0",trigger_level=1.5)

		#self.DAQmx.configure()

	def readDAQmx(self,preview=False,print_dt=False):
		start = time.time()
		with nidaqmx.Task() as master_task:
			#master_task = nidaqmx.Task()
			master_task.ai_channels.add_ai_voltage_chan("Dev1/ai0,Dev1/ai2,Dev1/ai3", max_val=10, min_val=-10)
			master_task.timing.cfg_samp_clk_timing(
				100000, sample_mode=AcquisitionType.FINITE)
			master_task.control(TaskMode.TASK_COMMIT)
			master_task.triggers.start_trigger.cfg_dig_edge_start_trig("PFI0")
			master_task.start()
			#start = time.time()
			for i in range(100):
				master_data = master_task.read(number_of_samples_per_channel=1000)
				if master_task.is_task_done():
					break
			r,d,d1 = master_data
			#pp.pprint(master_data)
			#print(time.time()-start,master_task.is_task_done())
		r,d,d1 = master_data
		r = np.array(r)
		d = np.array(d)
		d1 = np.array(d1)

		d = d[int(self.ui.DAQmx_shift.value()):]
		d1 = d1[int(self.ui.DAQmx_shift.value()):]
		r = r[0:int(-self.ui.DAQmx_shift.value())]
		if self.ui.DAQmx_preview.isChecked():
			self.line_DAQmx_sig.setData(d)
			self.line_DAQmx_sig1.setData(d1)
			self.line_DAQmx_ref.setData(r)

		#pp.pprint(master_data)
		if print_dt:
			print("readDAQmx(tdiff):\n", time.time()-start)

		w = r>r.mean()
		out = abs(d[w].mean()-d[~w].mean())
		out1 = abs(d1[w].mean()-d1[~w].mean())
		print(time.time()-start,i)

		return out, out1

	def optimizeDAQmx(self):
		start = time.time()
		with nidaqmx.Task() as master_task:
			#master_task = nidaqmx.Task()
			master_task.ai_channels.add_ai_voltage_chan("Dev1/ai0,Dev1/ai2,Dev1/ai3", max_val=10, min_val=-10)

			master_task.timing.cfg_samp_clk_timing(
				100000, sample_mode=AcquisitionType.FINITE)

			master_task.control(TaskMode.TASK_COMMIT)

			master_task.triggers.start_trigger.cfg_dig_edge_start_trig("PFI0")


			master_task.start()
			#start = time.time()
			for i in range(100):
				master_data = master_task.read(number_of_samples_per_channel=1000)
				if master_task.is_task_done():
					break
			r,d,d1 = master_data
			#pp.pprint(master_data)
			#print(time.time()-start,master_task.is_task_done())
		r = np.array(r)
		d = np.array(d)
		data_shift = 0
		shift_array = []
		for data_shift in range(len(r)//2):
			d_ = d[int(data_shift):]
			r_ = r[0:int(-data_shift)]
			w = r_>r_.mean()
			out = abs(d_[w].mean()-d_[~w].mean())
			shift_array.append(out)


		shift_array = np.array(shift_array)
		self.line_DAQmx_ref.setData(shift_array)
		m = argrelextrema(shift_array, np.greater)[0]
		#pp.pprint(master_data)
		print("optimizeDAQmx(tdiff):\n", time.time()-start,m)
		self.ui.DAQmx_shift.setValue(m[1])

		return m

	############################################################################
	###############################   Picoscope	#############################

	def connect_pico(self,state):
		if state:
			self.initPico()
		else:
			try:
				del self.ps
				self.ps = None
			except:
				traceback.print_exc()

	def pico_set(self):
		self.n_captures = self.ui.pico_n_captures.value()
		self.pico_config['n_captures'] = self.n_captures
		self.pico_config['pulseFreq'] = 80e6
		if MODE == 'sim':
			self.pico_config['pulseFreq'] = 1666.
		ChA_VRange = self.ui.pico_ChA_VRange.currentText()
		#ChA_VRange = self.pico_VRange_dict[ChA_VRange]
		self.pico_config['ChA_VRange'] = ChA_VRange
		ChA_Offset = self.ui.pico_ChA_offset.value()
		self.pico_config['ChA_Offset'] = ChA_Offset
		ChB_VRange = self.ui.pico_ChB_VRange.currentText()
		self.pico_config['ChB_VRange'] = ChB_VRange
		#ChB_VRange = self.pico_VRange_dict[ChB_VRange]
		ChB_Offset = self.ui.pico_ChB_offset.value()
		self.pico_config['ChB_Offset'] = ChB_Offset

		self.ps.setChannel("A", coupling="DC", VRange=ChA_VRange, VOffset=ChA_Offset)
		self.ps.setChannel("B", coupling="DC", VRange=ChB_VRange, VOffset=ChB_Offset)

		sampleInterval = float(self.ui.pico_sampleInterval.text())
		samplingDuration = float(self.ui.pico_samplingDuration.text())
		pico_pretrig = float(self.ui.pico_pretrig.text())
		self.pico_config['sampleInterval'] = sampleInterval
		self.pico_config['samplingDuration'] = samplingDuration
		self.pico_config['pico_pretrig'] = pico_pretrig
		self.ps.setSamplingInterval(sampleInterval,samplingDuration,pre_trigger=pico_pretrig,
			number_of_frames=self.n_captures, downsample=1, downsample_mode='NONE')

		trigSrc = self.ui.pico_TrigSource.currentText()
		if trigSrc == 'External':
			trigSrc = 'ext'
		threshold_V = self.ui.pico_TrigThreshold.value()
		direction = self.ui.pico_Trig_mode.currentText().upper()
		self.pico_config['trigSrc'] = trigSrc
		self.pico_config['threshold_V'] = threshold_V
		self.pico_config['direction'] = direction


		self.ps.setSimpleTrigger(trigSrc=trigSrc, threshold_V=threshold_V, direction=direction,
								 timeout_ms=5, enabled=True)
		#self.samples_per_segment = self.ps.memorySegments(self.n_captures)
		#self.ps.setNoOfCaptures(self.n_captures)
		#print(self.n_captures)

	def initPico(self):

		#self.ps.open()
		self.ps = picopy.Pico3k()
		self.pico_set()

	def readPico(self):
		if not self.ui.connect_pico.isChecked():
			self.ui.connect_pico.setChecked(True)
		#dataA = np.zeros((self.n_captures, self.samples_per_segment), dtype=np.int16)
		#dataB = np.zeros((self.n_captures, self.samples_per_segment), dtype=np.int16)
		#t1 = time.time()

		#self.ps.runBlock()
		try:
			r = self.ps.capture_prep_block(return_scaled_array=1)



			if self.pico_config['n_captures'] == 1:
				N = int(self.pico_config['samplingDuration']*self.pico_config['pulseFreq'])

				if self.ui.pico_photon_counter.isChecked():
					dataA = r[0]['A']
					dataB = r[0]['B']
					dataA = dataA[0]
					dataB = dataB[0]
					dataT = r[2]
					scanA, scanT, dt = photon_count(dataA,dataT,15e-8,float(self.ui.pico_photon_counter_threshold.text()))
					scanB, scanT, dt = photon_count(dataB,dataT,15e-8,float(self.ui.pico_photon_counter_threshold.text()))
					print(dt)


				else:
					dataA = r[0]['A']
					dataB = r[0]['B']
					#print(dataA, dataA.shape)
					a = np.array(np.split( dataA[:,:int(dataA.shape[1]//N*N)],N,axis=1)).mean(axis=1)
					scanA = abs(a.max(axis=1) - a.min(axis=1))
					b = np.array(np.split( dataB[:,:int(dataB.shape[1]//N*N)],N,axis=1)).mean(axis=1)
					scanB = abs(b.max(axis=1) - b.min(axis=1))

					dataA = dataA.mean(axis=0)[:int(dataA.shape[1]//N*N)]
					dataB = dataB.mean(axis=0)[:int(dataB.shape[1]//N*N)]

					scanT = r[2][:len(dataB)]
					scanT = np.linspace(scanT.min(),scanT.max(),len(scanA))
			else:
				dataA = r[0]['A'].mean(axis=0)
				dataB = r[0]['B'].mean(axis=0)
				scanT = r[1]

			if self.ui.raw_data_preview.isChecked():
				t = np.linspace(0,float(self.ui.pico_samplingDuration.text()),len(dataA))
				try:
					#print(6)
					self.line_pico_ChA.setData(x=t,y=dataA)
					self.line_pico_ChB.setData(x=t,y=dataB)
				except :
					pass
					#traceback.print_exc()


			if self.ui.pico_AutoRange.isChecked():

				indexChA = self.ui.pico_ChA_VRange.currentIndex()
				ChA_VRange = self.pico_VRange_dict[self.ui.pico_ChA_VRange.currentText()]
				if abs(dataA.min())> ChA_VRange*0.9 and indexChA<8:
					indexChA += 1

				self.ui.pico_ChA_VRange.setCurrentIndex(indexChA)
				ChA_VRange = self.ui.pico_ChA_VRange.currentText()
				#print('AutoA',ChA_VRange)
				ChA_Offset = self.ui.pico_ChA_offset.value()
				self.ps.setChannel(channel="A", coupling="DC", VRange=ChA_VRange, VOffset=ChA_Offset)



				indexChB = self.ui.pico_ChB_VRange.currentIndex()
				ChB_VRange = self.pico_VRange_dict[self.ui.pico_ChB_VRange.currentText()]
				if abs(dataB.min())> ChB_VRange*0.9 and indexChB<8:
					indexChB += 1

				self.ui.pico_ChB_VRange.setCurrentIndex(indexChB)

				ChB_VRange = self.ui.pico_ChB_VRange.currentText()
				#ChB_VRange = self.pico_VRange_dict[ChB_VRange]
				ChB_Offset = self.ui.pico_ChB_offset.value()
				self.ps.setChannel(channel="B", coupling="DC", VRange=ChB_VRange, VOffset=ChB_Offset)

			#dataA = self.ps.rawToV(channel="A", dataRaw=dataA)
			#dataB = self.ps.rawToV(channel="B", dataRaw=dataB)
			if self.pico_config['n_captures'] == 1:
				dataA_p2p = scanA.mean()
				dataB_p2p = scanB.mean()
			else:
				dataA_p2p = abs(dataA.max() - dataA.min())
				dataB_p2p = abs(dataB.max() - dataB.min())

			self.ui.pico_ChA_value.setText(str(round(dataA_p2p,8)))
			self.ui.pico_ChB_value.setText(str(round(dataB_p2p,8)))
		except:
			traceback.print_exc()
			return [0,0]

		return dataA_p2p, dataB_p2p

	############################################################################
	###############################   Powermeter	 ###########################
	###############################   Thorlabs PM100 ###########################

	def pm100Connect(self,state):
		if state:
			rm = visa.ResourceManager()
			self.power_meter_instr = rm.open_resource(
				'USB0::0x1313::0x8078::P0011470::INSTR',timeout=1)
			self.power_meter = ThorlabsPM100(inst=self.power_meter_instr)
			self.readPower()
		else:
			self.power_meter_instr.close()

	def readPower(self):
		if not self.ui.pm100Connect.isChecked():
			self.ui.pm100Connect.setChecked(True)
		self.power_meter.sense.correction.wavelength = self.ui.laserWavelength.value()
		val = np.nan
		try:
			val = self.power_meter.read
			self.ui.pm100Power.setText(str(val))
		except:
			traceback.print_exc()

		return val

	def pm100Average(self,val):
		if not self.ui.pm100Connect.isChecked():
			self.ui.pm100Connect.setChecked(True)
		self.power_meter.sense.average.count = val

	############################################################################
	###############################   HWP	###################################

	def initHWP(self):
		pos = self.HWP.getPos()
		self.ui.HWP_angle.setText(str(round(pos,6)))

	def HWP_go(self):
		to_angle = self.ui.HWP_move_to_angle.value()
		self.HWP.mAbs(to_angle)
		pos = self.HWP.getPos()
		self.ui.HWP_angle.setText(str(round(pos,6)))

	def HWP_go_home(self):
		self.HWP.go_home()
		pos = self.HWP.getPos()
		self.ui.HWP_angle.setText(str(round(pos,6)))

	def HWP_negative_step(self):
		to_angle = -self.ui.HWP_rel_step.value()
		self.HWP.mRel(to_angle)
		pos = self.HWP.getPos()
		self.ui.HWP_angle.setText(str(round(pos,6)))
	def HWP_positive_step(self):
		to_angle = self.ui.HWP_rel_step.value()
		self.HWP.mRel(to_angle)
		pos = self.HWP.getPos()
		self.ui.HWP_angle.setText(str(round(pos,6)))

	############################################################################
	###############################   PiNanoCube	############################

	def initPiStage(self):
		print(self.piStage.ConnectUSB())
		print(self.piStage.qSAI())
		print(self.piStage.SVO(b'1 2 3', [True, True, True]))
		vel = self.ui.Pi_Velocity.value()
		self.piStage.VEL([vel]*3,b'1 2 3')
		print("VEL:",self.piStage.qVEL())
		#print(self.piStage.DCO([1,1,1],b'1 2 3'))
		#print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=1,waitUntilReady=True))
		time.sleep(0.2)
		#print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=2,waitUntilReady=True))
		#time.sleep(0.2)
		#print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=3,waitUntilReady=True))
		pos = self.piStage.qPOS()
		if sum(pos<-100) or sum(pos>130):
			print(self.piStage.ATZ())
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)
		time.sleep(1)

	def setUiPiPos(self,pos):
		self.ui.Pi_XPos.setText(str(pos[0]))
		self.ui.Pi_YPos.setText(str(pos[1]))
		self.ui.Pi_ZPos.setText(str(pos[2]))
		self.statusBar_Position_X.setText('%.5f'%pos[0])
		self.statusBar_Position_Y.setText('%.5f'%pos[1])
		self.statusBar_Position_Z.setText('%.5f'%pos[2])
		self.piStage_position.setData(x=[pos[0]],y=[pos[1]])
		self.piStage_position1.setData(x=[pos[0]],y=[pos[1]])
		self.ui.zSlider.blockSignals(True)
		self.ui.zSlider.setValue(pos[2]*1000)
		self.ui.zSlider.blockSignals(False)

	def setPiStateZPosition(self,index):
		print(self.piStage.MOV([index/1000],axis=b'3',waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_X_go(self):
		pos = self.ui.Pi_X_move_to.value()
		print(self.piStage.MOV([pos],axis=b'1',waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_Y_go(self):
		pos = self.ui.Pi_Y_move_to.value()
		print(self.piStage.MOV([pos],axis=b'2',waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_Z_go(self):
		pos = self.ui.Pi_Z_move_to.value()
		print(self.piStage.MOV([pos],axis=b'3',waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_XYZ_50mkm(self):
		print(self.piStage.MOV([50,50,50],axis=b'1 2 3',waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_autoZero(self):
		print(self.piStage.ATZ())
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_Set(self):
		vel = self.ui.Pi_Velocity.value()
		self.piStage.VEL([vel]*3,b'1 2 3')
		print('VEL:',self.piStage.qVEL())

	############################################################################
	###############################   Shamrock	################################
	def _shamrockConnect(self):
		self.ui.shamrockConnect.blockSignals(True)
		self.ui.shamrockConnect.setChecked(False)
		self.shamrock.Initialize()
		self.shamrock.Connect()
		self.ui.shamrockConnect.setChecked(True)
		self.ui.shamrockConnect.blockSignals(False)


		wavelength = self.shamrock.shamrock.GetWavelength()
		self.ui.shamrockWavelength.setValue(wavelength)
		port = self.shamrock.shamrock.GetPort()
		self.ui.shamrockPort.blockSignals(True)
		self.ui.shamrockPort.setCurrentIndex(port)
		self.ui.shamrockPort.blockSignals(False)
		grating = self.shamrock.shamrock.GetGrating()
		self.ui.shamrockGrating.blockSignals(True)
		self.ui.shamrockGrating.setCurrentIndex(grating-1)
		self.ui.shamrockGrating.blockSignals(False)

	def shamrockConnect(self, state):
		if state:
			if not self.shamrockConnect_thread is None:
				print('shamrockConnect_thread:join')
				self.shamrockConnect_thread.join()


				#self.shamrockConnect_thread = None


			self.shamrockConnect_thread = threading.Thread(target=self._shamrockConnect)
			self.shamrockConnect_thread.daemon = True
			self.shamrockConnect_thread.start()
		else:
			self.shamrock.Close()
			self.shamrockConnect_thread = None


	def shamrockSetWavelength(self, wl=None):

		if not self.ui.shamrockConnect.isChecked():
			self.ui.shamrockConnect.setChecked(True)
		if wl is None or wl == False:
			wl = self.ui.shamrockWavelength_to_set.value()
		print(wl)
		self.shamrock.shamrock.SetWavelength(wl)
		wavelength = self.shamrock.shamrock.GetWavelength()
		self.ui.shamrockWavelength.setValue(wavelength)
		if self.ui.shamrock_use_MOCO.isChecked():
			if wavelength < self.ui.shamrock_MOCO_limit.value():
				self.ui.mocoSection.setCurrentIndex(2)
			if wavelength >= self.ui.shamrock_MOCO_limit.value():
				self.ui.mocoSection.setCurrentIndex(1)

	def shamrockSetPort(self,val):
		if not self.ui.shamrockConnect.isChecked():
			self.ui.shamrockConnect.setChecked(True)
		port = val
		self.shamrock.shamrock.SetPort(port)
		port = self.shamrock.shamrock.GetPort()
		self.ui.shamrockPort.blockSignals(True)
		self.ui.shamrockPort.setCurrentIndex(port)
		self.ui.shamrockPort.blockSignals(False)
		self.statusBar_ShamrockPort.setText('['+self.ui.shamrockPort.currentText()+']')

	def shamrockSetGrating(self,val):
		if not self.ui.shamrockConnect.isChecked():
			self.ui.shamrockConnect.setChecked(True)
		grating = val+1
		self.shamrock.shamrock.SetGrating(grating)
		grating = self.shamrock.shamrock.GetGrating()
		self.ui.shamrockGrating.blockSignals(True)
		self.ui.shamrockGrating.setCurrentIndex(grating-1)
		self.ui.shamrockGrating.blockSignals(False)

	############################################################################
	###############################   AndorCamera	############################
	def _andorCamera_connect(self):
		self.ui.andorCameraConnect.blockSignals(True)
		self.ui.andorCameraConnect.setChecked(False)
		self.andorCCD.Initialize()
		self.ui.andorCameraConnect.setChecked(True)
		self.ui.andorCameraConnect.blockSignals(False)

		self.andorCCD.SetExposureTime(self.ui.andorCameraExposure.value())
		self.andorCCDBaseline = np.array([])
		size=self.andorCCD.GetPixelSize()
		self.andorCCD_wavelength = np.arange(size[0])
		self.andorCCD_wavelength_center = self.ui.shamrockWavelength.value()
		if not self.ui.shamrockConnect.isChecked():
			self.ui.shamrockConnect.setChecked(True)
			#self.shamrockConnect_thread.join()
			t0 = time.time()
			while time.time()-t0<1000:
				time.sleep(1)
				if self.ui.shamrockConnect.isChecked(): break
				print('wait_shamrock')
		shape=self.andorCCD.GetDetector()
		size=self.andorCCD.GetPixelSize()
		self.shamrock.shamrock.SetPixelWidth(size[0])
		self.shamrock.shamrock.SetNumberPixels(shape[0])
		self.andorCCD_wavelength = self.shamrock.shamrock.GetCalibration()
		self.andorCCD_prev_centr_wavelength = self.ui.shamrockWavelength.value()

	def andorCameraConnect(self, state=True):
		if state:
			if not self.andorCameraConnect_thread is None:
				print('andorCameraConnect_thread:join')
				self.andorCameraConnect_thread.join()

				#self.andorCameraConnect_thread = None
			self._andorCamera_connect()
			#self.andorCameraConnect_thread = threading.Thread(target=self._andorCamera_connect)
			#self.andorCameraConnect_thread.daemon = True
			#self.andorCameraConnect_thread.start()
			#self.andorCameraConnect_thread.join()
		else:
			self.andorCCD.ShutDown()
			self.andorCameraConnect_thread = None

	def andorCameraSetExposure(self,val):
		if not self.ui.andorCameraConnect.isChecked():
			self.ui.andorCameraConnect.setChecked(True)
		self.andorCCD.SetExposureTime(val)
		val = self.andorCCD.GetAcquisitionTimings()[0]

	def andorCameraSetExposure_(self,val):
		self.ui.andorCameraExposure.setValue(val)
		self.andorCamera_prevExposure = val

	def andorCameraSetReadoutMode(self,val):
		if not self.ui.andorCameraConnect.isChecked():
			self.ui.andorCameraConnect.setChecked(True)
		mode = self.ui.andorCameraReadoutMode.currentText()
		self.andorCCD.SetReadMode(mode)

	def andorCameraGetData(self,state=False, line_index=0):
		if not self.ui.andorCameraConnect.isChecked():
			self.ui.andorCameraConnect.setChecked(True)
		data_center = 0
		data = np.array([])
		if state:
			if not self.ui.andorCameraLive.isChecked():
				print('andorCameraGetData')
				self.andorCCD.StartAcquisition()
				self.andorCCD.WaitForAcquisition()
				data = self.andorCCD.GetMostRecentImage()
				if len(self.andorCCDBaseline) == len(data):
					data-= self.andorCCDBaseline
				print(data)

				if not self.ui.shamrockConnect.isChecked():
					self.ui.shamrockConnect.setChecked(True)
				if self.andorCCD_prev_centr_wavelength != self.ui.shamrockWavelength.value():
					#c = float(self.ui.shamrockWavelength.text())
					#w_c = [c-20, c+20]
					#if len(integr_range)>0:
					#	w_c = integr_range
					#if not w_c == self.andorCCD_wavelength_center:
					shape=self.andorCCD.GetDetector()
					size=self.andorCCD.GetPixelSize()
					self.shamrock.shamrock.SetPixelWidth(size[0])
					self.shamrock.shamrock.SetNumberPixels(shape[0])
					self.andorCCD_wavelength = self.shamrock.shamrock.GetCalibration()
					self.andorCCD_prev_centr_wavelength = self.ui.shamrockWavelength.value()
					#w_r = (self.andorCCD_wavelength>w_c[0])&(self.andorCCD_wavelength<w_c[1])
					#data_center = float(data[w_r].mean())
				self.line_spectra[line_index].setData(x=self.andorCCD_wavelength, y=data)
				self.ui.andorCameraGetData.blockSignals(True)
				self.ui.actionSpectra.blockSignals(True)
				self.ui.andorCameraGetData.setChecked(False)
				self.ui.actionSpectra.setChecked(False)
				self.ui.andorCameraGetData.blockSignals(False)
				self.ui.actionSpectra.blockSignals(False)

			else:
				self.andorCameraLiveTimer.start(10)
		else:
			self.andorCameraLiveTimer.stop()
		return data, self.andorCCD_wavelength

	def onAndorCameraLiveTimeout(self):
		if not self.ui.andorCameraConnect.isChecked():
			self.ui.andorCameraConnect.setChecked(True)

		self.andorCameraLiveTimer.stop()
		self.andorCCD.StartAcquisition()
		self.andorCCD.WaitForAcquisition()
		data = self.andorCCD.GetMostRecentImage()
		print(data)
		wavelength = np.arange(len(data))
		if self.ui.shamrockConnect.isChecked():
			shape=self.andorCCD.GetDetector()
			size=self.andorCCD.GetPixelSize()
			self.shamrock.shamrock.SetPixelWidth(size[0])
			self.shamrock.shamrock.SetNumberPixels(shape[0])
			self.shamrock.shamrock.GetCalibration()
			wavelength = self.shamrock.shamrock.GetCalibration()
		self.line_spectra[0].setData(x=wavelength, y = data)
		if data.max()>40000 and self.ui.andorCameraExposureAdaptive.isChecked():
			self.ui.andorCameraExposure.setValue(self.ui.andorCameraExposure.value()*0.8)
		self.andorCameraLiveTimer.start(100)

	def andorCameraGetBaseline(self):
		print('andorCameraGetBaseline>')
		if not self.ui.andorCameraConnect.isChecked():
			self.ui.andorCameraConnect.setChecked(True)


		self.andorCCD.StartAcquisition()
		self.andorCCD.WaitForAcquisition()
		data = self.andorCCD.GetMostRecentImage()
		self.andorCCDBaseline = data

		if not self.ui.shamrockConnect.isChecked():
			self.ui.shamrockConnect.setChecked(True)
		if self.andorCCD_prev_centr_wavelength != self.ui.shamrockWavelength.value():
			shape=self.andorCCD.GetDetector()
			size=self.andorCCD.GetPixelSize()
			self.shamrock.shamrock.SetPixelWidth(size[0])
			self.shamrock.shamrock.SetNumberPixels(shape[0])
			self.shamrock.shamrock.GetCalibration()
			self.andorCCD_wavelength = self.shamrock.shamrock.GetCalibration()
			self.andorCCD_prev_centr_wavelength = self.ui.shamrockWavelength.value()
		self.line_spectra[-1].setData(x=self.andorCCD_wavelength, y = data)
		print('<andorCameraGetBaseline')
	def andorCameraCleanLines(self):
		for i in range(len(self.line_spectra)):
			self.line_spectra[i].setData(x=[], y = [])
	############################################################################
	###############################   HWP_stepper	############################

	def HWP_stepper_Connect(self,state):
		if state:
			angle = 0
			try:
				angle = self.settings.value('HWP_stepper_angle',type=float)
			except:
				traceback.print_exc()
			self.HWP_stepper = HWP_stepper(4000,0.01, "dev1/ctr1", reset=True,currentAngle=angle)

			self.ui.HWP_stepper_angle.setText(str(angle))

		else:
			try:
				self.HWP_stepper.close()
			except:
				traceback.print_exc()
			self.HWP_stepper = None
	def HWP_stepper_moveTo(self,angle,wait=True):
		if not self.ui.HWP_stepper_Connect.isChecked():
			self.ui.HWP_stepper_Connect.setChecked(True)
		self.settings.setValue('HWP_stepper_angle',angle)
		self.ui.HWP_stepper_angle.setText(str(angle))
		self.HWP_stepper.moveTo(float(angle),wait=True)
	def HWP_stepper_getAngle(self):
		if not self.ui.HWP_stepper_Connect.isChecked():
			self.ui.HWP_stepper_Connect.setChecked(True)
		ang = self.HWP_stepper.getAngle()
		self.ui.HWP_stepper_angle.setText(str(ang))
		return ang
	def HWP_stepper_MoveTo_Go(self):
		if not self.ui.HWP_stepper_Connect.isChecked():
			self.ui.HWP_stepper_Connect.setChecked(True)

		if not self.HWP_stepper is None:
			angle = self.ui.HWP_stepper_MoveTo.value()
			wait = self.ui.HWP_stepper_wait.isChecked()
			self.HWP_stepper.moveTo(angle,wait=wait)
			angle_ = self.HWP_stepper.getAngle()
			self.settings.setValue('HWP_stepper_angle',angle_)
			self.ui.HWP_stepper_angle.setText(str(angle_))
	def HWP_stepper_CW(self):
		if not self.ui.HWP_stepper_Connect.isChecked():
			self.ui.HWP_stepper_Connect.setChecked(True)
		if not self.HWP_stepper is None:
			self.HWP_stepper.direction(1)
			step = self.ui.HWP_stepper_step.value()
			wait = self.ui.HWP_stepper_wait.isChecked()
			self.HWP_stepper.start(step=step,wait=wait)
			angle_ = self.HWP_stepper.getAngle()
			self.settings.setValue('HWP_stepper_angle',angle_)
			self.ui.HWP_stepper_angle.setText(str(angle_))
	def HWP_stepper_CCW(self):
		if not self.ui.HWP_stepper_Connect.isChecked():
			self.ui.HWP_stepper_Connect.setChecked(True)
		if not self.HWP_stepper is None:
			self.HWP_stepper.direction(0)
			step = self.ui.HWP_stepper_step.value()
			wait = self.ui.HWP_stepper_wait.isChecked()
			self.HWP_stepper.start(step=step,wait=wait)
			angle_ = self.HWP_stepper.getAngle()
			self.settings.setValue('HWP_stepper_angle',angle_)
			self.ui.HWP_stepper_angle.setText(str(angle_))

	def HWP_stepper_Reset(self):
		if not self.ui.HWP_stepper_Connect.isChecked():
			self.ui.HWP_stepper_Connect.setChecked(True)
		if not self.HWP_stepper is None:
			self.HWP_stepper.resetAngel()
			self.settings.setValue('HWP_stepper_angle',0)
			self.ui.HWP_stepper_angle.setText(str(0))

	############################################################################
	###############################   rotPiezoStage	#########################

	def connect_rotPiezoStage(self,state):
		if state:
			self.rotPiezoStage.connect()
		else:
			self.rotPiezoStage.close()
	def rotPiezoStage_Go(self):
		toAngle = self.ui.rotPiezoStage_Step.value()
		wait = self.ui.rotPiezoStage_wait.isChecked()
		self.rotPiezoStage.move(toAngle,waitUntilReady=wait)
		self.ui.rotPiezoStage_Angle.setText(str(self.rotPiezoStage.getAngle()))

	############################################################################
	###############################   Spectrometer	##########################

	def initSpectrometer(self):
		print(self.spectrometer.init())
		print(self.spectrometer.setIntegrationTime(self.ui.usbSpectr_integr_time.value()))
	def usbSpectr_set_integr_time(self):
		print(self.spectrometer.setIntegrationTime(self.ui.usbSpectr_integr_time.value()))

	def getSpectra(self):
		self.spectrometer.startScanExtTrg()
		#self.spectrometer.getDeviceStatus()
		data = self.spectrometer.getScanData()
		return data

	############################################################################
	###############################   MOCO	##########################

	def mocoConnect(self,state):
		if state:
			self.moco = MOCO()
			self.ui.mocoPosition.setValue(self.moco.getPosition())
			self.mocoGetSection()
		else:
			if not self.moco is None:
				self.moco.close()
				self.moco = None

	def mocoGetSection(self):
		index = -1
		if not self.moco is None:
			pos = self.moco.getPosition()
			self.ui.mocoPosition.setValue(pos)
			for j,section in enumerate(self.mocoSections):
				if abs(pos-section)<10000:
					index = j

		if index!=-1:
			self.ui.mocoSection.setCurrentIndex(index)

		return index
	def mocoMoveAbs(self,state=1,pos=None):
		if not self.ui.mocoConnect.isChecked():
			self.ui.mocoConnect.setChecked(True)

		if not self.moco is None:
			if pos is None:
				pos = self.ui.mocoMoveAbs_target.value()
			print(pos)
			self.moco.moveAbs(pos,True)
			self.mocoGetSection()
			pos = self.moco.getPosition()
			self.ui.mocoPosition.setValue(pos)

	def mocoSetSection(self, section):
		if not self.ui.mocoConnect.isChecked():
			self.ui.mocoConnect.setChecked(True)

		if not self.moco is None:
			self.moco.moveAbs(self.mocoSections[section],True)
			pos = self.moco.getPosition()
			self.ui.mocoPosition.setValue(pos)
			self.statusBar_Filter.setText('['+self.ui.mocoSection.currentText()+']')

	############################################################################
	###############################   scan3D	################################

	def scan3D_path_dialog(self):
		fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', self.ui.scan3D_path.text())
		if type(fname)==tuple:
			fname = fname[0]
		try:
			self.ui.scan3D_path.setText(fname)
		except:
			traceback.print_exc()
	def startCalibr(self,state):
		if state:
			if not self.ui.connect_pico.isChecked():
				self.ui.connect_pico.setChecked(True)
			#self.calibrTimer.start(100)
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.live_x = np.array([])
			self.live_y = np.array([])

			self.live_integr_spectra = np.array([])
			self.startCalibr_thread = threading.Thread(target=self.onCalibrTimer)
			self.startCalibr_thread.daemon = True
			self.startCalibr_thread.start()
		else:
			self.calibrTimer.stop()

	def onCalibrTimer(self):
		#self.calibrTimer.stop()
		#spectra = np.zeros(3648)#self.getSpectra()
		while self.ui.startCalibr.isChecked() and not self.ui.actionStop.isChecked():
			pmt_valA,pmt_valB = self.readPico()#self.readDAQmx(print_dt=True)
			self.live_pmtA = np.hstack((self.live_pmtA,pmt_valA))
			self.live_pmtB = np.hstack((self.live_pmtB,pmt_valB))
			if len(self.live_pmtA)>800:
				self.live_pmtA = self.live_pmtA[1:]
			if len(self.live_pmtB)>800:
				self.live_pmtB = self.live_pmtB[1:]
			#s_from = self.ui.usbSpectr_from.value()
			#s_to = self.ui.usbSpectr_to.value()
			#self.live_integr_spectra.append(np.sum(spectra[s_from:s_to])/1000)
			#setLine(spectra)
			self.line_pmtA.setData(self.live_pmtA)
			self.line_pmtB.setData(self.live_pmtB)
			#self.pw_preview.update()
			#self.pw1.update()
			#self.pw.update()

			time.sleep(0.01)
		#self.line_spectra.setData(self.live_integr_spectra)

		#self.calibrTimer.start(100)
		#app.processEvents()
	def confParam_scan(self, state):
		if state:
			self.confParam_thread = threading.Thread(target=self.confParam_)
			self.confParam_thread.daemon = True
			self.confParam_thread.start()

	def confParam_scan_(self, state=True):
		if state:
			start_wavelength = self.ui.calibr_wavelength_start.value()
			end_wavelength = self.ui.calibr_wavelength_end.value()
			step_wavelength = self.ui.calibr_wavelength_step.value()
			if start_wavelength > end_wavelength:
				step_wavelength = -step_wavelength
			self.scan3DisAlive = True
			wl_range = np.arange(start_wavelength,end_wavelength+step_wavelength,step_wavelength)
			with pd.HDFStore('data/confParam_scan'+str(round(time.time()))+'.h5') as store:
				store.keys()
				self.live_x = np.array([])
				self.live_integr_spectra = np.array([])
				self.live_pmtA = np.array([])
				self.live_pmtB = np.array([])

				for wl in wl_range:
					self.laserSetWavelength_(status=1,wavelength=wl)

					time.sleep(3)
					t0 = time.time()
					while not wl == float(self.ui.laserWavelength.text()) and time.time()-t0<10 and self.scan3DisAlive or not self.ui.confParam_scan.isChecked():
						time.sleep(0.5)
						print('wait:Laser')
						app.processEvents()


					self.shamrockSetWavelength((wl/2+wl/3)/2)

					#integr_intens,intens,wavelength_arr = self.andorCameraGetData(state=1,integr_range=[wl/3-20,wl/3+20])


					start_Z = self.ui.confParam_scan_start.value()
					end_Z = self.ui.confParam_scan_end.value()
					step_Z = self.ui.confParam_scan_step.value()
					Range_Z = np.arange(start_Z,end_Z,step_Z)

					print(self.piStage.MOV([0],axis=b'3',waitUntilReady=True))
					self.andorCameraGetBaseline()

					df = pd.DataFrame(self.andorCCDBaseline, index=self.andorCCD_wavelength,columns=['baseline'])
					for z in Range_Z:
						print(self.piStage.MOV([z],axis=b'3',waitUntilReady=True))
						real_position = self.piStage.qPOS()
						self.setUiPiPos(pos=real_position)
						z_real = real_position[2]
						intens,wavelength_arr = self.andorCameraGetData(state=1)
						w2 = (wavelength_arr>(wl/2-20))&(wavelength_arr<(wl/2+20))
						w3 = (wavelength_arr>(wl/3-20))&(wavelength_arr<(wl/3+20))

						integr_intensA = intens[w2].sum()
						integr_intensB = intens[w3].sum()
						df[str(z)] = intens
						self.live_x = np.hstack((self.live_x, z_real))
						self.live_pmtA = np.hstack((self.live_pmtA, integr_intensA))
						self.live_pmtB = np.hstack((self.live_pmtB, integr_intensB))

						self.line_pmtA.setData(x=self.live_x,y=self.live_pmtA)
						self.line_pmtB.setData(x=self.live_x,y=self.live_pmtB)
						app.processEvents()
						if not self.scan3DisAlive or not self.ui.confParam_scan.isChecked():
							store.put("scan_"+str(wl), df)
							return
							break
					spectra_name = "scan_"+str(wl)
					store.put(spectra_name, df)
					store.get_storer(spectra_name).attrs.centerXYZ = real_position
					store.get_storer(spectra_name).attrs.zRange = Range_Z
					store.get_storer(spectra_name).attrs.exposure = self.ui.andorCameraExposure.value()
					store.get_storer(spectra_name).attrs.laserBultInPower = self.ui.laserPower_internal.value()
					store.get_storer(spectra_name).attrs.HWP_power = float(self.ui.HWP_angle.text())
					store.get_storer(spectra_name).attrs.HWP_stepper = float(self.ui.HWP_stepper_angle.text())
					store.get_storer(spectra_name).attrs.endTime = time.time()
					store.get_storer(spectra_name).attrs.filter = self.ui.mocoSection.currentText()

		else:
			self.scan3DisAlive = False
	def laserPowerCalibr(self, state):
		if state:
			self.laserPowerCalibr_thread = threading.Thread(target=self.laserPowerCalibr_)
			self.laserPowerCalibr_thread.daemon = True
			self.laserPowerCalibr_thread.start()

	def laserPowerCalibr_(self, state=True):
		mode = self.ui.powerCalibrMode.currentText()
		if mode == 'HWP_Power':
			def move_function(pos):
				self.HWP.mAbs(pos)
				pos = self.HWP.getPos()
				self.ui.HWP_angle.setText(str(round(pos,6)))
		elif mode == 'HWP_Polarization':
			def move_function(pos):
				if not self.ui.HWP_stepper_Connect.isChecked():
					self.ui.HWP_stepper_Connect.setChecked(True)
				self.HWP_stepper_moveTo(float(pos),wait=True)
				pos = self.HWP_stepper.getAngle()
				self.ui.HWP_stepper_angle.setText(str(round(pos,6)))
		else:
			return
		if state:
			start_wavelength = self.ui.calibr_wavelength_start.value()
			end_wavelength = self.ui.calibr_wavelength_end.value()
			step_wavelength = self.ui.calibr_wavelength_step.value()
			if start_wavelength > end_wavelength:
				step_wavelength = -step_wavelength
			self.scan3DisAlive = True
			wl_range = np.arange(start_wavelength,end_wavelength+step_wavelength,step_wavelength)
			with pd.HDFStore('data/powerCalibr'+str(round(time.time()))+'.h5') as store:
				store.keys()
				self.live_x = np.array([])
				self.live_integr_spectra = np.array([])
				self.live_pmtA = np.array([])
				self.live_pmtB = np.array([])

				for wl in wl_range:
					self.laserSetWavelength_(status=1,wavelength=wl)

					time.sleep(3)
					t0 = time.time()
					while not wl == float(self.ui.laserWavelength.text()) and time.time()-t0<10 :
						time.sleep(0.5)
						print('wait:Laser')
						app.processEvents()

						if self.ui.actionPause.isChecked():
							while self.ui.actionPause.isChecked():
								app.processEvents()
						if self.ui.actionStop.isChecked(): break

					if self.ui.actionPause.isChecked():
						while self.ui.actionPause.isChecked():
							app.processEvents()
					if self.ui.actionStop.isChecked(): break
					start_angle = self.ui.calibr_start.value()
					end_angle = self.ui.calibr_end.value()
					step_angle = self.ui.calibr_step.value()

					steps_range = np.arange(start_angle, end_angle+step_angle, step_angle)
					power = self.readPower()
					bi_power = self.ui.laserPower_internal.value()
					initData = [power,bi_power,time.time()]
					data = np.zeros((len(steps_range),len(initData)))


					for j,new_pos in enumerate(steps_range):
						move_function(new_pos)
						if not self.alive: break

						HWP_angle = float(self.ui.HWP_angle.text())
						HWP_stepper_angle = float(self.ui.HWP_stepper_angle.text())

						if mode == 'HWP_Power':
							x = HWP_angle
						elif mode == 'HWP_Polarization':
							x = HWP_stepper_angle
						power = self.readPower()
						bi_power = self.ui.laserPower_internal.value()

						data[j] = np.array([power,bi_power,time.time()])

						self.live_x = np.hstack((self.live_x, x))
						self.live_pmtA = np.hstack((self.live_pmtA, power))
						self.live_pmtB = np.hstack((self.live_pmtB, bi_power))

						self.line_pmtA.setData(x=self.live_x,y=self.live_pmtA)
						self.line_pico_ChA.setData(x=self.live_x,y=self.live_pmtB)


						app.processEvents()


						app.processEvents()
						if not self.scan3DisAlive or not self.ui.powerCalibr_go.isChecked():
							df = pd.DataFrame(data, index=steps_range,columns=['power','laserBultInPower','Time'])
							store.put("scan_"+str(wl), df)
							return
							break
						if self.ui.actionPause.isChecked():
							while self.ui.actionPause.isChecked():
								app.processEvents()
						if self.ui.actionStop.isChecked(): break
					df = pd.DataFrame(data, index=steps_range,columns=['power','laserBultInPower','Time'])
					calibr_name = "powerCalibr_"+str(wl)
					store.put(calibr_name, df)

		else:
			self.scan3DisAlive = False


	def meas_laser_spectra_go(self,state):
		if state:
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			start_wavelength = self.ui.meas_laser_spectra_start.value()
			end_wavelength = self.ui.meas_laser_spectra_end.value()
			step_wavelength = self.ui.meas_laser_spectra_step.value()
			self.scan3DisAlive = True
			wl_range = np.arange(start_wavelength,end_wavelength,step_wavelength)
			with pd.HDFStore('data/measLaserSpectra'+str(round(time.time()))+'.h5') as store:
				store.keys()
				self.live_x = np.array([])
				self.live_integr_spectra = np.array([])

				for wl in wl_range:
					time_list = []
					time_list.append(time.time())
					self.laserSetWavelength_(status=1,wavelength=wl)

					#time.sleep(3)
					for i in range(100):
						time.sleep(0.03)
						app.processEvents()
					t0 = time.time()

					while not wl == float(self.ui.laserWavelength.text()) and time.time()-t0<10 and self.scan3DisAlive:
						time.sleep(0.1)
						print('wait:Laser')
						app.processEvents()
						if not self.scan3DisAlive:
							break

					time_list.append(time.time())

					bg_center = np.array([ float(self.ui.meas_laser_spectra_probe.item(0,i).text()) for i in range(3)])


					spectra_center = np.array([ float(self.ui.meas_laser_spectra_probe.item(1,i).text()) for i in range(3)])


					if self.ui.meas_laser_spectra_track.isChecked():
						self.ui.shamrockPort.setCurrentIndex(1)
						self.shamrockSetWavelength(wl/3)
						centerA, centerB, panelA, panelB = self.center_optim()

						print(centerA, centerB)

						if self.ui.meas_laser_spectra_track_channel.currentIndex()==0:
							center = centerA
						else:
							center = centerB

						pos = list(self.rectROI.pos())
						size = list(self.rectROI.size())
						pos[0] = center[0] - size[0]/2
						pos[1] = center[1] - size[1]/2

						delta = center - spectra_center
						bg_center = bg_center + delta
						spectra_center = center
						self.ui.meas_laser_spectra_probe.item(0,0).setText(str(bg_center[0]))
						self.ui.meas_laser_spectra_probe.item(0,1).setText(str(bg_center[1]))
						self.ui.meas_laser_spectra_probe.item(0,2).setText(str(bg_center[2]))

						self.ui.meas_laser_spectra_probe.item(1,0).setText(str(center[0]))
						self.ui.meas_laser_spectra_probe.item(1,1).setText(str(center[1]))
						self.ui.meas_laser_spectra_probe.item(1,2).setText(str(center[2]))

						self.rectROI.setPos(pos)
						z_start = float(self.ui.scan3D_config.item(0,1).text())
						z_end = float(self.ui.scan3D_config.item(0,2).text())
						self.ui.scan3D_config.item(0,1).setText(str(z_start+delta[2]))
						self.ui.scan3D_config.item(0,2).setText(str(z_end+delta[2]))



					if not self.scan3DisAlive:
						break
					time_list.append(time.time())

					self.ui.shamrockPort.setCurrentIndex(0)
					self.shamrockSetWavelength((wl/2+wl/3)/2)

					self.piStage.MOV(bg_center,b' 1 2 3',waitUntilReady=True)
					pos = self.piStage.qPOS()
					self.setUiPiPos(pos=pos)
					self.andorCameraGetBaseline()

					self.piStage.MOV(spectra_center,b' 1 2 3',waitUntilReady=True)
					pos = self.piStage.qPOS()
					self.setUiPiPos(pos=pos)


					#self.andorCameraGetBaseline()

					#integr_intens,intens,wavelength_arr = self.andorCameraGetData(state=1,integr_range=[wl/3-20,wl/3+20])


					#start_Z = self.ui.confParam_scan_start.value()
					#end_Z = self.ui.confParam_scan_end.value()
					#step_Z = self.ui.confParam_scan_step.value()
					#Range_Z = np.arange(start_Z,end_Z,step_Z)

					df = pd.DataFrame(self.andorCCDBaseline, index=self.andorCCD_wavelength,columns=['baseline'])
					#for z in Range_Z:
					#	print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
					#	real_position = self.piStage.qPOS()
					#	z_real = real_position[2]
					time_list.append(time.time())
					intens,wavelength_arr = self.andorCameraGetData(state=1)
					time_list.append(time.time())
					w3 = (wavelength_arr>wl/3-20)&(wavelength_arr>wl/3+20)
					integr_intens_THG = intens[w3].sum()
					w2 = (wavelength_arr>wl/2-20)&(wavelength_arr>wl/2+20)
					integr_intens_SHG = intens[w2].sum()
					df['wavelength'] = wavelength_arr
					df['intens'] = intens

					self.live_x = np.hstack((self.live_x, wl))
					self.live_pmtA = np.hstack((self.live_pmtA, integr_intens_SHG))

					self.line_pmtA.setData(x=self.live_x,y=self.live_pmtA)
					self.live_pmtB = np.hstack((self.live_pmtB, integr_intens_THG))

					self.line_pmtB.setData(x=self.live_x,y=self.live_pmtB)
					app.processEvents()
					if not self.scan3DisAlive:
						store.put("forceEnd_"+str(wl), df)
						return
						break
					store.put("spectra_"+str(wl), df)
					store.put("time_"+str(wl), pd.DataFrame(time_list))
					if self.ui.meas_laser_spectra_track.isChecked():
						store.put("trackA_"+str(wl), panelA)
						store.put("trackB_"+str(wl), panelB)
						store.put("center_"+str(wl), pd.DataFrame(spectra_center))
						store.put("bg_center_"+str(wl), pd.DataFrame(bg_center))




		else:
			self.scan3DisAlive = False
	def nMeasLaserSpectra_filePath_find(self):
		fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', self.ui.nMeasLaserSpectra_filePath.text())
		if type(fname)==tuple:
			fname = fname[0]
		try:
			self.ui.nMeasLaserSpectra_filePath.setText(fname)
		except:
			traceback.print_exc()

	def nMeasLaserSpectra_setPoint(self,state):
		if state:
			self.nMeasLaserSpectra_probe_currentRow = self.ui.nMeasLaserSpectra_probe.currentRow()

	def nMeasLaserSpectra_go(self,state):
		if state:
			#self.nMeasLaserSpectra_go_()

			self.nMeasLaserSpectra_thread = threading.Thread(target=self.nMeasLaserSpectra_go_)
			self.nMeasLaserSpectra_thread.daemon = True
			self.nMeasLaserSpectra_thread.start()
		else:
			self.scan3DisAlive = False
	def nMeasLaserSpectra_go_(self,state=True):
		if state:
			self.ui.actionStop.setChecked(False)
			self.live_x = np.array([])
			self.live_y = np.array([])
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.nMeasLaserSpectra_testPolar_counter = 0
			self.nMeasLaserSpectra_rescan2D_counter = 0

			start_wavelength = self.ui.nMeasLaserSpectra_start.value()
			end_wavelength = self.ui.nMeasLaserSpectra_end.value()
			step_wavelength = self.ui.nMeasLaserSpectra_step.value()
			if start_wavelength > end_wavelength:
				step_wavelength = -step_wavelength
			self.scan3DisAlive = True
			wl_range = np.arange(start_wavelength,end_wavelength+step_wavelength,step_wavelength)
			self.live_x = np.array([])
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			centerA, centerB, panelA, panelB = [None]*4
			with pd.HDFStore(self.ui.nMeasLaserSpectra_filePath.text()+str(round(time.time()))+'.h5') as store:
				store.keys()
				self.live_x = np.array([])
				self.live_integr_spectra = np.array([])

				for wl in wl_range:
					if self.ui.actionPause.isChecked():
						while self.ui.actionPause.isChecked():
							#app.processEvents()
							time.sleep(0.02)
					if self.ui.actionStop.isChecked(): break
					time_list = []
					time_list.append(time.time())
					self.laserSetWavelength_(status=1,wavelength=wl)

					#time.sleep(3)
					for i in range(100):
						time.sleep(0.03)
						#app.processEvents()
					t0 = time.time()

					while not wl == float(self.ui.laserWavelength.text()) and time.time()-t0<10 and self.scan3DisAlive:
						time.sleep(0.1)
						print('wait:Laser')
						#app.processEvents()
						if not self.ui.nMeasLaserSpectra_go.isChecked():
							break

						if self.ui.actionStop.isChecked(): break

					time_list.append(time.time())


					bg_center = np.array([ float(self.ui.nMeasLaserSpectra_probe.item(0,i).text()) for i in range(3)])


					spectra_center = np.array([ float(self.ui.nMeasLaserSpectra_probe.item(1,i).text()) for i in range(3)])

					NP_centers = np.array([[ float(self.ui.nMeasLaserSpectra_probe.item(j,i).text()) for i in range(3)] for j in range(1,self.ui.nMeasLaserSpectra_probe.rowCount())])

					z_start = self.ui.nMeasLaserSpectra_Z_start.value()
					z_end = self.ui.nMeasLaserSpectra_Z_end.value()
					z_step = self.ui.nMeasLaserSpectra_Z_step.value()

					Range_z = np.arange(z_start,z_end,z_step)
					data_ZA = np.zeros(len(Range_z))
					data_ZB = np.zeros(len(Range_z))

					if self.ui.nMeasLaserSpectra_track.isChecked():


						if self.ui.nMeasLaserSpectra_MOCO_PMT.isChecked():
							self.ui.mocoSection.setCurrentIndex(3)
						else:
							source = self.ui.readoutSources.currentText()
							if source == 'AndorCamera':
								pass
							elif source == 'Picoscope':
								self.ui.shamrockPort.setCurrentIndex(1)
							self.shamrockSetWavelength((wl/2+wl/3)/2)

						if self.ui.nMeasLaserSpectra_Z_interface_optim.isChecked():
							if self.ui.nMeasLaserSpectra_MOCO_PMT.isChecked():
								print(self.piStage.MOV(bg_center,axis=b'1 2 3',waitUntilReady=True))

								for zi,z in enumerate(Range_z):
									self.piStage.MOV([z],axis=b'3',waitUntilReady=True)
									real_position = self.piStage.qPOS()
									print(real_position)
									self.setUiPiPos(real_position)
									pmt_valA, pmt_valB,pmt_valC = self.getABData()
									data_ZB[zi] = pmt_valB
									data_ZA[zi] = pmt_valA
									self.line_pmtB.setData(x=Range_z,y=data_ZB)
									self.line_pmtA.setData(x=Range_z,y=data_ZA)

									#app.processEvents()
									if self.ui.actionPause.isChecked():
										while self.ui.actionPause.isChecked():
											time.sleep(0.02)
											#app.processEvents()
									if self.ui.actionStop.isChecked(): break
								dz = medfilt(data_ZB,9)
								interf_z = Range_z[dz==dz.max()][0]
								if self.ui.nMeasLaserSpectra_Z_interface_fit.isChecked():

									gmodel = Model(gaussian)
									result = gmodel.fit(dz, x=Range_z, amp=dz.max(), cen=interf_z, wid=2,bg=dz.min())
									print(result.params)
									store.put("scan_Z_"+str(wl), pd.DataFrame(np.vstack([data_ZB,result.best_fit]).T,index=Range_z,columns=['scan_Z','fit']))
									print(interf_z,result.params['cen'].value)
									interf_z = result.params['cen'].value
									self.line_pmtA.setData(x=Range_z,y=result.best_fit)

								else:
									store.put("scan_Z_"+str(wl), pd.DataFrame(data_ZB,index=Range_z,columns=['scan_Z']))
							else:
								self.andorCamera_prevExposure = self.ui.andorCameraExposure.value()
								self.ui.andorCameraExposure.setValue(self.ui.nMeasLaserSpectra_scanSpectraExp.value())
								#app.processEvents()

								print(self.piStage.MOV(bg_center,axis=b'1 2 3',waitUntilReady=True))
								self.andorCameraGetBaseline()
								current_row = np.where([self.ui.nMeasLaserSpectra_probe.item(i,3) for i in range(10)])[0][0]-1#self.ui.nMeasLaserSpectra_probe.currentRow()-1
								print(self.piStage.MOV(NP_centers[current_row],axis=b'1 2 3',waitUntilReady=True))
								N = self.ui.optim1step_n.value()
								pos,v = self.center3Doptim(center=NP_centers[current_row],N=N)
								interf_z = pos[2]
								self.ui.andorCameraExposure.setValue(self.andorCamera_prevExposure)
								#app.processEvents()
								delta = pos - NP_centers[current_row]
								for i in range(len(NP_centers)):
									for j in range(3):
										self.ui.nMeasLaserSpectra_probe.item(i+1,j).setText(str(NP_centers[i,j]+delta[j]))
								self.ui.nMeasLaserSpectra_probe.item(0,0).setText(str(bg_center[0]+delta[0]))
								self.ui.nMeasLaserSpectra_probe.item(0,1).setText(str(bg_center[1]+delta[1]))
								self.ui.nMeasLaserSpectra_probe.item(0,2).setText(str(bg_center[2]+delta[2]))
									#self.ui.nMeasLaserSpectra_probe.item(i+1,2).setText(str(np_scan_Z))

							self.ui.nMeasLaserSpectra_Z_interface.setValue(interf_z)
						else:
							interf_z = self.ui.nMeasLaserSpectra_Z_interface.value()
						z_offset = self.ui.nMeasLaserSpectra_Z_offset.value()

						np_scan_Z = interf_z + z_offset
						print(self.piStage.MOV([np_scan_Z],axis=b'3',waitUntilReady=True))

						if self.ui.nMeasLaserSpectra_rescan2D.isChecked():
							if self.nMeasLaserSpectra_rescan2D_counter>=self.ui.nMeasLaserSpectra_rescan2D_freq.value():
								self.nMeasLaserSpectra_rescan2D_counter = 0
								self.andorCamera_prevExposure = self.ui.andorCameraExposure.value()
								self.ui.andorCameraExposure.setValue(self.ui.nMeasLaserSpectra_scanSpectraExp.value())

								centerA, centerB, panelA, panelB = self.center_optim(z_start=np_scan_Z, z_end=np_scan_Z+0.1, z_step=0.2,update=True)
								NP_centers = self.scan3D_peak_find()
								'''
								for i in range(len(NP_centers)):
									for j in range(2):
										self.ui.nMeasLaserSpectra_probe.item(i+1,j).setText(str(NP_centers[i,j]))
									self.ui.nMeasLaserSpectra_probe.item(i+1,2).setText(str(np_scan_Z))
								'''
								self.ui.andorCameraExposure.setValue(self.andorCamera_prevExposure)
								store.put("scanA_"+str(wl), panelA)
								store.put("scanB_"+str(wl), panelB)
							self.nMeasLaserSpectra_rescan2D_counter += 1

						if self.ui.nMeasLaserSpectra_FloatingWindow.isChecked():
							center_tmp = NP_centers.mean(axis=0)
							if len(center_tmp)>0:
								pos = np.array(self.rectROI.pos())
								size = self.rectROI.size()
								prev_center = np.array([size[0]/2+pos[0],size[1]/2+pos[1]])
								shift = center_tmp - prev_center
								self.ui.nMeasLaserSpectra_probe.item(0,0).setText(str(bg_center[0]+shift[0]))
								self.ui.nMeasLaserSpectra_probe.item(0,1).setText(str(bg_center[1]+shift[1]))
								self.rectROI.setPos(pos+shift)

						if self.ui.scan3D_byMask.isChecked():
							self.generate2Dmask(view=False)


					if not self.ui.nMeasLaserSpectra_go.isChecked():
						break
					time_list.append(time.time())

					#self.ui.nMeasLaserSpectra_probe.item(0,2).setText(str(np_scan_Z))
					bg_center = np.array([ float(self.ui.nMeasLaserSpectra_probe.item(0,i).text()) for i in range(3)])

					self.ui.mocoSection.setCurrentIndex(2)

					self.ui.shamrockPort.setCurrentIndex(0)
					self.shamrockSetWavelength((wl/2+wl/3)/2)

					interf_z = self.ui.nMeasLaserSpectra_Z_interface.value()
					self.piStage.MOV(bg_center,b'1 2 3',waitUntilReady=True)
					self.piStage.MOV([interf_z],b'3',waitUntilReady=True)
					self.andorCameraGetBaseline()
					df = pd.DataFrame(self.andorCCDBaseline, index=self.andorCCD_wavelength,columns=['Z_interface_'+str(interf_z)])
					store.put("spectra_"+str(wl)+'_interface', df)


					self.piStage.MOV(bg_center,b'1 2 3',waitUntilReady=True)
					pos = self.piStage.qPOS()

					self.setUiPiPos(pos=pos)
					self.andorCameraGetBaseline()

					exposure = self.ui.andorCameraExposure.value()
					intens = np.array([0]*len(self.andorCCD_wavelength))




					for index,NP_c in enumerate(NP_centers):
						if np.isnan(NP_c).any(): break
						print(NP_c,index)
						self.piStage.MOV(NP_c,b'1 2 3',waitUntilReady=True)
						pos = self.piStage.qPOS()
						pos = self.piStage.qPOS()
						self.setUiPiPos(pos=pos)

						if not self.ui.nMeasLaserSpectra_go.isChecked():
							break
						if self.ui.actionPause.isChecked():
							while self.ui.actionPause.isChecked():
								#app.processEvents()
								time.sleep(0.02)
						if self.ui.actionStop.isChecked(): break
						#self.andorCameraGetBaseline()

						#integr_intens,intens,wavelength_arr = self.andorCameraGetData(state=1,integr_range=[wl/3-20,wl/3+20])


						#start_Z = self.ui.confParam_scan_start.value()
						#end_Z = self.ui.confParam_scan_end.value()
						#step_Z = self.ui.confParam_scan_step.value()
						#Range_Z = np.arange(start_Z,end_Z,step_Z)

						df = pd.DataFrame(self.andorCCDBaseline,columns=['baseline'])
						df['wavelength'] = self.andorCCD_wavelength
						#for z in Range_Z:
						#	print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
						#	real_position = self.piStage.qPOS()
						#	z_real = real_position[2]

						if self.ui.nMeasLaserSpectra_testPolar.isChecked() and index == 0 and \
							self.nMeasLaserSpectra_testPolar_counter >= self.ui.nMeasLaserSpectra_testPolarFreq.value():
							self.nMeasLaserSpectra_testPolar_counter = 0
							def HWP_move_function(pos):
								if not self.ui.HWP_stepper_Connect.isChecked():
									self.ui.HWP_stepper_Connect.setChecked(True)
								self.HWP_stepper_moveTo(float(pos),wait=True)
								pos = self.HWP_stepper_getAngle()
								#self.ui.HWP_stepper_angle.setText(str(round(pos,6)))
							start_angle = self.ui.nMeasLaserSpectra_testPolar_start.value()
							end_angle = self.ui.nMeasLaserSpectra_testPolar_end.value()
							step_angle = self.ui.nMeasLaserSpectra_testPolar_step.value()
							HWP_range = np.arange(start_angle,end_angle+step_angle,step_angle)

							for ang in HWP_range:

								if self.ui.actionPause.isChecked():
									while self.ui.actionPause.isChecked():
										#app.processEvents()
										time.sleep()
								if self.ui.actionStop.isChecked(): break
								HWP_move_function(ang)
								print(ang)
								time_list.append(time.time())
								intens,wavelength_arr = self.andorCameraGetData(state=1,line_index=index+1)
								time_list.append(time.time())
								w3 = abs(wavelength_arr-wl/3)<10
								integr_intens_THG = intens[w3].sum()
								w2 = abs(wavelength_arr-wl/2)<10
								integr_intens_SHG = intens[w2].sum()
								df['intens_'+str(ang)] = intens

								self.live_x = np.hstack((self.live_x, wl))
								self.live_y = np.hstack((self.live_y, ang))
								self.live_pmtA = np.hstack((self.live_pmtA, integr_intens_SHG))
								self.live_pmtB = np.hstack((self.live_pmtB, integr_intens_THG))

								self.line_pmtA.setData(x=self.live_x,y=self.live_pmtA)
								self.line_pmtB.setData(x=self.live_x,y=self.live_pmtB)
								self.line_pico_ChA.setData(x=self.live_y,y=self.live_pmtA)
								self.line_pico_ChB.setData(x=self.live_y,y=self.live_pmtB)
								#app.processEvents()

							HWP_move_function(start_angle)
							#app.processEvents()

						else:
							time_list.append(time.time())
							intens,wavelength_arr = self.andorCameraGetData(state=1,line_index=index+1)
							time_list.append(time.time())
							w3 = abs(wavelength_arr-wl/3)<10
							integr_intens_THG = intens[w3].sum()
							w2 = abs(wavelength_arr-wl/2)<10
							integr_intens_SHG = intens[w2].sum()
							df['intens'] = intens

							self.live_x = np.hstack((self.live_x, wl))
							ang = self.HWP_stepper_getAngle()
							self.live_y = np.hstack((self.live_y, ang))
							self.live_pmtA = np.hstack((self.live_pmtA, integr_intens_SHG))
							self.live_pmtB = np.hstack((self.live_pmtB, integr_intens_THG))

							self.line_pmtA.setData(x=self.live_x,y=self.live_pmtA)
							self.line_pmtB.setData(x=self.live_x,y=self.live_pmtB)



						#app.processEvents()
						if not self.ui.nMeasLaserSpectra_go.isChecked():
							store.put("forceEnd_"+str(wl), df)
							return
							break
						spectra_name = "spectra_"+str(wl)+'_NP'+str(index)
						store.put(spectra_name, df)
						store.get_storer(spectra_name).attrs.centerXYZ = pos
						store.get_storer(spectra_name).attrs.exposure = self.ui.andorCameraExposure.value()
						store.get_storer(spectra_name).attrs.laserBultInPower = self.ui.laserPower_internal.value()
						store.get_storer(spectra_name).attrs.HWP_power = float(self.ui.HWP_angle.text())
						store.get_storer(spectra_name).attrs.HWP_stepper = float(self.ui.HWP_stepper_angle.text())
						store.get_storer(spectra_name).attrs.endTime = time.time()
						store.get_storer(spectra_name).attrs.filter = self.ui.mocoSection.currentText()


						store.put("time_"+str(wl), pd.DataFrame(time_list))
						#if self.ui.nMeasLaserSpectra_track.isChecked():
							#if self.ui.nMeasLaserSpectra_rescan2D.isChecked():


						store.put("NPsCenters_"+str(wl), pd.DataFrame(NP_centers))
						store.put("bgCenter_"+str(wl), pd.DataFrame(bg_center))
						exposure = self.ui.andorCameraExposure.value()
					if intens.max()>40000 and self.ui.andorCameraExposureAdaptive.isChecked():
						self.ui.andorCameraExposure.setValue(exposure*0.8)

					if self.ui.actionPause.isChecked():
						while self.ui.actionPause.isChecked():
							#app.processEvents()
							time.sleep(0.02)
					if self.ui.actionStop.isChecked(): break

					self.nMeasLaserSpectra_testPolar_counter += 1

		else:
			self.scan3DisAlive = False

	def meas_laser_spectra_probe_update(self):
		#self.spectra_bg_probe =  pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(100, 0, 255))
		x=float(self.ui.meas_laser_spectra_probe.item(0,0).text())
		y=float(self.ui.meas_laser_spectra_probe.item(0,1).text())
		z=float(self.ui.meas_laser_spectra_probe.item(0,1).text())

		self.spectra_bg_probe.setData(x=[x],y=[y])
		self.spectra_bg_probe1.setData(x=[x],y=[y])

		#self.spectra_signal_probe = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 10))
		x=float(self.ui.meas_laser_spectra_probe.item(1,0).text())
		y=float(self.ui.meas_laser_spectra_probe.item(1,1).text())
		z=float(self.ui.meas_laser_spectra_probe.item(1,1).text())

		self.spectra_signal_probe.setData(x=[x],y=[y])
		self.spectra_signal_probe1.setData(x=[x],y=[y])
		#self.img.addItem(self.spectra_bg_probe)
		#self.img.addItem(self.spectra_signal_probe)


	def center_optim(self, x_start=None, x_end=None, x_step=None, y_start=None, y_end=None, y_step=None, z_start=None, z_end=None,z_step=None, update=False ):
		#self.scan3DisAlive = True

		if x_start is None:
			x_start = float(self.ui.scan3D_config.item(2,1).text())
		if x_end is None:
			x_end = float(self.ui.scan3D_config.item(2,2).text())
		if x_step is None:
			x_step = float(self.ui.scan3D_config.item(2,3).text())

		if y_start is None:
			y_start = float(self.ui.scan3D_config.item(1,1).text())
		if y_end is None:
			y_end = float(self.ui.scan3D_config.item(1,2).text())
		if y_step is None:
			y_step = float(self.ui.scan3D_config.item(1,3).text())

		if z_start is None:
			z_start = float(self.ui.scan3D_config.item(0,1).text())
		if z_end is None:
			z_end = float(self.ui.scan3D_config.item(0,2).text())
		if z_step is None:
			z_step = float(self.ui.scan3D_config.item(0,3).text())

		Range_z = np.arange(z_start,z_end,z_step)
		Range_zi = np.arange(len(Range_z))

		y_start = float(self.ui.scan3D_config.item(1,1).text())
		y_end = float(self.ui.scan3D_config.item(1,2).text())
		y_step = float(self.ui.scan3D_config.item(1,3).text())

		Range_y = np.arange(y_start,y_end,y_step)
		Range_yi = np.arange(len(Range_y))

		x_start = float(self.ui.scan3D_config.item(2,1).text())
		x_end = float(self.ui.scan3D_config.item(2,2).text())
		x_step = float(self.ui.scan3D_config.item(2,3).text())

		Range_x = np.arange(x_start,x_end,x_step)
		Range_xi = np.arange(len(Range_x))
		data_pmtA = np.zeros((len(Range_z),len(Range_xi),len(Range_yi)))
		data_pmtB = np.zeros((len(Range_z),len(Range_xi),len(Range_yi)))
		layerIndex = 0

		if self.ui.meas_laser_spectra_track_fast.isChecked():
			data_pmtA, data_pmtB,(Range_x,Range_y,Range_z) = self.fast3DScan2()
		else:


			for z,zi in zip(Range_z,Range_zi):
				if not self.scan3DisAlive: break
				self.piStage.MOV(z,axis=3,waitUntilReady=True)
				yi = 0
				xi = 0
				y = 0
				x = 0
				for target in generate2Dtoolpath(Range_x,Range_y,self.dataMask,self.ui.scan3D_maskStep.value()):
					#print('target',target)
					if not self.scan3DisAlive: break
					if target[0] == b'2':
						axis, yi, y = target
						r = self.piStage.MOV([y],axis=axis,waitUntilReady=True)
						if not r: break
					elif target[0] == b'1':
						axis, xi, x = target
						start=time.time()
						#print('Start',start)
						if not self.scan3DisAlive: break
						r = self.piStage.MOV([x],axis=axis,waitUntilReady=True)
						if not r: break

						#real_position0 = self.piStage.qPOS()
						pmt_valA, pmt_valB,_ = self.getABData()
						data_pmtA[zi,xi,yi] = pmt_valA
						data_pmtB[zi,xi,yi] = pmt_valB
						print(xi,yi,zi)
						#app.processEvents()
						self.setUiPiPos([x,y,z])
						if self.ui.actionPause.isChecked():
							while self.ui.actionPause.isChecked():
								#app.processEvents()
								time.sleep(0.02)
						#if update:
							#self.img.update()
							#self.img1.update()
						if self.ui.actionStop.isChecked(): break
						#real_position = self.piStage.qPOS()
						#########################################
						#if self.ui.andorCameraConnect.isChecked():
						#	pmt_valA,dd,wl = self.andorCameraGetData(1)

						#print(real_position0,real_position)
						#################################################
						#x_real = np.mean([real_position0[0], real_position[0]])
						#y_real = np.mean([real_position0[1], real_position[1]])

		self.img.setImage(data_pmtA,pos=(Range_x.min(),Range_y.min()),
		scale=(x_step,y_step),xvals=Range_z)
		self.img1.setImage(data_pmtB,pos=(Range_x.min(),Range_y.min()),
		scale=(x_step,y_step),xvals=Range_z)
		self.data2D_A = np.array(data_pmtA[zi])
		self.data2D_B = np.array(data_pmtB[zi])

		self.data2D_Range_x = Range_x
		self.data2D_Range_y = Range_y

		w1 = (data_pmtA**2)>(data_pmtA**2).mean()
		w2 = (data_pmtB**2)>(data_pmtB**2).mean()
		centerA = np.array(center_of_mass(data_pmtA*w1))[[1,2,0]]
		centerB = np.array(center_of_mass(data_pmtB*w2))[[1,2,0]]

		centerA = centerA*np.array([x_step,y_step,z_step])+ \
			np.array([Range_x.min(),Range_y.min(),Range_z.min()])
		centerB = centerB*np.array([x_step,y_step,z_step]) + \
			np.array([Range_x.min(),Range_y.min(),Range_z.min()])

		layerIndex = np.abs(Range_z - centerA[2]).argmin()
		layerIndex1 = np.abs(Range_z - centerB[2]).argmin()

		self.img.setCurrentIndex(layerIndex)
		self.img1.setCurrentIndex(layerIndex1)

		panelA = pd.Panel(data_pmtA,items=Range_z,
			major_axis=Range_x,
			minor_axis=Range_y)
		panelB = pd.Panel(data_pmtA,items=Range_z,
			major_axis=Range_x,
			minor_axis=Range_y)

		return centerA, centerB, panelA, panelB

	def center3Doptim(self,center=None,N=1):
		if center is None:
			center = self.piStage.qPOS()
			center = self.piStage.qPOS()
		else:
			self.piStage.MOV(center,axis=b'1 2 3', waitUntilReady=True)
			#center = self.piStage.qPOS()
		data_pos_res = []
		ch = self.ui.nMeasLaserSpectra_track_channel.currentIndex()
		col = [3,4,5][ch]

		x_step = float(self.ui.scan3D_config.item(2,3).text())
		y_step = float(self.ui.scan3D_config.item(1,3).text())
		z_step = float(self.ui.scan3D_config.item(0,3).text())

		for n in range(N):
			pmt_valA, pmt_valB, pmt_valC = self.getABData()
			x_step = x_step/1.6**n
			y_step = y_step/1.6**n
			z_step = z_step/1.6**n
			pos4test = []
			if self.ui.optim1step_zCompensation.isChecked():
				z_positions = [zCompensation(self.ui.laserWavelength.value(),self.ui.optim1step_zCompensationShift.value())]
			else:
				z_positions = [center[2]-z_step,center[2], center[2]+z_step]
			x_positions = [center[0]-x_step,center[0],center[0]+x_step]
			y_positions = [center[1]-y_step,center[1],center[1]+y_step]
			pos4test = np.array(np.meshgrid(x_positions, y_positions, z_positions)).T.reshape(-1,3)

			#f=[sum(k==center)!=3 for k in pos4test]
			#pos4test = pos4test[f]
			print(pos4test)
			data_pos = [np.hstack([center,pmt_valA, pmt_valB, pmt_valC])]
			for p in pos4test:
				self.piStage.MOV(p,axis=b'1 2 3', waitUntilReady=True)
				#self.piStage.MOV(p,axis=b'1 2 3', waitUntilReady=True)
				p = self.piStage.qPOS()
				p = self.piStage.qPOS()
				self.setUiPiPos(p)
				pmt_valA, pmt_valB, pmt_valC = self.getABData()
				data_pos.append(np.hstack([p,pmt_valA, pmt_valB, pmt_valC]))
				#app.processEvents()
				if self.ui.actionPause.isChecked():
					while self.ui.actionPause.isChecked():
						#app.processEvents()
						time.sleep(0.02)
				if self.ui.actionStop.isChecked(): break
			data_pos = np.array(data_pos)


			print(ch,data_pos)
			optim_pos = data_pos[data_pos[:,col]==data_pos[:,col].max()][0]
			print(optim_pos)
			pos = optim_pos[:3]
			self.piStage.MOV(pos,axis=b'1 2 3', waitUntilReady=True)
			#self.piStage.MOV(pos,axis=b'1 2 3', waitUntilReady=True)
			p = self.piStage.qPOS()
			p = self.piStage.qPOS()
			center_prev = center.copy()
			center1 = self.piStage.qPOS()
			center_ = center1+(center1-center_prev)

			self.setUiPiPos(center1)
			pmt_valA, pmt_valB, pmt_valC = self.getABData()
			pmt_valA1, pmt_valB1, pmt_valC1 = self.getABData()
			pmt_valA = (pmt_valA+pmt_valA1)/2
			pmt_valB = (pmt_valB+pmt_valB1)/2
			pmt_valC = (pmt_valC+pmt_valC1)/2
			data_pos_res.append(np.hstack([center1,pmt_valA, pmt_valB, pmt_valC]))
			center = center_

		data_pos_res = np.array(data_pos_res)
		print(ch,data_pos_res)
		optim_pos = data_pos_res[data_pos_res[:,col]==data_pos_res[:,col].max()][0]
		print(optim_pos)
		pos = optim_pos[:3]
		self.piStage.MOV(pos,axis=b'1 2 3', waitUntilReady=True)
		self.piStage.MOV(pos,axis=b'1 2 3', waitUntilReady=True)
		p = self.piStage.qPOS()
		p = self.piStage.qPOS()
		center = self.piStage.qPOS()
		self.setUiPiPos(center)
		pmt_valA, pmt_valB, pmt_valC = self.getABData()

		zCompens = zCompensation(self.ui.laserWavelength.value(),center[2])
		self.ui.optim1step_zCompensationShift.setValue(zCompens)

		return center, (pmt_valA, pmt_valB, pmt_valC)

	def optim1step(self):
		self.ui.actionStop.setChecked(False)
		N = self.ui.optim1step_n.value()
		self.optim1step_thread = threading.Thread(target=self.center3Doptim,kwargs={'N':N})
		self.optim1step_thread.daemon = True
		self.optim1step_thread.start()

	def start3DScan(self, state):
		print(state)
		if state:
			try:
				self.live_pmtA = np.array([])
				self.live_pmtB = np.array([])
				self.live_x = np.array([])
				self.live_y = np.array([])

				self.scan3DisAlive = True
				self.scan3D_thread = threading.Thread(target=self.scan3D)
				self.scan3D_thread.daemon = True
				self.scan3D_thread.start()

			except:
				traceback.print_exc()
		else:
			self.live_pmtA = []
			self.live_pmtB = []
			self.live_x = []
			self.live_y = []

			self.scan3DisAlive = False

	def getABData(self, update=False):
		source = self.ui.readoutSources.currentText()
		if source == 'Picoscope':
			A,B = self.readPico()
			C = np.nan
		elif source == 'AndorCamera':
			intens,wavelength_arr = self.andorCameraGetData(state=1)
			Ex_wl = self.ui.laserWavelength.value()
			w3 = (wavelength_arr>Ex_wl/3-10)&(wavelength_arr<Ex_wl/3+10)
			w2 = (wavelength_arr>Ex_wl/2-10)&(wavelength_arr<Ex_wl/2+10)
			A = intens[w2].sum()
			B = intens[w3].sum()
			C = intens.sum()
		else:
			A,B,C = [np.nan]*3
		#if update:
		#	self.pw_preview.update()
		#	self.pw1.update()
		#	self.pw_spectra.update()
		#	self.pw.update()
		return A,B,C


	def scan3D(self):
		self.ui.actionStop.setChecked(False)
		self.calibrTimer.stop()
		data = []
		rows = self.ui.scan3D_config.rowCount()
		columns = self.ui.scan3D_config.columnCount()
		import csv
		for r in range(rows):
			d = []
			for c in range(columns):
				d.append(self.ui.scan3D_config.item(r,c).text())
			data.append(d)
		with open("scanArea.csv", "w+") as myCsv:
			csvWriter = csv.writer(myCsv, delimiter='\t')
			csvWriter.writerows(data)

		wait = self.ui.Pi_wait.isChecked()
		path = self.ui.scan3D_path.text()
		if "_" in path:
			path = "".join(path.split("_")[:-1])#+"_"+str(round(time.time()))
		else:
			path = path #+ "_"+str(round(time.time()))

		self.ui.scan3D_path.setText(path)
		spectra_range = [self.ui.usbSpectr_save_from.value(),self.ui.usbSpectr_save_to.value()]
		try:
			z_start = float(self.ui.scan3D_config.item(0,1).text())
			z_end = float(self.ui.scan3D_config.item(0,2).text())
			z_step = float(self.ui.scan3D_config.item(0,3).text())
			if z_step == 0:
				Range_z = np.array([z_start]*100)
			else:
				Range_z = np.arange(z_start,z_end,z_step)
			Range_zi = np.arange(len(Range_z))
			y_start = float(self.ui.scan3D_config.item(1,1).text())
			y_end = float(self.ui.scan3D_config.item(1,2).text())
			y_step = float(self.ui.scan3D_config.item(1,3).text())

			Range_y = np.arange(y_start,y_end,y_step)
			Range_yi = np.arange(len(Range_y))


			x_start = float(self.ui.scan3D_config.item(2,1).text())
			x_end = float(self.ui.scan3D_config.item(2,2).text())
			x_step = float(self.ui.scan3D_config.item(2,3).text())

			Range_x = np.arange(x_start,x_end,x_step)
			Range_xi = np.arange(len(Range_x))

			data_pmtA = np.zeros((len(Range_z),len(Range_x),len(Range_y)))
			data_pmtB = np.zeros((len(Range_z),len(Range_x),len(Range_y)))
			print(data_pmtA.shape)
			layerIndex = 0
			for z,zi in zip(Range_z,Range_zi):
				if not self.scan3DisAlive: break
				print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
				#time.sleep(0.1)

				fname = path+"Z"+str(z)+"_"+str(round(time.time()))+'.txt'
				with open(fname,'a') as f:
					f.write("#X\tY\tZ\tpmtA_signal\tpmtB_signal\ttime\n")



				#data_spectra = np.zeros((len(Range_yi),len(Range_xi)))


				forward = True
				xi = 0
				yi = 0
				mask = self.dataMask
				if not self.ui.scan3D_byMask.isChecked():
					mask = None
				for target in generate2Dtoolpath(Range_x,Range_y,mask,self.ui.scan3D_maskStep.value()):
					#print('target',target)
					if not self.scan3DisAlive: break
					axis, index_pos, t_pos = target
					if axis == b'2':
						yi = index_pos
						r = self.piStage.MOV([t_pos],axis=axis,waitUntilReady=True)
						self.live_pmtA = np.array([])
						self.live_pmtB = np.array([])
						self.live_x = np.array([])
						self.live_y = np.array([])


					elif axis == b'1':
						if not r: break


						xi = index_pos
						start=time.time()
						#print('Start',start)
						if not self.scan3DisAlive: break
						r = self.piStage.MOV([t_pos],axis=axis,waitUntilReady=wait)
						if not r: break

						real_position0 = self.piStage.qPOS()
						pmt_valA, pmt_valB, _ = self.getABData()#self.readPico()
						real_position = self.piStage.qPOS()
						#########################################
						#if self.ui.andorCameraConnect.isChecked():
							#pmt_valA,dd,wl = self.andorCameraGetData(1)

						print(real_position0,real_position)
						#################################################
						self.live_pmtA = np.hstack((self.live_pmtA, pmt_valA))
						self.live_pmtB = np.hstack((self.live_pmtB, pmt_valB))
						x_real = np.mean([real_position0[0], real_position[0]])
						y_real = np.mean([real_position0[1], real_position[1]])

						self.live_x = np.hstack((self.live_x,x_real))
						self.live_y = np.hstack((self.live_y,y_real))

						#spectra = np.zeros(3648)
						dataSet = real_position +[pmt_valA, pmt_valB, time.time()]# + spectra[spectra_range[0]:spectra_range[1]]
						#print(dataSet[-1])
						with open(fname,'a') as f:
							f.write("\t".join([str(i) for i in dataSet])+"\n")
						#print(time.time()-start)
						#s_from = self.ui.usbSpectr_from.value()
						#s_to = self.ui.usbSpectr_to.value()
						#print(data_spectra.shape,yi,xi)
						if wait:
							xi_ = xi
							yi_ = yi
						else:
							xi_ = int(round((self.live_x[-1]-Range_x[0])*len(Range_x)/(Range_x.max()-Range_x.min())))
							yi_ = int(round((self.live_y[-1]-Range_y[0])*len(Range_y)/(Range_y.max()-Range_y.min())))
							#data_spectra[yi_,xi_] = np.sum(spectra[s_from:s_to])
							if xi_>= len(Range_x):
								xi_ = len(Range_x)-1
							if yi_>= len(Range_y):
								yi_ = len(Range_y)-1
						data_pmtA[zi,xi_,yi_] = pmt_valA
						data_pmtB[zi,xi_,yi_] = pmt_valB
						#print(self.live_x[-1], x, xi_,xi, yi_, yi)

						#self.live_integr_spectra.append(np.sum(spectra[s_from:s_to]))
						self.line_pmtA.setData(x=self.live_x,y=self.live_pmtA)
						self.line_pmtB.setData(x=self.live_x,y=self.live_pmtB)
						#self.line_spectra.setData(self.live_integr_spectra)

						#self.setLine(spectra)
						#print(time.time()-start)
						self.setUiPiPos(real_position)
						print("[\t%.5f\t%.5f\t%.5f\t]\t%.5f"%tuple(list(real_position)+[time.time()-start]))


						#time.sleep(0.01)
						wait = self.ui.Pi_wait.isChecked()
						if self.ui.actionPause.isChecked():
							while self.ui.actionPause.isChecked():
								time.sleep(0.02)
								#app.processEvents()
						if self.ui.actionStop.isChecked(): break
						#print(time.time()-start)
					#self.setImage(data_spectra)

					self.img.setImage(data_pmtA,pos=(Range_x.min(),Range_y.min()),
					scale=(x_step,y_step),xvals=Range_z)
					self.img1.setImage(data_pmtB,pos=(Range_x.min(),Range_y.min()),
					scale=(x_step,y_step),xvals=Range_z)
					self.img.setCurrentIndex(layerIndex)
					self.img1.setCurrentIndex(layerIndex)

					self.data2D_A = np.array(data_pmtA[zi])
					self.data2D_B = np.array(data_pmtB[zi])
					self.data2D_Range_x = Range_x
					self.data2D_Range_y = Range_y

					#print(sum(data_pmtA),sum(data_pmtB))

				#imsave(fname+"_pmtA.tif",data_pmtB.astype(np.int16))
				#imsave(fname+"_pmtB.tif",data_pmtB.astype(np.int16))
				layerIndex+=1

		except KeyboardInterrupt:

			data_pmtA = data_pmtA[data_pmtA.sum(axis=2).sum(axis=1)!=0]
			data_pmtB = data_pmtB[data_pmtB.sum(axis=2).sum(axis=1)!=0]

			imsave(fname+"_pmtA.tif",data_pmtA.astype(np.float32), imagej=True, resolution=(x_step*1e-4,y_step*1e-4,'cm'))
			imsave(fname+"_pmtB.tif",data_pmtB.astype(np.float32), imagej=True, resolution=(x_step*1e-4,y_step*1e-4,'cm'))

			print(self.spectrometer.close())
			print(self.piStage.CloseConnection())
			return

		data_pmtA = data_pmtA[data_pmtA.sum(axis=2).sum(axis=1)!=0]
		data_pmtB = data_pmtB[data_pmtB.sum(axis=2).sum(axis=1)!=0]
		w1 = data_pmtA>data_pmtA.mean()
		w2 = data_pmtB>data_pmtB.mean()
		centerA = np.array(center_of_mass(data_pmtA*w1))[[1,2,0]]
		centerB = np.array(center_of_mass(data_pmtB*w2))[[1,2,0]]
		centerA = centerA*np.array([x_step,y_step,z_step])+ \
			np.array([Range_x.min(),Range_y.min(),Range_z.min()])
		centerB = centerB*np.array([x_step,y_step,z_step]) + \
			np.array([Range_x.min(),Range_y.min(),Range_z.min()])
		print(centerA,centerB)
		#data_pmt_16 = data_pmtA/data_pmtA.max()*32768*2-32768
		#data_pmtB_16 = data_pmtB/data_pmtB.max()*32768*2-32768
		#imsave(fname+"_pmtA.tif",data_pmt_16.astype(np.int16), imagej=True)
		#imsave(fname+"_pmtB.tif",data_pmtB_16.astype(np.int16), imagej=True)
		try:
			imsave(fname+"_pmtA.tif",data_pmtA.astype(np.float32), imagej=True, resolution=(x_step*1e-4,y_step*1e-4,'cm'))
			imsave(fname+"_pmtB.tif",data_pmtB.astype(np.float32), imagej=True, resolution=(x_step*1e-4,y_step*1e-4,'cm'))
		except :
			traceback.print_exc()


		self.ui.start3DScan.setChecked(False)
		#print(self.spectrometer.close())
		#print(self.piStage.CloseConnection())
	def generate2Dmask(self, state=True, view=True,NPsCenters = []):
		try:
			y_start = float(self.ui.scan3D_config.item(1,1).text())
			y_end = float(self.ui.scan3D_config.item(1,2).text())
			y_step = float(self.ui.scan3D_config.item(1,3).text())

			x_start = float(self.ui.scan3D_config.item(2,1).text())
			x_end = float(self.ui.scan3D_config.item(2,2).text())
			x_step = float(self.ui.scan3D_config.item(2,3).text())
			xr = np.arange(x_start,x_end,x_step)
			yr = np.arange(y_start,y_end,y_step)

			if MODE == 'sim':

				self.data2D_A = np.zeros((len(xr),len(yr)))+1
				self.data2D_B = np.zeros((len(xr),len(yr)))+1
				self.data2D_A[len(xr)//3,len(yr)//4] = 100
				self.data2D_A[len(xr)//2,len(yr)//2] = 30
				self.data2D_B[2+len(xr)//3,3+len(yr)//4] = 80
				self.data2D_B[2+len(xr)//2,1+len(yr)//2] = 40
				self.data2D_A = gaussian_filter(self.data2D_A,sigma=10)
				self.data2D_B = gaussian_filter(self.data2D_B,sigma=18)
				self.data2D_Range_x = xr
				self.data2D_Range_y = yr


			ch = self.ui.nMeasLaserSpectra_track_channel.currentIndex()
			if ch==0:
				data = self.data2D_A
			elif ch==1:
				data = self.data2D_B
			else:
				data = self.data2D_A * self.data2D_B
			s = self.ui.scan3D_maskSigma.value()
			t = self.ui.scan3D_maskThreshold.value()
			if len(NPsCenters)==0:
				NPsCenters = self.NPsCenters
			if len(NPsCenters)==0:
				d = gaussian_filter(data,sigma=s)
				d_min = d[d>0].min()
				d = abs(d - d_min)
				self.dataMask = d>d.max()*t/100.

			else:
				Y,X = np.meshgrid(yr,xr)
				mask = np.zeros(data.shape)
				for c in NPsCenters:
					print(c)
					mask = mask + (((X-c[0])**2+(Y-c[1])**2)<=s**2).astype(np.int16)
				self.dataMask = mask>0


			if view:
				self.img.setImage(self.dataMask*self.data2D_A,pos=(x_start,y_start),
				scale=(x_step,y_step))
				self.img1.setImage(self.dataMask*self.data2D_B,pos=(x_start,y_start),
				scale=(x_step,y_step))
		except:
			traceback.print_exc()
			self.dataMask = None


	def start_fast3DScan(self,state):
		if state:
			self.calibrTimer.stop()
			try:
				#self.HWP.cleanUpAPT()
				self.scan3DisAlive = True
				self.fast3DScan2()
			except:
				traceback.print_exc()
				#self.pico_reader_proc.terminate()
			self.ui.fast3DScan.blockSignals(True)
			self.ui.fast3DScan.setChecked(False)
			self.ui.fast3DScan.blockSignals(False)
		else:
			self.scan3DisAlive = False
			#self.pico_control_queue.put('kill')
			#self.pico_reader_proc.terminate()
			self.ui.fast3DScan.blockSignals(True)
			self.ui.fast3DScan.setChecked(False)
			self.ui.fast3DScan.blockSignals(False)
			#self.HWP = APTMotor(83854487, HWTYPE=31)




	def fast3DScan(self):
		if self.ps:
			del self.ps

		self.pico_reader_proc = multiprocessing.Process(target=create_pico_reader,
			args=[self.pico_config,self.pico_shared_buf,self.pico_control_queue])
		self.pico_reader_proc.deamon = True
		self.pico_reader_proc.start()

		z_start = float(self.ui.scan3D_config.item(0,1).text())
		z_end = float(self.ui.scan3D_config.item(0,2).text())
		z_step = float(self.ui.scan3D_config.item(0,3).text())
		if z_step == 0:
			Range_z = np.array([z_start]*100)
		else:
			Range_z = np.arange(z_start,z_end,z_step)
		Range_zi = np.arange(len(Range_z))
		y_start = float(self.ui.scan3D_config.item(1,1).text())
		y_end = float(self.ui.scan3D_config.item(1,2).text())
		y_step = float(self.ui.scan3D_config.item(1,3).text())

		Range_y = np.arange(y_start,y_end,y_step)
		Range_yi = np.arange(len(Range_y))


		x_start = float(self.ui.scan3D_config.item(2,1).text())
		x_end = float(self.ui.scan3D_config.item(2,2).text())
		x_step = float(self.ui.scan3D_config.item(2,3).text())


		Range_x = np.arange(x_start,x_end,x_step)
		Range_xi = np.arange(len(Range_x))

		layerIndex = 0
		data = np.frombuffer(self.pico_shared_buf['data'].get_obj(), dtype='d').reshape(self.pico_shared_buf['shape'])

		time.sleep(5)
		while data.sum()==0:
			time.sleep(0.1)
		inRange = 0

		for z,zi in zip(Range_z,Range_zi):
			#if not self.scan3DisAlive: break
			print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
			#fname = path+"Z"+str(z)+"_"+str(round(time.time()))+'.txt'
			#with open(fname,'a') as f:
			#	f.write("#X\tY\tZ\tpmtA_signal\tpmtB_signal\ttime\n")
			data_pmtA = []
			data_pmtB = []

			forward = True
			for y,yi in zip(Range_y,Range_yi):
				#if not self.scan3DisAlive: break
				Range_x_tmp = Range_x
				if forward:
					Range_x_tmp = Range_x[::-1]
					Range_xi_tmp = Range_xi[::-1]
					#forward = False
				else:
					Range_x_tmp = Range_x
					Range_xi_tmp = Range_xi
					#forward = True
				r = self.piStage.MOV(y,axis=2,waitUntilReady=True)
				if not r: break
				self.live_pmtA = np.array([])
				self.live_pmtB = np.array([])
				self.live_x = np.array([])
				self.live_y = np.array([])

				self.live_integr_spectra = []
				tmp = []
				tmp1 = []
				t_tmp = []
				r_tmp = []
				#for x,xi in zip(Range_x_tmp,Range_xi_tmp):

				real_position0 = self.piStage.qPOS()
				t0 = time.time()
				r = self.piStage.MOV([Range_x.max()],axis=b'1',waitUntilReady=1)
				if MODE == 'sim':
					time.sleep(0.6)
				if not r: break
				t1 = time.time()
				real_position = self.piStage.qPOS()
				r_tmp.append(real_position0)
				r_tmp.append(real_position)
				t_tmp.append(t0)
				t_tmp.append(t1)

				w = (data[:,0]>=t0)&(data[:,0]<=t1)
				print("inRange:",sum(w),t1-t0)
				inRange +=sum(w)
				try:
					dataA = resample(data[w,1],len(Range_y))
					dataB = resample(data[w,2],len(Range_y))
				except:
					dataA = data[-len(Range_y):,1]
					dataB = data[-len(Range_y):,2]
				tmp.append(dataA)
				tmp1.append(dataB)
					#print(tmp)
				#if len(tmp)>=1 and len(tmp1)>=1:

				if forward:
					forward = False
					#data_pmtA.append(np.hstack(tmp))
					#data_pmtB.append(np.hstack(tmp1))
				#
				else:
					forward = True

				data_pmtA.append(np.hstack(tmp))
				data_pmtB.append(np.hstack(tmp1))

				r = self.piStage.MOV([Range_x.min()],axis=b'1',waitUntilReady=1)

				if not r: break
				if inRange==0: time.sleep(1)
				inRange = 0
		#print(data_pmtA)
		data_pmtA = np.array(data_pmtA).T
		data_pmtB = np.array(data_pmtB).T

		self.img.setImage(data_pmtA,pos=(Range_x.min(),Range_y.min()),
		scale=((Range_x.max()-Range_x.min())/len(data_pmtA),y_step),xvals=Range_z)
		self.img1.setImage(data_pmtB,pos=(Range_x.min(),Range_y.min()),
		scale=((Range_x.max()-Range_x.min())/len(data_pmtA),y_step),xvals=Range_z)


		#start0=time.time()
		self.pico_control_queue.put('kill')
		#self.pico_reader_proc.terminate()



		self.initPico()

	def fast3DScan1(self):
		if self.ps:
			del self.ps

		self.pico_shared_buf_shape = (10000,3)
		unshared_arr = np.zeros(self.pico_shared_buf_shape[0]*self.pico_shared_buf_shape[1])
		sa = Array('d', int(np.prod(self.pico_shared_buf_shape)))
		self.pico_shared_buf = {'data':sa, 'shape':self.pico_shared_buf_shape}
		self.pico_reader_proc = multiprocessing.Process(target=create_pico_reader,args=[self.pico_config,self.pico_shared_buf,self.pico_control_queue])
		self.pico_reader_proc.daemon = True
		self.pico_reader_proc.start()
		while self.pico_control_queue.qsize()==0:
			time.sleep(0.1)
		print(self.pico_control_queue.get())
		self.search_q = Queue()
		self.out_q = Queue()
		search_p = multiprocessing.Process(target=search_time_range,args=[self.pico_shared_buf,self.search_q,self.out_q])
		search_p.daemon = True
		search_p.start()


		z_start = float(self.ui.scan3D_config.item(0,1).text())
		z_end = float(self.ui.scan3D_config.item(0,2).text())
		z_step = float(self.ui.scan3D_config.item(0,3).text())
		if z_step == 0:
			Range_z = np.array([z_start]*100)
		else:
			Range_z = np.arange(z_start,z_end,z_step)
		Range_zi = np.arange(len(Range_z))
		y_start = float(self.ui.scan3D_config.item(1,1).text())
		y_end = float(self.ui.scan3D_config.item(1,2).text())
		y_step = float(self.ui.scan3D_config.item(1,3).text())

		Range_y = np.arange(y_start,y_end,y_step)
		Range_yi = np.arange(len(Range_y))


		x_start = float(self.ui.scan3D_config.item(2,1).text())
		x_end = float(self.ui.scan3D_config.item(2,2).text())
		x_step = float(self.ui.scan3D_config.item(2,3).text())


		Range_x = np.arange(x_start,x_end,x_step)
		Range_xi = np.arange(len(Range_x))

		layerIndex = 0
		data = np.frombuffer(self.pico_shared_buf['data'].get_obj(), dtype='d').reshape(self.pico_shared_buf['shape'])

		time.sleep(5)
		while data.sum()==0:
			time.sleep(0.1)
		inRange = 0

		for z,zi in zip(Range_z,Range_zi):
			#if not self.scan3DisAlive: break
			print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
			#fname = path+"Z"+str(z)+"_"+str(round(time.time()))+'.txt'
			#with open(fname,'a') as f:
			#	f.write("#X\tY\tZ\tpmtA_signal\tpmtB_signal\ttime\n")
			data_pmtA = []
			data_pmtB = []

			forward = True
			for y,yi in zip(Range_y,Range_yi):
				#if not self.scan3DisAlive: break
				Range_x_tmp = Range_x
				if forward:
					Range_x_tmp = Range_x[::-1]
					Range_xi_tmp = Range_xi[::-1]
					#forward = False
				else:
					Range_x_tmp = Range_x
					Range_xi_tmp = Range_xi
					#forward = True
				r = self.piStage.MOV(y,axis=2,waitUntilReady=True)
				if not r: break
				self.live_pmtA = np.array([])
				self.live_pmtB = np.array([])
				self.live_x = np.array([])
				self.live_y = np.array([])

				self.live_integr_spectra = []
				tmp = []
				tmp1 = []
				t_tmp = []
				r_tmp = []
				#for x,xi in zip(Range_x_tmp,Range_xi_tmp):

				real_position0 = self.piStage.qPOS()
				t0 = time.time()
				r = self.piStage.MOV([Range_x.max()],axis=b'1',waitUntilReady=1)
				if MODE == 'sim':
					time.sleep(0.6)
				if not r: break
				t1 = time.time()
				real_position = self.piStage.qPOS()
				r_tmp.append(real_position0)
				r_tmp.append(real_position)
				t_tmp.append(t0)
				t_tmp.append(t1)

				w = (data[:,0]>=t0)&(data[:,0]<=t1)
				print("inRange:",sum(w),t1-t0)
				inRange +=sum(w)
				try:
					dataA = resample(data[w,1],len(Range_y))
					dataB = resample(data[w,2],len(Range_y))
				except:
					dataA = data[-len(Range_y):,1]
					dataB = data[-len(Range_y):,2]
				tmp.append(dataA)
				tmp1.append(dataB)
					#print(tmp)
				#if len(tmp)>=1 and len(tmp1)>=1:

				if forward:
					forward = False
					#data_pmtA.append(np.hstack(tmp))
					#data_pmtB.append(np.hstack(tmp1))
				#
				else:
					forward = True

				data_pmtA.append(np.hstack(tmp))
				data_pmtB.append(np.hstack(tmp1))

				r = self.piStage.MOV([Range_x.min()],axis=b'1',waitUntilReady=1)

				if not r: break
				if inRange==0: time.sleep(1)
				inRange = 0
		#print(data_pmtA)
		data_pmtA = np.array(data_pmtA).T
		data_pmtB = np.array(data_pmtB).T

		self.img.setImage(data_pmtA,pos=(Range_x.min(),Range_y.min()),
		scale=((Range_x.max()-Range_x.min())/len(data_pmtA),y_step),xvals=Range_z)
		self.img1.setImage(data_pmtB,pos=(Range_x.min(),Range_y.min()),
		scale=((Range_x.max()-Range_x.min())/len(data_pmtA),y_step),xvals=Range_z)


		#start0=time.time()
		self.pico_control_queue.put('kill')
		#self.pico_reader_proc.terminate()



		self.initPico()

	def fast3DScan2(self):




		wait = self.ui.Pi_wait.isChecked()

		z_start = float(self.ui.scan3D_config.item(0,1).text())
		z_end = float(self.ui.scan3D_config.item(0,2).text())
		z_step = float(self.ui.scan3D_config.item(0,3).text())
		if z_step == 0:
			Range_z = np.array([z_start]*100)
		else:
			Range_z = np.arange(z_start,z_end,z_step)
		Range_zi = np.arange(len(Range_z))
		y_start = float(self.ui.scan3D_config.item(1,1).text())
		y_end = float(self.ui.scan3D_config.item(1,2).text())
		y_step = float(self.ui.scan3D_config.item(1,3).text())

		Range_y = np.arange(y_start,y_end,y_step)
		Range_yi = np.arange(len(Range_y))


		x_start = float(self.ui.scan3D_config.item(2,1).text())
		x_end = float(self.ui.scan3D_config.item(2,2).text())
		x_step = float(self.ui.scan3D_config.item(2,3).text())


		Range_x = np.arange(x_start,x_end,x_step)
		Range_xi = np.arange(len(Range_x))
		self.piStage.MOV([Range_x.min()],axis=b'1',waitUntilReady=1)
		vel = self.ui.Pi_Velocity.value()
		fs_table = np.loadtxt('hardware/fastScan_table.txt',skiprows=1)
		eq=interp1d(fs_table[:,1],fs_table[:,0],kind='quadratic',bounds_error=False,fill_value="extrapolate")
		calibr = eq(self.rectROI.size()[0])

		self.ui.fastScan_time_calibr.setValue(calibr)
		print('time_calibr',calibr)
		#calibr = self.ui.fastScan_time_calibr.value()
		time_for_scan = (Range_x.max()-Range_x.min())/vel*calibr
		pico_frame_time = 15e-7
		n_frames = int(time_for_scan/pico_frame_time)
		print('time_for_scan',time_for_scan)

		pico_frame_time_prev = self.ui.pico_samplingDuration.text()
		n_frames_prev = self.ui.pico_n_captures.value()

		self.ui.pico_samplingDuration.setText(str(pico_frame_time))
		self.ui.pico_n_captures.setValue(n_frames)
		self.pico_set()

		#scan_t = []
		#for i in range(10):
		#	t0 = time.time()
		#	self.piStage.MOV([100],axis=b'1',waitUntilReady=1)
		#	self.piStage.MOV([0],axis=b'1',waitUntilReady=1)
		#	t1 = time.time()
		#	scan_t.append((t1-t0)/2)
		#scan_t = np.mean(scan_t)
		#vel = 100/scan_t
		#print('scan_t',scan_t, vel,(Range_x.max()-Range_x.min())/vel)

		layerIndex = 0

		inRange = 0
		data_pmtA = []
		data_pmtB = []

		for z,zi in zip(Range_z,Range_zi):
			if not self.scan3DisAlive: break
			print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
			#fname = path+"Z"+str(z)+"_"+str(round(time.time()))+'.txt'
			#with open(fname,'a') as f:
			#	f.write("#X\tY\tZ\tpmtA_signal\tpmtB_signal\ttime\n")
			data_pmtA_ = np.array([])
			data_pmtB_ = np.array([])

			forward = True
			t0=time.time()
			for y,yi in zip(Range_y,Range_yi):
				if not self.scan3DisAlive: break
				#Range_x_tmp = Range_x
				#if forward:
				#	Range_x_tmp = Range_x[::-1]
				#	Range_xi_tmp = Range_xi[::-1]
					#forward = False
				#else:
				#	Range_x_tmp = Range_x
				#	Range_xi_tmp = Range_xi
					#forward = True
				if self.ui.fastScan_pattern.currentIndex()==0:
					r = self.piStage.MOV([x_start,y],axis=b'1 2',waitUntilReady=True)
				else:
					r = self.piStage.MOV([y],axis=b'2',waitUntilReady=True)
				if self.ui.fastScan_pattern.currentIndex()==0:
					x = x_end
				else:
					if forward:
						x = x_end
					else:
						x = x_start
				#thr = threading.Thread(target=move,args=[x])
				#thr.start()
				t0_,t1_,t_ind = self.ps.capture_prep_block_start(return_scaled_array=1)

				self.piStage.MOV([x],axis=b'1',waitUntilReady=wait)

				r = self.ps.capture_prep_block_end(t0_, t1_, return_scaled_array=1)
				#thr.join()
				dataA = r[0]['A']
				dataB = r[0]['B']
				#dataA = np.hstack(dataA)
				#dataB = np.hstack(dataB)
				dataT = r[2]
				N = int(self.pico_config['samplingDuration']*self.pico_config['pulseFreq'])
				scanA = np.array([])
				scanB = np.array([])

				#dataA_= np.hstack(dataA[:,:int(dataA.shape[1]//N*N)])
				#a=np.array(np.split(dataA_,N*dataA.shape[0]))
				a = dataA
				scanA = abs(a.max(axis=1) - a.min(axis=1))
				#dataB_= np.hstack(dataB[:,:int(dataB.shape[1]//N*N)])
				#b=np.array(np.split(dataB_,N*dataB.shape[0]))
				b = dataB
				scanB = abs(b.max(axis=1) - b.min(axis=1))
				#print(scanA.shape)
				real_position = self.piStage.qPOS()
				print(real_position)
				time_delay = int(self.ui.fastScan_time_delay.value()*len(scanA))
				time_cutoff = int((1-self.ui.fastScan_time_cutoff.value())*len(scanA))
				print(scanA.shape, time_delay, time_cutoff)
				if self.ui.fastScan_pattern.currentIndex()==0:
					scanA = scanA[time_delay:time_cutoff]
					scanB = scanB[time_delay:time_cutoff]
				else:
					if not forward:
						scanA = scanA[time_delay:time_cutoff][::-1]
						scanB = scanB[time_delay:time_cutoff][::-1]

				#if forward:
				a = np.array([i.mean() for i in np.array_split(scanA,len(Range_x))])
				if len(data_pmtA_)==0:
					data_pmtA_ = a
				else:
					data_pmtA_ =np.vstack((data_pmtA_,a))
				b = np.array([i.mean() for i in np.array_split(scanB,len(Range_x))])
				if len(data_pmtB_)==0:
					data_pmtB_ = b
				else:
					data_pmtB_ =np.vstack((data_pmtB_,b))

				forward = not forward
				#else:
				#	data_pmtA_.append([i.mean() for i in np.array_split(scanA,len(Range_x))][::-1])
				#	data_pmtB_.append([i.mean() for i in np.array_split(scanB,len(Range_x))][::-1])
				#	forward = True
				app.processEvents()
				if self.ui.actionStop.isChecked():
					self.ui.pico_samplingDuration.setText(pico_frame_time_prev)
					self.ui.pico_n_captures.setValue(n_frames_prev)
					self.ui.connect_pico.setChecked(False)
					self.ui.connect_pico.setChecked(True)
					break

			t1=time.time()
			print("dt: %.9f\n"%(t1-t0))

			data_pmtA.append(np.array(data_pmtA_).T)
			data_pmtB.append(np.array(data_pmtB_).T)
			self.data2D_A = np.array(data_pmtA_).T
			self.data2D_B = np.array(data_pmtB_).T
			self.data2D_Range_x = Range_x
			self.data2D_Range_y = Range_y
			print(x,y,z)
		data_pmtA = np.array(data_pmtA)
		data_pmtB = np.array(data_pmtB)

		self.img.setImage(data_pmtA,pos=(Range_x.min(),Range_y.min()),
		scale=(x_step,y_step),xvals=Range_z)
		self.img1.setImage(data_pmtB,pos=(Range_x.min(),Range_y.min()),
		scale=(x_step,y_step),xvals=Range_z)


		self.ui.pico_samplingDuration.setText(pico_frame_time_prev)
		self.ui.pico_n_captures.setValue(n_frames_prev)
		self.ui.connect_pico.setChecked(False)
		self.ui.connect_pico.setChecked(True)

		#self.pico_set()

		#self.pico_set()
		real_position = self.piStage.qPOS()
		self.setUiPiPos(real_position)

		return data_pmtA, data_pmtB, (Range_x, Range_y, Range_z)

	def scan3D_peak_find(self):
		num_peaks = self.ui.scan3D_num_peaks.value()
		threshold_rel = self.ui.scan3D_threshold_rel.value()
		min_distance = self.ui.Scan3D_min_distance.value()

		peaks_A = peak_local_max(gaussian_filter(self.data2D_A,sigma=2), min_distance=min_distance, threshold_rel=threshold_rel,num_peaks=num_peaks)
		peaks_B = peak_local_max(gaussian_filter(self.data2D_B,sigma=2), min_distance=min_distance, threshold_rel=threshold_rel,num_peaks=num_peaks)

		Range_y = self.data2D_Range_y

		Range_x = self.data2D_Range_x
		interf_z = self.ui.nMeasLaserSpectra_Z_interface.value()

		z_offset = self.ui.nMeasLaserSpectra_Z_offset.value()

		np_scan_Z = abs(interf_z + z_offset)


		centers = {}
		centers["A"] = np.array([Range_x[peaks_A[:,0]], Range_y[peaks_A[:,1]], [np_scan_Z]*len(Range_y[peaks_A[:,1]])]).T
		centers["B"] = np.array([Range_x[peaks_B[:,0]], Range_y[peaks_B[:,1]], [np_scan_Z]*len(Range_y[peaks_A[:,1]])]).T
		centers["AB"] = np.vstack((centers["A"],centers["B"]))
		'''
		try:
			centers["mid"] = (centers["A"]+centers["B"])/2
		except:
			traceback.print_exc()
			if len(centers['A'])>len(centers['B']):
				centers["mid"] = (centers["A"][:len(centers['B'])]+centers["B"])/2
			elif len(centers['A'])<len(centers['B']):
				centers["mid"] = (centers["A"]+centers["B"][:len(centers['A'])])/2
		'''
		print(centers)
		try:
			self.NP_centersA.setData(x=centers["A"][:,0],y=centers["A"][:,1])
			self.NP_centersB.setData(x=centers["B"][:,0],y=centers["B"][:,1])

			ch = self.ui.nMeasLaserSpectra_track_channel.currentIndex()
			ch_str = ['A','B','AB','mid']
			#self.ui.nMeasLaserSpectra_probe.setRowCount(len(centers[ch_str[ch]])+1)

			for i in range(1,len(centers[ch_str[ch]])+1):
				if i<=len(centers[ch_str[ch]]):
					c_coord = centers[ch_str[ch]][i-1]
				else:
					c_coord = [0,50]
				self.ui.nMeasLaserSpectra_probe.item(i,0).setText(str(c_coord[0]))
				self.ui.nMeasLaserSpectra_probe.item(i,1).setText(str(c_coord[1]))
				self.ui.nMeasLaserSpectra_probe.item(i,2).setText(str(np_scan_Z))
			print(centers)

			self.NPsCenters = centers[ch_str[ch]]

			return centers[ch_str[ch]]
		except:
			traceback.print_exc()
			return [[np.nan]*3]

	############################################################################
	###############################   scan3D	################################

	def scan1D_filePath_find(self):
		fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', self.ui.scan1D_filePath.text())
		if type(fname)==tuple:
			fname = fname[0]
		try:
			self.ui.scan1D_filePath.setText(fname)
		except:
			traceback.print_exc()

	def polarScan(self):
		fname = self.ui.scan1D_filePath.text()+"_"+str(round(time.time()))+".txt"
		with open(fname,'a') as f:
			f.write("#X\tY\tZ\tHWP\tpmtA_signal\tpmtB_signal\ttime\n")
		self.rotPiezoStage.move(self.ui.scanPolar_angle.value())
		isMoving = self.rotPiezoStage.isMoving()
		n = 0
		while isMoving and self.alive:
			print(n)
			n = n+1
			if n>10:
				isMoving = self.rotPiezoStage.isMoving()
				n = 0
			pmt_valA,pmt_valB = self.readPico()#self.readDAQmx(print_dt=True)
			self.live_pmtA = np.hstack((self.live_pmtA, pmt_valA))
			self.live_pmtB = np.hstack((self.live_pmtB, pmt_valB))

			self.line_pmtA.setData(self.live_pmtA)
			self.line_pmtB.setData(self.live_pmtB)
			app.processEvents()
			real_position = [round(p,4) for p in self.piStage.qPOS()]
			HWP_angle = float(self.ui.HWP_angle.text())

			dataSet = real_position +[HWP_angle, pmt_valA, pmt_valB, time.time()]# + spectra[spectra_range[0]:spectra_range[1]]
			#print(dataSet[-1])
			with open(fname,'a') as f:
				f.write("\t".join([str(round(i,6)) for i in dataSet])+"\n")
			print(n)
		self.ui.scanPolar.setChecked(False)

	def scanPolar(self,state):
		if state:
			print(state)
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.live_x = np.array([])
			self.alive = True
			self.polarScan()

		else:
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.live_x = np.array([])
			self.alive = False
			self.rotPiezoStage.stop()

	def scan1D(self):
		self.calibrTimer.stop()
		self.live_x = np.array([0])
		self.live_pmtA = np.array([0])
		self.live_pmtB = np.array([0])
		fname = self.ui.scan1D_filePath.text()+"_"+str(round(time.time()))+".txt"
		with open(fname,'a') as f:
			f.write("#X\tY\tZ\tHWP_power\tHWP_stepper\tpmtA_signal\tpmtB_signal\ttime\n")
		axis = self.ui.scan1D_axis.currentText()
		move_function = None
		if axis == "X":
			move_function = lambda pos: self.piStage.MOV([pos],axis=b'1',waitUntilReady=True)
		elif axis == "Y":
			move_function = lambda pos: self.piStage.MOV([pos],axis=b'2',waitUntilReady=True)
		elif axis == "Z":
			move_function = lambda pos: self.piStage.MOV([pos],axis=b'3',waitUntilReady=True)
		elif axis == 'HWP_Power':
			def move_function(pos):
				self.HWP.mAbs(pos)
				pos = self.HWP.getPos()
				self.ui.HWP_angle.setText(str(round(pos,6)))
		elif axis == 'HWP_Polarization':
			def move_function(pos):
				if not self.ui.HWP_stepper_Connect.isChecked():
					self.ui.HWP_stepper_Connect.setChecked(True)
				self.HWP_stepper_moveTo(float(pos),wait=True)
				pos = self.HWP_stepper_getAngle()
				#self.ui.HWP_stepper_angle.setText(str(round(pos,6)))

		steps_range = np.arange(self.ui.scan1D_start.value(),
								self.ui.scan1D_end.value(),
								self.ui.scan1D_step.value())
		for new_pos in steps_range:
			if not self.alive: break
			pmt_valA,pmt_valB,_ = self.getABData(update=True)#self.readDAQmx(print_dt=True)

			real_position = [round(p,4) for p in self.piStage.qPOS()]
			HWP_angle = float(self.ui.HWP_angle.text())
			HWP_stepper_angle = float(self.ui.HWP_stepper_angle.text())

			if axis == 'X':
				x = real_position[0]
				self.setUiPiPos(pos=real_position)
			elif axis == 'Y':
				x = real_position[1]
				self.setUiPiPos(pos=real_position)
			elif axis == 'Z':
				x = real_position[2]
				self.setUiPiPos(pos=real_position)
			elif axis == 'HWP_Power':
				x = HWP_angle
			elif axis == 'HWP_Polarization':
				x = HWP_stepper_angle
			else: x = 0
			self.live_pmtA = np.hstack((self.live_pmtA, pmt_valA))
			self.live_pmtB = np.hstack((self.live_pmtB, pmt_valB))
			self.live_x = np.hstack((self.live_x, x))
			#print(self.live_x,self.live_pmtA,self.live_pmtB)

			self.line_pmtA.setData(x=self.live_x,y=self.live_pmtA)
			self.line_pmtB.setData(x=self.live_x,y=self.live_pmtB)

			app.processEvents()
			dataSet = real_position +[HWP_angle, HWP_stepper_angle, pmt_valA, pmt_valB, time.time()]# + spectra[spectra_range[0]:spectra_range[1]]
			#print(dataSet[-1])
			with open(fname,'a') as f:
				f.write("\t".join([str(round(i,10)) for i in dataSet])+"\n")
			move_function(new_pos)
		self.ui.scan1D_Scan.setChecked(False)

	def scan1D_Scan(self,state):
		if state:
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.live_x = np.array([])
			self.alive = True

			self.scan1D_thread = threading.Thread(target=self.scan1D())
			self.scan1D_thread.daemon = True
			self.scan1D_thread.start()
		else:
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.live_x = np.array([])
			self.alive = False
			#self.rotPiezoStage.stop()

	############################################################################
	##########################   Ui   ##########################################
	def initUI(self):
		self.laserStatus.start(500)

		self.ui.actionExit.toggled.connect(self.closeEvent)

		self.ui.optim1step.clicked.connect(self.optim1step)

		self.ui.mocoConnect.toggled[bool].connect(self.mocoConnect)
		self.ui.mocoSection.currentIndexChanged[int].connect(self.mocoSetSection)
		self.ui.mocoMoveAbs.clicked.connect(self.mocoMoveAbs)

		self.ui.scan3D_config.cellChanged[int,int].connect(self.syncRectROI_table)
		self.ui.meas_laser_spectra_probe.cellChanged[int,int].connect(self.meas_laser_spectra_probe_update)

		self.ui.scan1D_filePath_find.clicked.connect(self.scan1D_filePath_find)
		self.ui.nMeasLaserSpectra_filePath_find.clicked.connect(self.nMeasLaserSpectra_filePath_find)

		self.ui.scan3D_path_dialog.clicked.connect(self.scan3D_path_dialog)

		self.ui.scan3D_peak_find.clicked.connect(self.scan3D_peak_find)

		self.ui.scan3D_generateMask.clicked.connect(self.generate2Dmask)

		self.ui.usbSpectr_set_integr_time.clicked.connect(self.usbSpectr_set_integr_time)

		self.ui.connect_DAQmx.toggled[bool].connect(self.connect_DAQmx)
		self.ui.DAQmx_find_shift.clicked.connect(self.optimizeDAQmx)

		self.ui.HWP_go.clicked.connect(self.HWP_go)
		self.ui.HWP_go_home.clicked.connect(self.HWP_go_home)
		self.ui.HWP_negative_step.clicked.connect(self.HWP_negative_step)
		self.ui.HWP_positive_step.clicked.connect(self.HWP_positive_step)

		self.ui.Pi_X_go.clicked.connect(self.Pi_X_go)
		self.ui.Pi_Y_go.clicked.connect(self.Pi_Y_go)
		self.ui.Pi_Z_go.clicked.connect(self.Pi_Z_go)
		self.ui.Pi_XYZ_50mkm.clicked.connect(self.Pi_XYZ_50mkm)
		self.ui.Pi_autoZero.clicked.connect(self.Pi_autoZero)
		self.ui.zSlider.valueChanged[int].connect(self.setPiStateZPosition)

		self.ui.connect_rotPiezoStage.toggled[bool].connect(self.connect_rotPiezoStage)
		self.ui.rotPiezoStage_Go.clicked.connect(self.rotPiezoStage_Go)

		self.ui.scanPolar.toggled[bool].connect(self.scanPolar)

		self.ui.scan1D_Scan.toggled[bool].connect(self.scan1D_Scan)

		self.ui.start3DScan.toggled[bool].connect(self.start3DScan)

		self.ui.fast3DScan.toggled[bool].connect(self.start_fast3DScan)

		self.ui.startCalibr.toggled[bool].connect(self.startCalibr)
		self.calibrTimer.timeout.connect(self.onCalibrTimer)

		self.ui.laserSetWavelength.clicked.connect(self.laserSetWavelength)
		self.ui.laserSetShutter.clicked.connect(self.laserSetShutter)
		self.ui.actionShutter.toggled[bool].connect(self.laserSetShutter_)
		self.laserStatus.timeout.connect(self.onLaserStatus)

		self.ui.connect_pico.toggled[bool].connect(self.connect_pico)
		self.ui.pico_set.clicked.connect(self.pico_set)

		self.ui.Pi_Set.clicked.connect(self.Pi_Set)

		self.ui.shamrockConnect.toggled[bool].connect(self.shamrockConnect)
		self.ui.shamrockSetWavelength.clicked.connect(self.shamrockSetWavelength)
		self.ui.shamrockPort.currentIndexChanged[int].connect(self.shamrockSetPort)
		self.ui.shamrockGrating.currentIndexChanged[int].connect(self.shamrockSetGrating)


		self.ui.andorCameraConnect.toggled[bool].connect(self.andorCameraConnect)
		self.ui.andorCameraExposure.valueChanged[float].connect(self.andorCameraSetExposure)
		self.ui.andorCameraGetData.toggled[bool].connect(self.andorCameraGetData)
		self.ui.andorCameraReadoutMode.currentIndexChanged[int].connect(self.andorCameraSetReadoutMode)
		self.andorCameraLiveTimer.timeout.connect(self.onAndorCameraLiveTimeout)
		self.ui.andorCameraGetBaseline.clicked.connect(self.andorCameraGetBaseline)
		self.ui.actionBaseline.triggered.connect(self.andorCameraGetBaseline)
		self.ui.actionSpectra.toggled[bool].connect(self.andorCameraGetData)
		self.ui.andorCameraCleanLines.clicked.connect(self.andorCameraCleanLines)


		self.ui.HWP_stepper_Connect.toggled[bool].connect(self.HWP_stepper_Connect)
		self.ui.HWP_stepper_MoveTo_Go.clicked.connect(self.HWP_stepper_MoveTo_Go)
		self.ui.HWP_stepper_CW.clicked.connect(self.HWP_stepper_CW)
		self.ui.HWP_stepper_CCW.clicked.connect(self.HWP_stepper_CCW)
		self.ui.HWP_stepper_Reset.clicked.connect(self.HWP_stepper_Reset)


		self.ui.pm100Connect.toggled[bool].connect(self.pm100Connect)
		self.ui.pm100Average.valueChanged[int].connect(self.pm100Average)

		self.ui.confParam_scan.toggled[bool].connect(self.confParam_scan)

		self.ui.meas_laser_spectra_go.toggled[bool].connect(self.meas_laser_spectra_go)
		self.ui.nMeasLaserSpectra_go.toggled[bool].connect(self.nMeasLaserSpectra_go)

		self.ui.powerCalibr_go.toggled[bool].connect(self.laserPowerCalibr)

		self.ui.actionClean.triggered.connect(self.viewCleanLines)
		self.ui.saveGuiConfig.clicked.connect(self.saveGuiConfig)


		self.ui.nMeasLaserSpectra_setPoint.toggled[bool].connect(self.nMeasLaserSpectra_setPoint)
		###########################################################!!!!!!!!!!!!!
		self.comboReadoutSources = QtGui.QComboBox()
		self.ui.toolBar.addWidget(self.comboReadoutSources)
		sources = [self.ui.readoutSources.itemText(i) for i in range(self.ui.readoutSources.count())]
		self.comboReadoutSources.insertItems(1,sources)
		self.comboReadoutSources.currentIndexChanged[int].connect(self.ui.readoutSources.setCurrentIndex)
		self.comboReadoutSources.blockSignals(True)
		self.ui.readoutSources.currentIndexChanged[int].connect(self.comboReadoutSources.setCurrentIndex)
		self.comboReadoutSources.blockSignals(False)

		self.comboMocoSection = QtGui.QComboBox()
		self.ui.toolBar.addWidget(self.comboMocoSection)
		ports = [self.ui.mocoSection.itemText(i) for i in range(self.ui.mocoSection.count())]
		self.comboMocoSection.insertItems(1,ports)
		self.comboMocoSection.currentIndexChanged[int].connect(self.ui.mocoSection.setCurrentIndex)
		self.comboMocoSection.blockSignals(True)
		self.ui.mocoSection.currentIndexChanged[int].connect(self.comboMocoSection.setCurrentIndex)
		self.comboMocoSection.blockSignals(False)

		self.trackChannel = QtGui.QComboBox()
		self.ui.toolBar.addWidget(self.trackChannel)
		ports = [self.ui.nMeasLaserSpectra_track_channel.itemText(i) for i in range(self.ui.nMeasLaserSpectra_track_channel.count())]
		self.trackChannel.insertItems(1,ports)
		self.trackChannel.currentIndexChanged[int].connect(self.ui.nMeasLaserSpectra_track_channel.setCurrentIndex)
		self.trackChannel.blockSignals(True)
		self.ui.nMeasLaserSpectra_track_channel.currentIndexChanged[int].connect(self.trackChannel.setCurrentIndex)
		self.trackChannel.blockSignals(False)

		self.andorCameraExposure_toolBar = QtGui.QDoubleSpinBox()
		self.andorCameraExposure_toolBar.setKeyboardTracking(False)
		self.ui.toolBar.addWidget(self.andorCameraExposure_toolBar)
		self.andorCameraExposure_toolBar.valueChanged[float].connect(self.andorCameraSetExposure_)

		########################################################################
		########################################################################
		########################################################################




		self.tabColors = {
			0: 'green',
			1: 'red',
			2: 'wine',
			3: 'orange',
			4: 'blue',
			5: 'cyan',
			6: 'magenta',
			7: 'yellow'
		}
		self.ui.configTabWidget.tabBar().currentChanged.connect(self.styleTabs)

		self.pw = pg.PlotWidget(name='PlotMain')  ## giving the plots names allows us to link their axes together


		#self.ui.plotArea.addWidget(self.pw)
		self.line0 = self.pw.plot()
		self.line1 = self.pw.plot(pen=(255,0,0))
		self.line3 = self.pw.plot(pen=(255,0,255))

		self.pw_preview = pg.PlotWidget(name='preview')  ## giving the plots names allows us to link their axes together
		self.ui.previewArea.addWidget(self.pw_preview)



		self.line_DAQmx_sig = self.pw_preview.plot(pen=(255,0,0))
		self.line_DAQmx_sig1 = self.pw_preview.plot(pen=(255,255,0))
		self.line_DAQmx_ref = self.pw_preview.plot(pen=(255,0,255))

		self.line_pico_ChA = self.pw_preview.plot(pen=(255,215,0))
		self.line_pico_ChB = self.pw_preview.plot(pen=(0,255,255))

		self.pw1 = pg.PlotWidget(name='Scan')  ## giving the plots names allows us to link their axes together

		self.line_pmtA = self.pw1.plot(pen=(255,215,0))
		self.line_pmtB = self.pw1.plot(pen=(0,255,255))
		self.line_spectra_central = self.pw1.plot(pen=(100,255,0))

		self.pw_spectra = pg.PlotWidget(name='Spectra')
		self.line_spectra = []
		cr = range(0,255)
		for i in range(10):
			pen = (cr[::20][i],cr[::-20][i],cr[::25][i])
			print(pen)
			self.line_spectra.append(self.pw_spectra.plot(pen=pen))

		splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
		splitter.addWidget(self.pw1)
		splitter.addWidget(self.pw_spectra)

		self.ui.plotArea.addWidget(splitter)

		self.img = pg.ImageView()
		data = np.zeros((100,100))
		self.ui.imageArea.addWidget(self.img)
		self.img.setImage(data)
		self.img.view.invertY(False)

		colors = [(0, 0, 0),(255, 214, 112)]
		cmap = pg.ColorMap(pos=[0.,1.], color=colors)
		self.img.setColorMap(cmap)
		self.grid = pg.GridItem()
		self.img.addItem(self.grid)




		self.rectROI = pg.RectROI([0, 0], [100, 100], pen=(0,9))
		self.rectROI_1 = pg.RectROI([0, 0], [100, 100], pen=(0,9))


		self.img.addItem(self.rectROI)

		self.spectra_bg_probe =  pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(100, 0, 255))
		self.spectra_bg_probe1 =  pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(100, 0, 255))
		x=float(self.ui.meas_laser_spectra_probe.item(0,0).text())
		y=float(self.ui.meas_laser_spectra_probe.item(0,1).text())
		z=float(self.ui.meas_laser_spectra_probe.item(0,1).text())

		self.spectra_bg_probe.setData(x=[x],y=[y])
		self.spectra_bg_probe1.setData(x=[x],y=[y])

		self.spectra_signal_probe = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 10))
		self.spectra_signal_probe1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 10))

		self.NP_centersA = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 153))
		self.NP_centersB = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 153))
		self.img.addItem(self.NP_centersA)
		x=float(self.ui.meas_laser_spectra_probe.item(1,0).text())
		y=float(self.ui.meas_laser_spectra_probe.item(1,1).text())
		z=float(self.ui.meas_laser_spectra_probe.item(1,1).text())

		self.spectra_signal_probe.setData(x=[x],y=[y])
		self.spectra_signal_probe1.setData(x=[x],y=[y])

		#self.img.addItem(self.spectra_bg_probe)
		#self.img.addItem(self.spectra_signal_probe)





		self.img1 = pg.ImageView()  ## giving the plots names allows us to link their axes together
		data = np.zeros((100,100))
		self.ui.imageArea.addWidget(self.img1)
		self.img1.setImage(data)
		self.img1.view.invertY(False)

		colors = [(0, 0, 0),(204, 255, 255)]
		cmap = pg.ColorMap(pos=[0.,1.], color=colors)
		self.img1.setColorMap(cmap)
		self.grid1 = pg.GridItem()
		self.img1.addItem(self.grid1)
		self.img1.addItem(self.rectROI_1)
		self.rectROI.sigRegionChanged.connect(self.syncRectROI)
		self.rectROI_1.sigRegionChanged.connect(self.syncRectROI)

		#self.img1.addItem(self.spectra_bg_probe1)
		#self.img1.addItem(self.spectra_signal_probe1)
		self.img1.addItem(self.NP_centersB)
		self.move_scat_flag = False
		self.move_scat_item = None

		def onImgPos(event):
			pos = self.grid.mapFromScene(event.scenePos())

			if event.modifiers() & QtCore.Qt.ControlModifier:
				print(pos.x(),pos.y())
				if self.ui.nMeasLaserSpectra_setPoint.isChecked() and not self.nMeasLaserSpectra_probe_currentRow==-1:
					coord = [str(pos.x()),str(pos.y()),self.statusBar_Position_Z.text()]
					print(self.nMeasLaserSpectra_probe_currentRow)
					for i in range(3):
						self.ui.nMeasLaserSpectra_probe.item(self.nMeasLaserSpectra_probe_currentRow,i).setText(coord[i])
					self.ui.nMeasLaserSpectra_setPoint.setChecked(False)
			if event.modifiers() & QtCore.Qt.ShiftModifier:
				print(pos.x(),pos.y())
				if pos.x()>0 and pos.x()<100 and pos.y()>0 and pos.y()<100:
					self.piStage.MOV([pos.x(),pos.y()],b'1 2', waitUntilReady=1)
					real_position = self.piStage.qPOS()
					self.setUiPiPos(real_position)


		def onImg1Pos(event):
			pos = self.grid1.mapFromScene(event.scenePos())

			if event.modifiers() & QtCore.Qt.ControlModifier:
				print(pos.x(),pos.y())
				if self.ui.nMeasLaserSpectra_setPoint.isChecked() and not self.nMeasLaserSpectra_probe_currentRow==-1:
					coord = [str(pos.x()),str(pos.y()),self.statusBar_Position_Z.text()]

					for i in range(3):
						self.ui.nMeasLaserSpectra_probe.item(self.nMeasLaserSpectra_probe_currentRow,i).setText(coord[i])
					self.ui.nMeasLaserSpectra_setPoint.setChecked(False)
			if event.modifiers() & QtCore.Qt.ShiftModifier:
				print(pos.x(),pos.y())
				if pos.x()>0 and pos.x()<100 and pos.y()>0 and pos.y()<100:
					self.piStage.MOV([pos.x(),pos.y()],b'1 2', waitUntilReady=1)
					real_position = self.piStage.qPOS()
					self.setUiPiPos(real_position)

		self.grid.scene().sigMouseClicked.connect(onImgPos)
		self.grid1.scene().sigMouseClicked.connect(onImg1Pos)
		self.piStage_position = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0),symbol='+')
		self.img.addItem(self.piStage_position)

		self.piStage_position1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0),symbol='+')
		self.img1.addItem(self.piStage_position1)

		self.statusBar_Position_X = QtGui.QLabel('X')
		self.ui.statusbar.addWidget(self.statusBar_Position_X)
		self.statusBar_Position_X.setStyleSheet('color:orange;')
		self.statusBar_Position_Y = QtGui.QLabel('Y')
		self.ui.statusbar.addWidget(self.statusBar_Position_Y)
		self.statusBar_Position_Y.setStyleSheet('color:pink;')
		self.statusBar_Position_Z = QtGui.QLabel('Z')
		self.ui.statusbar.addWidget(self.statusBar_Position_Z)
		self.statusBar_Position_Z.setStyleSheet('color:cyan;')
		self.statusBar_ExWavelength = QtGui.QLabel('[exWl]')
		self.ui.statusbar.addWidget(self.statusBar_ExWavelength)
		self.statusBar_Shutter = QtGui.QLabel('[SHUTTER]')
		self.ui.statusbar.addWidget(self.statusBar_Shutter)

		self.statusBar_Filter = QtGui.QLabel('[Filter/PMT]')
		self.ui.statusbar.addWidget(self.statusBar_Filter)

		self.statusBar_ShamrockPort = QtGui.QLabel('[Shamrock]')
		self.statusBar_ShamrockPort.setStyleSheet('color:orange;')
		self.ui.statusbar.addWidget(self.statusBar_ShamrockPort)




		restore_gui(self.settings)
		self.ui.laserShutter.setChecked(False)
		try:
			with open('scanArea.csv') as f:
				data = list(csv.reader(f,delimiter='\t'))
			print(data)
			d = []
			for i in data:
				if not len(i)==0:
					d.append(i)
			data = d
			rows = self.ui.scan3D_config.rowCount()
			columns = self.ui.scan3D_config.columnCount()
			for r in range(rows):
				for c in range(columns):
					try:
						self.ui.scan3D_config.item(r,c).setText(data[r][c])
					except:
						traceback.print_exc()
						self.ui.scan3D_config.item(r,1).setText('0')
						self.ui.scan3D_config.item(r,2).setText('100')
						self.ui.scan3D_config.item(r,3).setText('10')
		except:
			print('scanArea_recovery_err')
		stop_flag = threading.Event()
		self.timer_thread = TimerThread(stop_flag)
		self.timer_thread.update.connect(self.update_ui)
		self.timer_thread.start()

	def update_ui(self):
		self.pw1.update()
		self.pw.update()
		self.pw_preview.update()
		self.pw_spectra.update()
		self.img.update()
		self.img1.update()
		#self.img.updateImage(autoHistogramRange=False)
		#self.img1.updateImage(autoHistogramRange=False)


		#self.ui.configTabWidget.setStyleSheet('QTabBar::tab[objectName="Readout"] {background-color=red;}')
	def viewCleanLines(self):
		for i in range(len(self.line_spectra)):
			self.line_spectra[i].setData(x=[], y = [])
		self.line_pmtA.setData(x=[], y = [])
		self.line_pmtB.setData(x=[], y = [])

		self.line_pico_ChA.setData(x=[], y = [])
		self.line_pico_ChB.setData(x=[], y = [])
		self.line_spectra_central.setData(x=[], y = [])
		self.live_x = np.array([])
		self.live_y = np.array([])
		self.live_pmtB = np.array([])
		self.live_pmtA = np.array([])
		self.live_integr_spectra = np.array([])




	def syncRectROI(self):
		sender = self.sender()
		pos = list(sender.pos())
		size = list(sender.size())
		sender.blockSignals(True)
		if pos[0] < 0:
			pos[0] = 0
		if pos[0] > 100:
			pos[0] = 100
		if pos[1] < 0:
			pos[1] = 0
		if pos[1] > 100:
			pos[1] = 100

		if pos[0]+size[0] > 100:
			size[0] = 100-pos[0]
		if pos[1]+size[1] > 100:
			size[1] = 100-pos[1]
		#self.spectra_signal_probe.setData(x=[pos[0]+size[0]/2],y=[pos[1]+size[1]/2])
		#self.spectra_signal_probe1.setData(x=[pos[0]+size[0]/2],y=[pos[1]+size[1]/2])
		#self.ui.meas_laser_spectra_probe
		#self.spectra_bg_probe.setData(x=[pos[0]],y=[pos[1]])
		#self.spectra_bg_probe1.setData(x=[pos[0]],y=[pos[1]])
		sender.setPos(pos)
		sender.setSize(size)
		sender.blockSignals(False)

		if sender == self.rectROI:
			self.rectROI_1.blockSignals(True)
			self.rectROI_1.setPos(pos)
			self.rectROI_1.setSize(size)
			self.rectROI_1.blockSignals(False)

		else:
			self.rectROI.blockSignals(True)
			self.rectROI.setPos(pos)
			self.rectROI.setSize(size)
			self.rectROI.blockSignals(False)
		self.ui.scan3D_config.item(2,1).setText(str(pos[0]))
		self.ui.scan3D_config.item(1,1).setText(str(pos[1]))

		self.ui.scan3D_config.item(2,2).setText(str(size[0]+pos[0]))
		self.ui.scan3D_config.item(1,2).setText(str(size[1]+pos[1]))

		if MODE == 'sim':
			self.ui.pico_n_captures.setValue(10)

			self.ui.pico_ChA_VRange.setCurrentIndex(4)
			self.ui.pico_ChB_VRange.setCurrentIndex(4)
			self.ui.pico_sampleInterval.setText('0.000001')
			self.ui.pico_samplingDuration.setText('0.007')

			self.ui.pico_TrigSource.setCurrentIndex(2)
			self.ui.pico_TrigThreshold.setValue(-0.330)
			self.pico_pretrig = 0.0004
			self.ui.pico_pretrig.setText('0.0004')

		#self.ui.meas_laser_spectra_probe.item(0,0).setText(str(pos[0]))
		#self.ui.meas_laser_spectra_probe.item(0,0).setText(str(pos[1]))



	def syncRectROI_table(self,row,col):
		pos = [0,0]
		pos[0] = float(self.ui.scan3D_config.item(2,1).text())
		pos[1] = float(self.ui.scan3D_config.item(1,1).text())

		size = [0,0]
		size[0] = float(self.ui.scan3D_config.item(2,2).text())-pos[0]
		size[1] = float(self.ui.scan3D_config.item(1,2).text())-pos[1]

		self.rectROI.blockSignals(True)
		self.rectROI_1.blockSignals(True)

		self.rectROI.setPos(pos)
		self.rectROI.setSize(size)

		self.rectROI_1.setPos(pos)
		self.rectROI_1.setSize(size)

		self.rectROI.blockSignals(False)
		self.rectROI_1.blockSignals(False)


	def styleTabs(self, index):
		self.ui.configTabWidget.setStyleSheet('''
			QTabBar::tab {{}}
			QTabBar::tab:selected {{background-color: {color};}}
			'''.format(color=self.tabColors[index]))
	def saveGuiConfig(self):
		save_gui(self.settings)

	def closeEvent(self, evnt):
		print('closeEvent')
		save_gui(self.settings)
		data = []
		rows = self.ui.scan3D_config.rowCount()
		columns = self.ui.scan3D_config.columnCount()
		import csv
		for r in range(rows):
			d = []
			for c in range(columns):
				d.append(self.ui.scan3D_config.item(r,c).text())
			data.append(d)
		with open("scanArea.csv", "w+") as myCsv:
			csvWriter = csv.writer(myCsv, delimiter='\t')
			csvWriter.writerows(data)
		try:
			self.piStage.CloseConnection()
		except:
			traceback.print_exc()
		try:
			self.spectrometer.close()
		except:
			traceback.print_exc()
		try:
			self.HWP.cleanUpAPT()
		except:
			traceback.print_exc()
		try:
			if self.power_meter_instr:
				self.power_meter_instr.close()
		except:
			traceback.print_exc()

		try:
			if self.ps:
				del self.ps
		except:
			traceback.print_exc()
		try:
			self.rotPiezoStage.close()
		except:
			traceback.print_exc()
		try:
			if self.ui.shamrockConnect.isChecked():
				self.shamrock.Close()
		except:
			traceback.print_exc()
		try:
			if self.ui.andorCameraConnect.isChecked():
				self.andorCCD.ShutDown()
		except:
			traceback.print_exc()
		try:
			if not self.HWP_stepper is None:

				self.HWP_stepper.close()
		except:
			traceback.print_exc()


	def setImage(self,data):
		self.img.setImage(data.T,levels=(data.min(), data.max()))
		#app.processEvents()

	def setImage1(self,data):
		self.img1.setImage(data.T,levels=(data.min(), data.max()))
		#app.processEvents()


		#app.processEvents()



if __name__ == '__main__':
	import sys
	__spec__ = None

	app = QtGui.QApplication(sys.argv)
	try:
		app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
	except RuntimeError:
		app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt())
	ex = microV()
	app.exec_()
