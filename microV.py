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

from skimage.feature.peak import peak_local_max

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
	from hardware.HWP_stepper import *

from hardware.pico_radar import fastScan
from hardware.pico_multiproc_picopy import *

#get_ipython().run_line_magic('matplotlib', 'qt')

import traceback
from multiprocessing import Process, Queue, Array


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


class microV(QtGui.QMainWindow):
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
	#ps = ps3000a.PS3000a(connect=False)
	ps = None
	n_captures = None

	shamrock = ShamRockController()
	andorCCD = AndorCamera()
	HWP_stepper = None
	#inst = visa.instrument('USB0::0x0000::0x0000::DG5Axxxxxxxxx::INSTR', term_chars='\n', timeout=1)
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

	processOut = Queue()
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
	def laserSetShutter(self):
		self.laserStatus.stop()
		state = self.ui.laserShutter.isChecked()
		if state==0:
			state = '1'
		else:
			state = '0'

		with open('laserIn','w+') as f:
			f.write('SHUTter '+state+'\n')
		self.laserStatus.start(1000)

	def laserSetWavelength(self,status=None,wavelength=None):
		self.laserStatus.stop()
		if wavelength is None or wavelength == False:
			wavelength = self.ui.laserWavelength_to_set.value()
		print(wavelength)
		with open('laserIn','w+') as f:
			f.write('WAVelength '+str(wavelength)+'\n')
		self.laserStatus.start(1000)

	def laserSetWavelength_(self,status=None,wavelength=None):
		self.laserStatus.stop()
		self.ui.laserWavelength_to_set.setValue(wavelength)
		print(wavelength)
		with open('laserIn','w+') as f:
			f.write('WAVelength '+str(wavelength)+'\n')
		self.laserStatus.start(1000)

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
			self.ui.laserShutter.setChecked(shutter!=0)
			self.ui.laserWavelength_ready.setChecked(wavelength_ready)

		except:
			traceback.print_exc()
			print('Laser: noData')

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
		#dataA = np.zeros((self.n_captures, self.samples_per_segment), dtype=np.int16)
		#dataB = np.zeros((self.n_captures, self.samples_per_segment), dtype=np.int16)
		#t1 = time.time()

		#self.ps.runBlock()
		try:
			r = self.ps.capture_prep_block(return_scaled_array=1)
		except:
			traceback.print_exc()
			return [0,0]


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
			self.line_pico_ChA.setData(x=t,y=dataA)
			self.line_pico_ChB.setData(x=t,y=dataB)

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
		self.power_meter.sense.correction.wavelength = float(self.ui.laserWavelength.text())
		val = self.power_meter.read
		self.ui.pm100Power.setText(str(val))
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
		print(self.piStage.qVEL())
		#print(self.piStage.DCO([1,1,1],b'1 2 3'))
		#print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=1,waitUntilReady=True))
		time.sleep(0.2)
		#print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=2,waitUntilReady=True))
		#time.sleep(0.2)
		#print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=3,waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)
		time.sleep(1)

	def setUiPiPos(self,pos):
		self.ui.Pi_XPos.setText(str(pos[0]))
		self.ui.Pi_YPos.setText(str(pos[1]))
		self.ui.Pi_ZPos.setText(str(pos[2]))
		self.statusBar_Position.setText('[%.5f\t%.5f\t%.5f]'%tuple(pos))

	def Pi_X_go(self):
		pos = self.ui.Pi_X_move_to.value()
		print(self.piStage.MOV(pos,axis=1,waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_Y_go(self):
		pos = self.ui.Pi_Y_move_to.value()
		print(self.piStage.MOV(pos,axis=2,waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)

	def Pi_Z_go(self):
		pos = self.ui.Pi_Z_move_to.value()
		print(self.piStage.MOV(pos,axis=3,waitUntilReady=True))
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
	def shamrockConnect(self, state):
		if state:
			self.shamrock.Initialize()
			self.shamrock.Connect()

			wavelength = self.shamrock.shamrock.GetWavelength()
			self.ui.shamrockWavelength.setText(str(wavelength))
			port = self.shamrock.shamrock.GetPort()
			self.ui.shamrockPort.blockSignals(True)
			self.ui.shamrockPort.setCurrentIndex(port)
			self.ui.shamrockPort.blockSignals(False)
			grating = self.shamrock.shamrock.GetGrating()
			self.ui.shamrockGrating.blockSignals(True)
			self.ui.shamrockGrating.setCurrentIndex(grating-1)
			self.ui.shamrockGrating.blockSignals(False)

		else:
			self.shamrock.Close()


	def shamrockSetWavelength(self, wl=None):

		if not self.ui.shamrockConnect.isChecked():
			self.ui.shamrockConnect.setChecked(True)
		if wl is None or wl == False:
			wl = self.ui.shamrockWavelength_to_set.value()
		print(wl)
		self.shamrock.shamrock.SetWavelength(wl)
		wavelength = self.shamrock.shamrock.GetWavelength()
		self.ui.shamrockWavelength.setText(str(wavelength))

	def shamrockSetPort(self,val):
		if not self.ui.shamrockConnect.isChecked():
			self.ui.shamrockConnect.setChecked(True)
		port = val
		self.shamrock.shamrock.SetPort(port)
		port = self.shamrock.shamrock.GetPort()
		self.ui.shamrockPort.blockSignals(True)
		self.ui.shamrockPort.setCurrentIndex(port)
		self.ui.shamrockPort.blockSignals(False)

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

	def andorCameraConnect(self, state):
		if state:

			self.andorCCD.Initialize()
			self.andorCCD.SetExposureTime(self.ui.andorCameraExposure.value())
			self.andorCCDBaseline = np.array([])
			size=self.andorCCD.GetPixelSize()
			self.andorCCD_wavelength = np.arange(size[0])
			self.andorCCD_wavelength_center = float(self.ui.shamrockWavelength.text())
			if self.ui.shamrockConnect.isChecked():
				shape=self.andorCCD.GetDetector()
				size=self.andorCCD.GetPixelSize()
				self.shamrock.shamrock.SetPixelWidth(size[0])
				self.shamrock.shamrock.SetNumberPixels(shape[0])
				self.andorCCD_wavelength = self.shamrock.shamrock.GetCalibration()
		else:
			self.andorCCD.ShutDown()

	def andorCameraSetExposure(self,val):

		self.andorCCD.SetExposureTime(val)
		val = self.andorCCD.GetAcquisitionTimings()[0]

	def andorCameraSetReadoutMode(self,val):
		mode = self.ui.andorCameraReadoutMode.currentText()
		self.andorCCD.SetReadMode(mode)

	def andorCameraGetData(self,state=False, integr_range = [], index=0):
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

				if self.ui.shamrockConnect.isChecked():
					c = float(self.ui.shamrockWavelength.text())
					w_c = [c-20, c+20]
					if len(integr_range)>0:
						w_c = integr_range
					if not w_c == self.andorCCD_wavelength_center:
						shape=self.andorCCD.GetDetector()
						size=self.andorCCD.GetPixelSize()
						self.shamrock.shamrock.SetPixelWidth(size[0])
						self.shamrock.shamrock.SetNumberPixels(shape[0])
						self.andorCCD_wavelength = self.shamrock.shamrock.GetCalibration()

					w_r = (self.andorCCD_wavelength>w_c[0])&(self.andorCCD_wavelength<w_c[1])
					data_center = float(data[w_r].mean())
				self.line_spectra[index].setData(x=self.andorCCD_wavelength, y = data)
				self.ui.andorCameraGetData.setChecked(False)
			else:
				self.andorCameraLiveTimer.start(10)
		else:
			self.andorCameraLiveTimer.stop()
		return data_center, data, self.andorCCD_wavelength

	def onAndorCameraLiveTimeout(self):
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

		self.andorCameraLiveTimer.start(100)

	def andorCameraGetBaseline(self):
		print('andorCameraGetBaseline')
		self.andorCCD.StartAcquisition()
		self.andorCCD.WaitForAcquisition()
		data = self.andorCCD.GetMostRecentImage()
		self.andorCCDBaseline = data
		wavelength = np.arange(len(data))
		if self.ui.shamrockConnect.isChecked():
			shape=self.andorCCD.GetDetector()
			size=self.andorCCD.GetPixelSize()
			self.shamrock.shamrock.SetPixelWidth(size[0])
			self.shamrock.shamrock.SetNumberPixels(shape[0])
			self.shamrock.shamrock.GetCalibration()
			wavelength = self.shamrock.shamrock.GetCalibration()
		self.line_spectra[0].setData(x=wavelength, y = data)
	############################################################################
	###############################   HWP_stepper	############################

	def HWP_stepper_Connect(self,state):
		if state:
			self.HWP_stepper = HWP_stepper(4000,0.01, "dev1/ctr1", reset=True)
			self.ui.HWP_stepper_angle.setText('0')

		else:
			try:
				self.HWP_stepper.close()
			except:
				traceback.print_exc()
			self.HWP_stepper = None
	def HWP_stepper_MoveTo_Go(self):
		if not self.ui.HWP_stepper_Connect.isChecked():
			self.ui.HWP_stepper_Connect.setChecked(True)

		if not self.HWP_stepper is None:
			angle = self.ui.HWP_stepper_MoveTo.value()
			wait = self.ui.HWP_stepper_wait.isChecked()
			self.HWP_stepper.moveTo(angle,wait=wait)
			angle_ = self.HWP_stepper.getAngle()
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
			self.ui.HWP_stepper_angle.setText(str(angle_))

	def HWP_stepper_Reset(self):
		if not self.HWP_stepper is None:
			self.HWP_stepper.resetAngel()

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
				#self.ui.connect_pico.toggled.emit(True)
				self.ui.connect_pico.setChecked(True)
			self.calibrTimer.start(100)
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.live_x = np.array([])
			self.live_y = np.array([])

			self.live_integr_spectra = np.array([])
		else:
			self.calibrTimer.stop()

	def onCalibrTimer(self):
		self.calibrTimer.stop()
		#spectra = np.zeros(3648)#self.getSpectra()
		pmt_valA,pmt_valB = self.readPico()#self.readDAQmx(print_dt=True)
		self.live_pmtA = np.hstack((self.live_pmtA,pmt_valA))
		self.live_pmtB = np.hstack((self.live_pmtB,pmt_valB))
		if len(self.live_pmtA)>800:
			self.live_pmtA = self.live_pmtA[1:]
		if len(self.live_pmtB)>800:
			self.live_pmtA = self.live_pmtA[1:]
		#s_from = self.ui.usbSpectr_from.value()
		#s_to = self.ui.usbSpectr_to.value()
		#self.live_integr_spectra.append(np.sum(spectra[s_from:s_to])/1000)
		#setLine(spectra)
		self.line_pmtA.setData(self.live_pmtA)
		self.line_pmtB.setData(self.live_pmtB)
		#self.line_spectra.setData(self.live_integr_spectra)

		self.calibrTimer.start(100)
		app.processEvents()

	def confParam_scan(self, state):
		if state:
			start_wavelength = self.ui.calibr_wavelength_start.value()
			end_wavelength = self.ui.calibr_wavelength_end.value()
			step_wavelength = self.ui.calibr_wavelength_step.value()
			self.scan3DisAlive = True
			wl_range = np.arange(start_wavelength,end_wavelength,step_wavelength)
			with pd.HDFStore('data/confParam_scan'+str(round(time.time()))+'.h5') as store:
				store.keys()
				self.live_x = np.array([])
				self.live_integr_spectra = np.array([])
				for wl in wl_range:
					self.laserSetWavelength_(status=1,wavelength=wl)

					time.sleep(3)
					t0 = time.time()
					while not wl == float(self.ui.laserWavelength.text()) and time.time()-t0<10 and self.scan3DisAlive:
						time.sleep(0.5)
						print('wait:Laser')
						app.processEvents()


					self.shamrockSetWavelength(wl/3)
					self.andorCameraGetBaseline()
					#integr_intens,intens,wavelength_arr = self.andorCameraGetData(state=1,integr_range=[wl/3-20,wl/3+20])


					start_Z = self.ui.confParam_scan_start.value()
					end_Z = self.ui.confParam_scan_end.value()
					step_Z = self.ui.confParam_scan_step.value()
					Range_Z = np.arange(start_Z,end_Z,step_Z)

					df = pd.DataFrame(self.andorCCDBaseline, index=self.andorCCD_wavelength,columns=['baseline'])
					for z in Range_Z:
						print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
						real_position = self.piStage.qPOS()
						z_real = real_position[2]
						integr_intens,intens,wavelength_arr = self.andorCameraGetData(state=1,integr_range=[wl/3-20,wl/3+20])
						df[str(z)] = intens
						self.live_x = np.hstack((self.live_x, z_real))
						self.live_integr_spectra = np.hstack((self.live_integr_spectra, integr_intens))

						self.line_spectra_central.setData(x=self.live_x,y=self.live_integr_spectra)
						app.processEvents()
						if not self.scan3DisAlive:
							store.put("scan_"+str(wl), df)
							return
							break
					store.put("scan_"+str(wl), df)



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
					integr_intens_SHG,intens,wavelength_arr = self.andorCameraGetData(state=1,integr_range=[wl/2-20,wl/2+20])
					time_list.append(time.time())
					w = (wavelength_arr>wl/3-20)&(wavelength_arr>wl/3+20)
					integr_intens_THG = intens[w].sum()
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

	def n_meas_laser_spectra_go(self,state):
		if state:
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			start_wavelength = self.ui.n_meas_laser_spectra_start.value()
			end_wavelength = self.ui.n_meas_laser_spectra_end.value()
			step_wavelength = self.ui.n_meas_laser_spectra_step.value()
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


					bg_center = np.array([ float(self.ui.n_meas_laser_spectra_probe.item(0,i).text()) for i in range(3)])


					spectra_center = np.array([ float(self.ui.n_meas_laser_spectra_probe.item(1,i).text()) for i in range(3)])

					NP_centers = np.array([[ float(self.ui.n_meas_laser_spectra_probe.item(j,i).text()) for i in range(3)] for j in range(self.ui.n_meas_laser_spectra_probe.rowCount())])

					z_start = self.ui.n_meas_laser_spectra_Z_start.value()
					z_end = self.ui.n_meas_laser_spectra_Z_end.value()
					z_step = self.ui.n_meas_laser_spectra_Z_step.value()

					Range_z = np.arange(z_start,z_end,z_step)
					data_Z = np.zeros(len(Range_z))

					if self.ui.n_meas_laser_spectra_track.isChecked():
						self.ui.shamrockPort.setCurrentIndex(1)
						self.shamrockSetWavelength(wl/3)



						print(self.piStage.MOV(bg_center,axis=b'1 2 3',waitUntilReady=True))

						for zi,z in enumerate(Range_z):
							self.piStage.MOV([z],axis=b'3',waitUntilReady=True)
							real_position = self.piStage.qPOS()
							print(real_position)
							self.setUiPiPos(real_position)
							pmt_valA, pmt_valB = self.readPico()
							data_Z[zi] = pmt_valB
							self.line_pmtB.setData(x=Range_z,y=data_Z)
						dz = medfilt(data_Z,9)
						interf_z = Range_z[dz==dz.max()][0]
						self.ui.n_meas_laser_spectra_Z_interface.setText(str(interf_z))

						z_offset = self.ui.n_meas_laser_spectra_Z_offset.value()

						np_scan_Z = interf_z + z_offset
						print(self.piStage.MOV([np_scan_Z],axis=b'3',waitUntilReady=True))


						centerA, centerB, panelA, panelB = self.center_optim(z_start=np_scan_Z, z_end=np_scan_Z+0.1, z_step=0.2)
						NP_centers = self.scan3D_peak_find()




					if not self.scan3DisAlive:
						break
					time_list.append(time.time())

					self.ui.n_meas_laser_spectra_probe.item(0,2).setText(str(np_scan_Z))
					bg_center = np.array([ float(self.ui.n_meas_laser_spectra_probe.item(0,i).text()) for i in range(3)])


					self.ui.shamrockPort.setCurrentIndex(0)
					self.shamrockSetWavelength((wl/2+wl/3)/2)

					self.piStage.MOV(bg_center,b' 1 2 3',waitUntilReady=True)
					pos = self.piStage.qPOS()
					self.setUiPiPos(pos=pos)
					self.andorCameraGetBaseline()

					for index,NP_c in enumerate(NP_centers):
						self.piStage.MOV(NP_c,b'1 2',waitUntilReady=True)
						pos = self.piStage.qPOS()
						self.setUiPiPos(pos=pos)

						if not self.scan3DisAlive:
							break
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
						integr_intens_SHG,intens,wavelength_arr = self.andorCameraGetData(state=1,integr_range=[wl/2-20,wl/2+20],index=index)
						time_list.append(time.time())
						w = (wavelength_arr>wl/3-20)&(wavelength_arr>wl/3+20)
						integr_intens_THG = intens[w].sum()
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
						store.put("spectra_"+str(wl)+'_NP'+str(index), df)
						store.put("time_"+str(wl), pd.DataFrame(time_list))
						if self.ui.n_meas_laser_spectra_track.isChecked():
							store.put("trackA_"+str(wl), panelA)
							store.put("trackB_"+str(wl), panelB)
							store.put("center_"+str(wl), pd.DataFrame(NP_centers))
							store.put("bg_center_"+str(wl), pd.DataFrame(bg_center))
							store.put("scan_Z_"+str(wl), pd.DataFrame(data_Z))




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


	def center_optim(self, z_start=None, z_end=None,z_step=None ):
		self.scan3DisAlive = True
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
				print(self.piStage.MOV(z,axis=3,waitUntilReady=True))

				forward = True
				for y,yi in zip(Range_y,Range_yi):
					if not self.scan3DisAlive: break
					Range_x_tmp = Range_x
					if forward:
						Range_x_tmp = Range_x[::-1]
						Range_xi_tmp = Range_xi[::-1]
						forward = False
					else:
						Range_x_tmp = Range_x
						Range_xi_tmp = Range_xi
						forward = True
					r = self.piStage.MOV(y,axis=2,waitUntilReady=True)
					if not r: break
					app.processEvents()
					for x,xi in zip(Range_x_tmp, Range_xi_tmp):

						start=time.time()
						#print('Start',start)
						if not self.scan3DisAlive: break
						r = self.piStage.MOV([x],axis=b'1',waitUntilReady=True)
						if not r: break

						#real_position0 = self.piStage.qPOS()
						pmt_valA, pmt_valB = self.readPico()
						data_pmtA[zi,xi,yi] = pmt_valA
						data_pmtB[zi,xi,yi] = pmt_valB
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



	def start3DScan(self, state):
		print(state)
		if state:
			try:
				self.live_pmtA = []
				self.live_pmtB = []
				self.live_x = []
				self.live_y = []

				self.scan3DisAlive = True
				self.scan3D()

			except:
				traceback.print_exc()
		else:
			self.live_pmtA = []
			self.live_pmtB = []
			self.live_x = []
			self.live_y = []

			self.scan3DisAlive = False

	def scan3D(self):
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
			data_pmtA = np.zeros((len(Range_z),len(Range_xi),len(Range_yi)))
			data_pmtB = np.zeros((len(Range_z),len(Range_xi),len(Range_yi)))
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
				for y,yi in zip(Range_y,Range_yi):
					if not self.scan3DisAlive: break
					Range_x_tmp = Range_x
					if forward:
						Range_x_tmp = Range_x[::-1]
						Range_xi_tmp = Range_xi[::-1]
						forward = False
					else:
						Range_x_tmp = Range_x
						Range_xi_tmp = Range_xi
						forward = True
					r = self.piStage.MOV(y,axis=2,waitUntilReady=True)
					if not r: break
					self.live_pmtA = np.array([])
					self.live_pmtB = np.array([])
					self.live_x = np.array([])
					self.live_y = np.array([])

					self.live_integr_spectra = []
					for x,xi in zip(Range_x_tmp, Range_xi_tmp):

						start=time.time()
						#print('Start',start)
						if not self.scan3DisAlive: break
						r = self.piStage.MOV([x],axis=b'1',waitUntilReady=wait)
						if not r: break

						real_position0 = self.piStage.qPOS()
						pmt_valA, pmt_valB = self.readPico()
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
							xi_ = int(round((self.live_x[-1]-Range_x[0])*len(Range_xi_tmp)/(Range_x_tmp.max()-Range_x_tmp.min())))
							yi_ = int(round((self.live_y[-1]-Range_y[0])*len(Range_yi)/(Range_y.max()-Range_y.min())))
							#data_spectra[yi_,xi_] = np.sum(spectra[s_from:s_to])
							if xi_>= len(Range_xi_tmp):
								xi_ = len(Range_xi_tmp)-1
							if yi_>= len(Range_yi):
								yi_ = len(Range_yi)-1
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
						app.processEvents()
						#time.sleep(0.01)
						wait = self.ui.Pi_wait.isChecked()
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
		imsave(fname+"_pmtA.tif",data_pmtA.astype(np.float32), imagej=True, resolution=(x_step*1e-4,y_step*1e-4,'cm'))
		imsave(fname+"_pmtB.tif",data_pmtB.astype(np.float32), imagej=True,resolution=(x_step*1e-4,y_step*1e-4,'cm'))

		self.ui.start3DScan.setChecked(False)
		#print(self.spectrometer.close())
		#print(self.piStage.CloseConnection())
	def start_fast3DScan(self,state):
		if state:
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
		self.pico_set()

		return data_pmtA, data_pmtB, (Range_x, Range_y, Range_z)

	def scan3D_peak_find(self):
		num_peaks = self.ui.scan3D_num_peaks.value()
		threshold_rel = self.ui.scan3D_threshold_rel.value()
		min_distance = self.ui.Scan3D_min_distance.value()
		peaks_A = peak_local_max(self.data2D_A, min_distance=min_distance, threshold_rel=threshold_rel,num_peaks=num_peaks)
		peaks_B = peak_local_max(self.data2D_B, min_distance=min_distance, threshold_rel=threshold_rel,num_peaks=num_peaks)

		Range_y = self.data2D_Range_y

		Range_x = self.data2D_Range_x

		centers = {}
		centers["A"] = np.array([Range_x[peaks_A[:,0]], Range_y[peaks_A[:,1]]]).T
		centers["B"] = np.array([Range_x[peaks_B[:,0]], Range_y[peaks_B[:,1]]]).T
		self.NP_centersA.setData(x=centers["A"][:,0],y=centers["A"][:,1])
		self.NP_centersB.setData(x=centers["B"][:,0],y=centers["B"][:,1])
		ch = self.ui.n_meas_laser_spectra_track_channel.currentIndex()
		ch_str = 'AB'
		#self.ui.n_meas_laser_spectra_probe.setRowCount(len(centers[ch_str[ch]])+1)
		interf_z = float(self.ui.n_meas_laser_spectra_Z_interface.text())

		z_offset = self.ui.n_meas_laser_spectra_Z_offset.value()

		np_scan_Z = abs(interf_z + z_offset)
		for i in range(1,len(centers[ch_str[ch]])+1):
			if i<=len(centers[ch_str[ch]]):
				c_coord = centers[ch_str[ch]][i-1]
			else:
				c_coord = [np.nan]*2
			self.ui.n_meas_laser_spectra_probe.item(i,0).setText(str(c_coord[0]))
			self.ui.n_meas_laser_spectra_probe.item(i,1).setText(str(c_coord[1]))
			self.ui.n_meas_laser_spectra_probe.item(i,2).setText(str(np_scan_Z))
		print(centers)

		return centers[ch_str[ch]]


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
		fname = self.ui.scan1D_filePath.text()+"_"+str(round(time.time()))+".txt"
		with open(fname,'a') as f:
			f.write("#X\tY\tZ\tHWP_power\tHWP_stepper\tpmtA_signal\tpmtB_signal\ttime\n")
		axis = self.ui.scan1D_axis.currentText()
		move_function = None
		if axis == "X":
			move_function = lambda pos: self.piStage.MOV(pos,axis=1,waitUntilReady=True)
		elif axis == "Y":
			move_function = lambda pos: self.piStage.MOV(pos,axis=2,waitUntilReady=True)
		elif axis == "Z":
			move_function = lambda pos: self.piStage.MOV(pos,axis=3,waitUntilReady=True)
		elif axis == 'HWP_Power':
			def move_function(pos):
				self.HWP.mAbs(pos)
				pos = self.HWP.getPos()
				self.ui.HWP_angle.setText(str(round(pos,6)))
		elif axis == 'HWP_stepper':
			def move_function(pos):
				self.HWP_stepper.moveTo(float(pos),wait=True)
				pos = self.HWP_stepper.getAngle()
				self.ui.HWP_stepper_angle.setText(str(round(pos,6)))

		steps_range = np.arange(self.ui.scan1D_start.value(),
								self.ui.scan1D_end.value(),
								self.ui.scan1D_step.value())
		for new_pos in steps_range:
			if not self.alive: break
			pmt_valA,pmt_valB = self.readPico()#self.readDAQmx(print_dt=True)

			real_position = [round(p,4) for p in self.piStage.qPOS()]
			HWP_angle = float(self.ui.HWP_angle.text())
			HWP_stepper_angle = float(self.ui.HWP_stepper_angle.text())

			if axis == 'X':
				x = real_position[0]
			if axis == 'Y':
				x = real_position[1]
			if axis == 'Z':
				x = real_position[2]
			if axis == 'HWP_Power':
				x = HWP_angle
			if axis == 'HWP_stepper':
				x = HWP_stepper_angle

			self.live_pmtA = np.hstack((self.live_pmtA, pmt_valA))
			self.live_pmtB = np.hstack((self.live_pmtB, pmt_valB))
			self.live_x = np.hstack((self.live_x, x))

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
			self.scan1D()

		else:
			self.live_pmtA = np.array([])
			self.live_pmtB = np.array([])
			self.live_x = np.array([])
			self.alive = False
			#self.rotPiezoStage.stop()

	############################################################################
	##########################   Ui   ##########################################
	def initUI(self):
		self.laserStatus.start(1000)

		self.ui.actionExit.toggled.connect(self.closeEvent)

		self.ui.scan3D_config.cellChanged[int,int].connect(self.syncRectROI_table)
		self.ui.meas_laser_spectra_probe.cellChanged[int,int].connect(self.meas_laser_spectra_probe_update)

		self.ui.scan1D_filePath_find.clicked.connect(self.scan1D_filePath_find)

		self.ui.scan3D_path_dialog.clicked.connect(self.scan3D_path_dialog)

		self.ui.scan3D_peak_find.clicked.connect(self.scan3D_peak_find)

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



		self.ui.HWP_stepper_Connect.toggled[bool].connect(self.HWP_stepper_Connect)
		self.ui.HWP_stepper_MoveTo_Go.clicked.connect(self.HWP_stepper_MoveTo_Go)
		self.ui.HWP_stepper_CW.clicked.connect(self.HWP_stepper_CW)
		self.ui.HWP_stepper_CCW.clicked.connect(self.HWP_stepper_CCW)
		self.ui.HWP_stepper_Reset.clicked.connect(self.HWP_stepper_Reset)


		self.ui.pm100Connect.toggled[bool].connect(self.pm100Connect)
		self.ui.pm100Average.valueChanged[int].connect(self.pm100Average)

		self.ui.confParam_scan.toggled[bool].connect(self.confParam_scan)

		self.ui.meas_laser_spectra_go.toggled[bool].connect(self.meas_laser_spectra_go)
		self.ui.n_meas_laser_spectra_go.toggled[bool].connect(self.n_meas_laser_spectra_go)

		########################################################################
		########################################################################
		########################################################################


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

		self.tabColors = {
			0: 'green',
			1: 'red',
			2: 'wine',
			3: 'orange',
			4: 'blue',
			5: 'cyan',
			6: 'magenta'
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
			self.line_spectra.append(self.pw_spectra.plot(pen=(cr[::10][i],cr[::-10][i],cr[::5][i])))

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

		self.img.addItem(self.spectra_bg_probe)
		self.img.addItem(self.spectra_signal_probe)



		'''
		x_arrow = pg.ArrowItem(angle=180, tipAngle=30, baseAngle=20, headLen=20, tailLen=40, tailWidth=2, pen=None, brush='r')
		x_arrow.setPos(40,0)
		self.img.addItem(x_arrow)
		t1 = pg.TextItem('X')
		self.img.addItem(t1)
		t1.setPos(40,-10)
		t2 = pg.TextItem('Y')
		self.img.addItem(t2)
		t2.setPos(-10,40)

		y_arrow = pg.ArrowItem(angle=90, tipAngle=30, baseAngle=20, headLen=20, tailLen=40, tailWidth=2, pen=None, brush='g')
		y_arrow.setPos(0,40)
		self.img.addItem(y_arrow)
		'''


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

		self.img1.addItem(self.spectra_bg_probe1)
		self.img1.addItem(self.spectra_signal_probe1)
		self.img1.addItem(self.NP_centersB)
		self.move_scat_flag = False
		self.move_scat_item = None

		def onImgScatClick(event):
			#event.accept()
			items = self.img.scene.items(event.scenePos())
			pos = event.scenePos()
			#print(event,event.button, event.pos(), pos)
			items_id = [id(i) for i in items]

			if id(self.spectra_bg_probe) in items_id:
				print('bg')
				self.move_scat_item = self.spectra_bg_probe

			if id(self.spectra_signal_probe) in items_id:
				print('NP1')
				self.move_scat_item = self.spectra_signal_probe

			modifiers = QtGui.QApplication.keyboardModifiers()
			if modifiers & QtCore.Qt.ControlModifier:
				if not self.move_scat_item is None:
					print(pos)
					self.move_scat_item.setData(x=[pos.x()],y=[pos.y()])
					self.move_scat_item = None

		self.img.scene.sigMouseClicked.connect(onImgScatClick)

		def onImg1ScatClick(event):
			#event.accept()
			items = self.img1.scene.items(event.scenePos())
			pos = event.scenePos()
			print(pos)
			#print(event,event.button, event.pos(), pos)
			items_id = [id(i) for i in items]
			if id(self.spectra_bg_probe1) in items_id:
				print('bg')
				self.move_scat_item = self.spectra_bg_probe1

			if id(self.spectra_signal_probe1) in items_id:
				print('NP1')
				self.move_scat_item = self.spectra_signal_probe1
			modifiers = QtGui.QApplication.keyboardModifiers()
			if modifiers & QtCore.Qt.ControlModifier:
				if not self.move_scat_item is None:

					self.move_scat_item.setData(x=[pos.x()],y=[pos.y()])

					if id(self.move_scat_item) == id(self.spectra_bg_probe1):
						self.ui.n_meas_laser_spectra_probe.item(0,0).setText(str(pos.x()))
						self.ui.n_meas_laser_spectra_probe.item(0,1).setText(str(pos.y()))
					if id(self.move_scat_item) == id(self.spectra_signal_probe1):
						self.ui.n_meas_laser_spectra_probe.item(1,0).setText(str(pos.x()))
						self.ui.n_meas_laser_spectra_probe.item(1,1).setText(str(pos.y()))
						#self.piStage.MOV([pos.x(),pos.y()],b'1 2', waitUntilReady=1)
						real_position = self.piStage.qPOS()
						self.setUiPiPos(real_position)
					self.move_scat_item = None

		self.img1.scene.sigMouseClicked.connect(onImg1ScatClick)

		self.statusBar_Position = QtGui.QLabel('[nan nan nan]')
		self.ui.statusbar.addWidget(self.statusBar_Position)


		#self.ui.configTabWidget.setStyleSheet('QTabBar::tab[objectName="Readout"] {background-color=red;}')
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

	def closeEvent(self, evnt):
		print('closeEvent')
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
