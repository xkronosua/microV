import sys,time
import qdarkstyle
from pyqtgraph.Qt import QtGui, QtCore, uic
import traceback

import pyqtgraph as pg
import numpy as np
from multiprocessing import Process
from scipy.signal import medfilt
if len(sys.argv)>1:
	if sys.argv[1] == 'sim':
		from hardware.sim.CCS200 import *
		from hardware.sim.ni import *
		from hardware.sim.E727 import *
		from hardware.sim.TDC001 import *
else:
	from hardware.ni1 import *
	from hardware.CCS200 import *
	from hardware.E727 import *
	from hardware.TDC001 import *
	from nidaqmx.constants import AcquisitionType, TaskMode
	import nidaqmx
	from picoscope import ps3000a

from scipy.signal import argrelextrema

import matplotlib
import scipy.misc
#get_ipython().run_line_magic('matplotlib', 'qt')


class microV(QtGui.QMainWindow):
	data = []
	alive = True
	piStage = E727()
	spectrometer = CCS200()
	HWP = APTMotor(83854487, HWTYPE=31)
	#DAQmx = nidaqmx.Task()
	#DAQmx = MultiChannelAnalogInput(["Dev1/ai0,Dev1/ai2"])
	live_pmt = []
	live_pmt1 = []
	live_integr_spectra = []
	def __init__(self, parent=None):
		QtGui.QMainWindow.__init__(self, parent)
		#from mainwindow import Ui_mw
		self.ui = uic.loadUi("microV.ui")#Ui_mw()
		self.ui.closeEvent = self.closeEvent
		self.ui.show()
		self._want_to_close = False
		self.initPiStage()
		self.initSpectrometer()
		self.initHWP()
		self.initPico()
		#self.initDAQmx()

		self.calibrTimer = QtCore.QTimer()
		self.initUI()

		#self.scan_image()
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

	def initPico(self):
		self.ps = ps3000a.PS3000a(connect=False)
		self.ps.open()
		self.n_captures = 100
		self.pico_ranges = [0.02,0.05,0.1,0.2,0.5,1,2,5,10,20]
		self.pico_Vrange_index = [5,5]
		self.ps.setChannel("A", coupling="DC", VRange=self.pico_ranges[self.pico_Vrange_index[0]])
		self.ps.setChannel("B", coupling="DC", VRange=self.pico_ranges[self.pico_Vrange_index[1]])
		self.ps.setSamplingInterval(200e-9,15e-6)
		self.ps.setSimpleTrigger(trigSrc="External", threshold_V=0.020, direction='Rising',
								 timeout_ms=5, enabled=True)
		self.samples_per_segment = self.ps.memorySegments(self.n_captures)
		self.ps.setNoOfCaptures(self.n_captures)


	def readPico(self):
		dataA = np.zeros((self.n_captures, self.samples_per_segment), dtype=np.int16)
		dataB = np.zeros((self.n_captures, self.samples_per_segment), dtype=np.int16)
		#t1 = time.time()
		self.ps.runBlock()
		self.ps.waitReady()
		#t2 = time.time()
		#print("Time to get sweep: " + str(t2 - t1))
		self.ps.getDataRawBulk(channel='A',data=dataA)
		self.ps.getDataRawBulk(channel='B',data=dataB)
		#t3 = time.time()
		#print("Time to read data: " + str(t3 - t2))
		dataA = dataA[:, 0:self.ps.noSamples].mean(axis=0)
		dataB = dataB[:, 0:self.ps.noSamples].mean(axis=0)

		if self.ui.DAQmx_preview.isChecked():
			self.line_DAQmx_sig.setData(dataA)
			self.line_DAQmx_sig1.setData(dataB)

		index = self.pico_Vrange_index
		if dataA.min()<-20000 and index[0]<10:
			index[0] = index[0] + 1
		elif dataA.min()>-200 and index[0]>0:
			index[0] = index[0] - 1

		if dataB.min()<-20000 and index[1]<10:
			index[1] = index[1] + 1
		elif dataB.min()>-200 and index[1]>0:
			index[1] = index[1] - 1

		self.pico_Vrange_index = index
		self.ps.setChannel("A", coupling="DC", VRange=self.pico_ranges[self.pico_Vrange_index[0]])
		self.ps.setChannel("B", coupling="DC", VRange=self.pico_ranges[self.pico_Vrange_index[1]])

		dataA = self.ps.rawToV( channel="A", dataRaw=dataA)
		dataB = self.ps.rawToV(channel="B", dataRaw=dataB)

		dataA_p2p = abs(dataA.max() - dataA.min())
		dataB_p2p = abs(dataB.max() - dataB.min())


		return dataA_p2p, dataB_p2p


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

	def initPiStage(self):
		print(self.piStage.ConnectUSB())
		print(self.piStage.qSAI())
		print(self.piStage.SVO())
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
		print(self.piStage.MOV(50,axis=1,waitUntilReady=True))
		time.sleep(0.2)
		print(self.piStage.MOV(50,axis=2,waitUntilReady=True))
		time.sleep(0.2)
		print(self.piStage.MOV(50,axis=3,waitUntilReady=True))
		pos = self.piStage.qPOS()
		self.setUiPiPos(pos=pos)
		time.sleep(1)

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

	def scan3D_path_dialog(self):
		fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', self.ui.scan3D_path.text())
		try:
			self.ui.scan3D_path.setText(fname)
		except:
			traceback.print_exc()
	def startCalibr(self,state):
		if state:
			self.calibrTimer.start(self.ui.usbSpectr_integr_time.value()*2000)
			self.live_pmt = []
			self.live_pmt1 = []
			self.live_integr_spectra = []
		else:
			self.calibrTimer.stop()

	def onCalibrTimer(self):
		self.calibrTimer.stop()
		spectra = np.zeros(3648)#self.getSpectra()
		pmt_val,pmt_val1 = self.readPico()#self.readDAQmx(print_dt=True)
		self.live_pmt.append(pmt_val)
		self.live_pmt1.append(pmt_val1)
		s_from = self.ui.usbSpectr_from.value()
		s_to = self.ui.usbSpectr_to.value()
		self.live_integr_spectra.append(np.sum(spectra[s_from:s_to])/1000)
		self.setLine(spectra)
		self.line_pmt.setData(self.live_pmt)
		self.line_pmt1.setData(self.live_pmt1)
		#self.line_spectra.setData(self.live_integr_spectra)

		self.calibrTimer.start(self.ui.usbSpectr_integr_time.value()*2000)
		app.processEvents()

	def initUI(self):
		self.ui.actionExit.triggered.connect(self.closeEvent)

		self.ui.scan3D_path_dialog.clicked.connect(self.scan3D_path_dialog)

		self.ui.usbSpectr_set_integr_time.clicked.connect(self.usbSpectr_set_integr_time)

		self.ui.DAQmx_find_shift.clicked.connect(self.optimizeDAQmx)

		self.ui.HWP_go.clicked.connect(self.HWP_go)
		self.ui.HWP_go_home.clicked.connect(self.HWP_go_home)
		self.ui.HWP_negative_step.clicked.connect(self.HWP_negative_step)
		self.ui.HWP_positive_step.clicked.connect(self.HWP_positive_step)

		self.ui.Pi_X_go.clicked.connect(self.Pi_X_go)
		self.ui.Pi_Y_go.clicked.connect(self.Pi_Y_go)
		self.ui.Pi_Z_go.clicked.connect(self.Pi_Z_go)
		self.ui.Pi_XYZ_50mkm.clicked.connect(self.Pi_XYZ_50mkm)

		self.ui.start3DScan.toggled[bool].connect(self.start3DScan)

		self.ui.startCalibr.toggled[bool].connect(self.startCalibr)
		self.calibrTimer.timeout.connect(self.onCalibrTimer)

		self.pw = pg.PlotWidget(name='PlotMain')  ## giving the plots names allows us to link their axes together
		self.ui.plotView.addWidget(self.pw)
		self.line0 = self.pw.plot()
		self.line1 = self.pw.plot(pen=(255,0,0))
		self.line3 = self.pw.plot(pen=(255,0,255))

		self.pw_DAQmx = pg.PlotWidget(name='DAQmx')  ## giving the plots names allows us to link their axes together
		self.ui.DAQmx_plot.addWidget(self.pw_DAQmx)

		self.line_DAQmx_sig = self.pw_DAQmx.plot(pen=(255,0,0))
		self.line_DAQmx_sig1 = self.pw_DAQmx.plot(pen=(255,255,0))
		self.line_DAQmx_ref = self.pw_DAQmx.plot(pen=(255,0,255))

		self.pw1 = pg.PlotWidget(name='Graph1')  ## giving the plots names allows us to link their axes together
		self.ui.plotView.addWidget(self.pw1)

		self.line_pmt = self.pw1.plot(pen=(255,0,0))
		self.line_pmt1 = self.pw1.plot(pen=(255,255,0))
		self.line_spectra = self.pw1.plot(pen=(0,255,0))

		self.img = pg.ImageView()  ## giving the plots names allows us to link their axes together
		data = np.zeros((100,100))
		self.ui.imageView.addWidget(self.img)
		self.img.setImage(data)

		self.img1 = pg.ImageView()  ## giving the plots names allows us to link their axes together
		data = np.zeros((100,100))
		self.ui.imageView.addWidget(self.img1)
		self.img1.setImage(data)


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
			self.DAQmx.close()
		except:
			traceback.print_exc()
		try:
			self.ps.close()
		except:
			traceback.print_exc()


	def setImage(self,data):
		self.img.setImage(data.T,levels=(data.min(), data.max()))
		#app.processEvents()

	def setImage1(self,data):
		self.img1.setImage(data.T,levels=(data.min(), data.max()))
		#app.processEvents()

	def setLine(self,data,skip=1):
		self.line0.setData(data[::skip])
		#app.processEvents()

	def start3DScan(self, state):
		print(state)
		if state:
			try:
				self.scan3DisAlive = True
				p = Process(target=self.scan3D())
				p.daemon = True
				p.start()

			except AttributeError:
				traceback.print_exc()
		else:
			self.scan3DisAlive = False


	def scan3D(self):
		path = self.ui.scan3D_path.text()
		if "_" in path:
			path = "".join(path.split("_")[:-1])#+"_"+str(round(time.time()))
		else:
			path = path #+ "_"+str(round(time.time()))

		self.ui.scan3D_path.setText(path)
		spectra_range = [self.ui.usbSpectr_save_from.value(),self.ui.usbSpectr_save_to.value()]
		try:
			z_start = float(self.ui.scan3D_config.item(0,0).text())
			z_end = float(self.ui.scan3D_config.item(0,1).text())
			z_step = float(self.ui.scan3D_config.item(0,2).text())

			for z in np.arange(z_start,z_end,z_step):
				if not self.scan3DisAlive: break
				print(self.piStage.MOV(z,axis=3,waitUntilReady=True))
				time.sleep(1)

				fname = path+"Z"+str(z)+"_"+str(round(time.time()))+'.txt'
				with open(fname,'a') as f:
					f.write("#X\tY\tZ\tpmt_signal\tpmt1_signal\tspectra_px[" +
						str(spectra_range[0]) + ":"+
						str(spectra_range[1]) +"]\n")




				y_start = float(self.ui.scan3D_config.item(1,0).text())
				y_end = float(self.ui.scan3D_config.item(1,1).text())
				y_step = float(self.ui.scan3D_config.item(1,2).text())

				Range_y = np.arange(y_start,y_end,y_step)
				Range_yi = np.arange(len(Range_y))


				x_start = float(self.ui.scan3D_config.item(2,0).text())
				x_end = float(self.ui.scan3D_config.item(2,1).text())
				x_step = float(self.ui.scan3D_config.item(2,2).text())

				Range_x = np.arange(x_start,x_end,x_step)
				Range_xi = np.arange(len(Range_x))

				data_spectra = np.zeros((len(Range_yi),len(Range_xi)))
				data_pmt = np.zeros((len(Range_yi),len(Range_xi)))
				data_pmt1 = np.zeros((len(Range_yi),len(Range_xi)))

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
					self.live_pmt = []
					self.live_pmt1 = []
					self.live_integr_spectra = []
					for x,xi in zip(Range_x_tmp, Range_xi_tmp):

						start=time.time()
						#print('Start',start)
						if not self.scan3DisAlive: break
						r = self.piStage.MOV(x,axis=1,waitUntilReady=True)
						if not r: break
						#print(time.time()-start)
						spectra = np.zeros(3648)#self.getSpectra()
						#print(time.time()-start)
						pmt_val,pmt_val1 = self.readPico()#readDAQmx()
						#print(time.time()-start,pmt_val)
						#spectra = list(medfilt(spectra,5))
						real_position = [round(p,4) for p in self.piStage.qPOS()]
						dataSet = real_position +[pmt_val,pmt_val1] + spectra[spectra_range[0]:spectra_range[1]]
						#print(dataSet[-1])
						with open(fname,'a') as f:
							f.write("\t".join([str(round(i,4)) for i in dataSet])+"\n")
						#print(time.time()-start)
						s_from = self.ui.usbSpectr_from.value()
						s_to = self.ui.usbSpectr_to.value()
						#print(data_spectra.shape,yi,xi)
						data_spectra[yi,xi] = np.sum(spectra[s_from:s_to])
						data_pmt[yi,xi] = pmt_val
						data_pmt1[yi,xi] = pmt_val1
						self.live_pmt.append(pmt_val)
						self.live_pmt1.append(pmt_val1)
						self.live_integr_spectra.append(np.sum(spectra[s_from:s_to]))
						if forward:
							self.line_pmt.setData(self.live_pmt)
							self.line_pmt1.setData(self.live_pmt1)
						else:
							self.line_pmt.setData(self.live_pmt[::-1])
							self.line_pmt1.setData(self.live_pmt1[::-1])
						#self.line_spectra.setData(self.live_integr_spectra)

						self.setLine(spectra)
						#print(time.time()-start)
						self.setUiPiPos(real_position)
						print("[\t%.5f\t%.5f\t%.5f\t]\t%.5f"%tuple(real_position+[time.time()-start]))
						app.processEvents()
						time.sleep(0.01)
						#print(time.time()-start)
					#self.setImage(data_spectra)
					self.setImage1(data_pmt1)
					self.setImage(data_pmt)

				#scipy.misc.toimage(data_spectra).save(fname+".png")
				scipy.misc.toimage(data_pmt1).save(fname+"_pmt1.png")
				scipy.misc.toimage(data_pmt).save(fname+"_pmt.png")
		except KeyboardInterrupt:
			#scipy.misc.toimage(data_spectra).save(fname+".png")
			scipy.misc.toimage(data_pmt).save(fname+"_pmt.png")
			scipy.misc.toimage(data_pmt1).save(fname+"_pmt1.png")
			print(self.spectrometer.close())
			print(self.piStage.CloseConnection())
			return
		self.ui.start3DScan.setChecked(False)
		#print(self.spectrometer.close())
		#print(self.piStage.CloseConnection())




if __name__ == '__main__':


	app = QtGui.QApplication(sys.argv)
	app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
	ex = microV()
	app.exec_()
