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

import matplotlib
import scipy.misc
#get_ipython().run_line_magic('matplotlib', 'qt')


class microV(QtGui.QMainWindow):
	data = []
	alive = True
	piStage = E727()
	spectrometer = CCS200()
	HWP = APTMotor(83854487, HWTYPE=31)
	DAQmx = nidaqmx.Task()
	#DAQmx = MultiChannelAnalogInput(["Dev1/ai0,Dev1/ai2"])
	live_pmt = []
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
		self.initDAQmx()

		self.calibrTimer = QtCore.QTimer()
		self.initUI()

		#self.scan_image()
	def initDAQmx(self):

		self.DAQmx.ai_channels.add_ai_voltage_chan("Dev1/ai0,Dev1/ai2", max_val=10, min_val=-10)

		self.DAQmx.timing.cfg_samp_clk_timing(1000000)

		self.DAQmx.control(TaskMode.TASK_COMMIT)

		self.DAQmx.triggers.start_trigger.cfg_dig_edge_start_trig("PFI0")

		#self.DAQmx.configure()
	def readDAQmx(self):
		start = time.time()
		try:
			self.DAQmx.start()
			master_data = self.DAQmx.read(number_of_samples_per_channel=200)
			self.DAQmx.wait_until_done(0.2)
			self.DAQmx.stop()
		except:
			return -1
		r,d = master_data
		r = np.array(r)
		d = np.array(d)
		#pp.pprint(master_data)
		print("readDAQmx(tdiff):\n", time.time()-start)
		DATA_SHIFT=6
		w = r>r.mean()
		out = abs(d[w].mean()-d[~w].mean())
		return out

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
		print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=1,waitUntilReady=True))
		time.sleep(0.2)
		print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=2,waitUntilReady=True))
		time.sleep(0.2)
		print(self.piStage.MOV(self.ui.Pi_X_move_to.value(),axis=3,waitUntilReady=True))
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
			self.live_integr_spectra = []
		else:
			self.calibrTimer.stop()

	def onCalibrTimer(self):
		self.calibrTimer.stop()
		spectra = self.getSpectra()
		pmt_val = self.readDAQmx()
		self.live_pmt.append(pmt_val)
		s_from = self.ui.usbSpectr_from.value()
		s_to = self.ui.usbSpectr_to.value()
		self.live_integr_spectra.append(np.sum(spectra[s_from:s_to])/1000)
		self.setLine(spectra)
		self.line_pmt.setData(self.live_pmt)
		self.line_spectra.setData(self.live_integr_spectra)

		self.calibrTimer.start(self.ui.usbSpectr_integr_time.value()*2000)
		app.processEvents()

	def initUI(self):
		self.ui.actionExit.triggered.connect(self.closeEvent)

		self.ui.scan3D_path_dialog.clicked.connect(self.scan3D_path_dialog)

		self.ui.usbSpectr_set_integr_time.clicked.connect(self.usbSpectr_set_integr_time)

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

		self.pw1 = pg.PlotWidget(name='Graph1')  ## giving the plots names allows us to link their axes together
		self.ui.plotView.addWidget(self.pw1)

		self.line_pmt = self.pw1.plot(pen=(255,0,0))
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
					f.write("#X\tY\tZ\tpmt_signal\tspectra_px[" +
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
					self.live_integr_spectra = []
					for x,xi in zip(Range_x_tmp, Range_xi_tmp):

						start=time.time()
						print('Start',start)
						if not self.scan3DisAlive: break
						r = self.piStage.MOV(x,axis=1,waitUntilReady=True)
						if not r: break
						print(time.time()-start)
						spectra = self.getSpectra()
						print(time.time()-start)
						pmt_val = self.DAQmx.getData()
						print(time.time()-start,pmt_val)
						#spectra = list(medfilt(spectra,5))
						real_position = [round(p,4) for p in self.piStage.qPOS()]
						dataSet = real_position +[pmt_val] + spectra[spectra_range[0]:spectra_range[1]]
						#print(dataSet[-1])
						with open(fname,'a') as f:
							f.write("\t".join([str(round(i,4)) for i in dataSet])+"\n")
						print(time.time()-start)
						s_from = self.ui.usbSpectr_from.value()
						s_to = self.ui.usbSpectr_to.value()
						print(data_spectra.shape,yi,xi)
						data_spectra[yi,xi] = np.sum(spectra[s_from:s_to])
						data_pmt[yi,xi] = pmt_val
						self.live_pmt.append(pmt_val)
						self.live_integr_spectra.append(np.sum(spectra[s_from:s_to]))
						if forward:
							self.line_pmt.setData(self.live_pmt)
						else:
							self.line_pmt.setData(self.live_pmt[::-1])
						#self.line_spectra.setData(self.live_integr_spectra)

						self.setLine(spectra)
						print(time.time()-start)
						self.setUiPiPos(real_position)
						print(real_position)
						app.processEvents()
						#time.sleep(0.001)
						print(time.time()-start)
					self.setImage(data_spectra)
					self.setImage1(data_pmt)

				scipy.misc.toimage(data_spectra).save(fname+".png")
				scipy.misc.toimage(data_pmt).save(fname+"_pmt.png")
		except KeyboardInterrupt:
			scipy.misc.toimage(data_spectra).save(fname+".png")
			scipy.misc.toimage(data_pmt).save(fname+"_pmt.png")
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
