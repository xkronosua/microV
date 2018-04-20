import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic, QtCore, QtGui
import visa
import time
import datetime
import re
from threading import RLock, Thread
import traceback

class laserDaemon(QWidget):
	resMan = visa.ResourceManager()
	Laser = None
	command = ''
	prev_command = ''
	wavelength_ready = False
	prev_power = 0
	current_power = 0
	def __init__(self,parent=None):
		QWidget.__init__(self, parent)
		self.ui = uic.loadUi('laser.ui', self)  # Loads all widgets of uifile.ui into self
		self.watchdogTimer = QtCore.QTimer()
		self.lock = RLock()
		with open('laserIn','w+') as f:
			f.write('')
		with open('laserOut','w+') as f:
			f.write('')
		self.initUI()


	def initUI(self):
		self.ui.setWindowTitle('Laser Daemon')
		self.ui.connect.toggled[bool].connect(self.connectLaser)
		self.ui.onOff.toggled[bool].connect(self.laserOnOff)
		self.ui.switchShutter.clicked.connect(self.setShutter)
		self.watchdogTimer.timeout.connect(self.onWatchdogTimerTimeout)
		self.ui.wavelength.editingFinished.connect(self.setWavelength)
		#self.ui.wavelength.valueChanged[int].connect(self.blockWavelengthEdit)

		self.ui.show()

	def statusLog(self,text, dir=''):
		out = ''
		ts = time.time()
		ds  = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
		if dir == '>>':
			out = ds + " >> "+text
		if dir == '<<':
			out = ds + " << "+text
		else:
			out = ds + " : "+text
		self.ui.laserLog.appendPlainText(out)
		print(out)



	def closeEvent(self, evnt=None):
		print('closeEvent')
		if self.Laser:
			self.Laser.close()
			self.statusLog('close')

	def onWatchdogTimerTimeout(self):
		with self.lock:
			laserPower = self.Laser.query("READ:POWer?").replace('\n','')
			self.ui.powerMeter.setValue(float(laserPower))
			self.ui.power.setText(laserPower+' W')
			wavelength = self.Laser.query("READ:WAVelength?").replace('\n','')
			if self.ui.wavelength.value()!= int(wavelength):
				if not self.ui.wavelength.signalsBlocked():
					self.ui.wavelength.setValue(float(wavelength))
			shutter = self.Laser.query("SHUTter?").replace('\n','')
			self.ui.shutter.setChecked(shutter=='1')
			status = self.Laser.query("*STB?").replace('\n','')
			v = int(status)
			shutter_ = v.to_bytes(32,byteorder='big')[2]
			emission = v.to_bytes(32,byteorder='big')[0]
			print(shutter,shutter_,emission)

			val = [time.time(), status, shutter, laserPower,wavelength,self.wavelength_ready]
			print(val)
			out = '\t'.join([str(i) for i in val])
			#self.ui.wavelength.setValue(float(wavelength))
			with open('laserOut','w+') as f:
				f.write(out)
			with open('laserIn','r+') as f:
				c = f.read()
				if c != self.prev_command:
					self.prev_command = self.command
					self.command = c
					if re.match('WAVelength [0-9]{3,4}\n', c):
						wl = int(self.command.split(' ')[-1])
						self.setWavelength(wl)
						print(wl)
					elif re.match('SHUTter [0-1]\n', c):
						sh = int(self.command.split(' ')[-1])
						self.setShutter(sh,manual=True)

			print(self.command)

		#self.statusLog(warmedup,dir='<<')

	def connectLaser(self,state):
		if state:
			self.statusLog('connect')
			self.Laser = self.resMan.open_resource('ASRL'+str(self.ui.port.value())+'::INSTR', baud_rate = 115200)
			self.statusLog("*IDN?",dir='>>')
			idn = self.Laser.query("*IDN?")
			self.statusLog(idn,dir='<<')

			self.statusLog("TIMer:WATChdog 10",dir='>>')
			with self.lock:
				self.Laser.write("TIMer:WATChdog 10")
			self.watchdogTimer.start(2000)
		else:
			self.Laser.close()
			self.statusLog('close')
			self.watchdogTimer.stop()

	def laserOnOff(self,state):
		if state:
			self.statusLog("ON",dir='>>')
			self.Laser.write("ON")
			t0 = time.time()
			#while time.time()-t0<120:
			#	time.sleep(0.1)
			#	app.processEvents()
			warmedup = 0
			while warmedup < 100:
				with self.lock:
					warmedup = self.Laser.query("READ:PCTWarmedup?")
				self.statusLog(warmedup,dir='<<')
				warmedup = int(warmedup)
				time.sleep(0.2)
				app.processEvents()
			t0 = time.time()
			on = '0'
			while time.time()-t0<130:
				try:
					with self.lock:
						on = self.Laser.query("ON")
				except:
					pass
				time.sleep(0.2)
				print(on)
				app.processEvents()
			print('ready')
			try:
				self.statusLog("ON",dir='>>')
				with self.lock:
					on = self.Laser.query("ON")
				self.statusLog(on,dir='<<')
			except:
				pass
		else:
			self.statusLog("OFF",dir='>>')
			self.Laser.write("OFF")
	def getPower(self):
		laserPower = 0
		with self.lock:
			laserPower = float(self.Laser.query("READ:POWer?").replace('\n',''))
		self.ui.power.setText(str(laserPower)+' W')
		self.ui.powerMeter.setValue(laserPower)
		return laserPower

	def getWavelength(self):
		wavelength = 0
		with self.lock:
			wavelength = float(self.Laser.query("READ:WAVelength?").replace('\n',''))
		self.ui.wavelength.blockSignals(True)
		self.ui.wavelength.setValue(wavelength)
		self.ui.wavelength.blockSignals(False)
		return wavelength

	def blockWavelengthEdit(self,val):
		self.ui.wavelength.blockSignals(True)
		def unlock(f):
			for i in range(50):
				print('locked')
				time.sleep(0.1)
			f(False)
		t = Thread(target=unlock,args=[self.ui.wavelength.blockSignals,])
		t.start()
	def setWavelength(self):

		self.ui.wavelength.blockSignals(True)
		val = self.ui.wavelength.value()
		 self.wavelength_ready = False
		if val>=680 and val<=1300:
			self.statusLog("WAVelength "+str(val),dir='>>')
			try:
				prev_power = self.getPower()
				with self.lock:
					self.Laser.write("WAVelength "+str(val))

				t0 = time.time()
				while time.time()-t0<60 and not self.wavelength_ready:
					time.sleep(0.1)
					new_power = self.getPower()
					new_wavelength = self.getWavelength()

					self.wavelength_ready = (new_power == prev_power and new_wavelength == val)
					prev_power = new_power
					print(time.time()-t0, self.wavelength_ready)
			except:
				traceback.print_exc()
				self.statusLog("err",dir='<<')

		self.ui.wavelength.blockSignals(False)

	def setShutter(self,state=None,manual=False):
		if not manual:
			print('int')
			state = self.ui.shutter.isChecked()
			if state > 0:
				state = '0'
			else:
				state = '1'
		else:
			if state > 0:
				state = '1'
			else:
				state = '0'
		self.statusLog("SHUTter "+state,dir='>>')
		try:
			with self.lock:
				self.Laser.write("SHUTter "+state)
				self.ui.shutter.setChecked(state!='0')
			#self.statusLog(wavelength,dir='<<')
		except:
			self.statusLog("err",dir='<<')

		#for i in range(5):
		#	self.Laser.write("*")
		#self.statusLog(wavelength,dir='<<')

if __name__ == '__main__':
	import os
	app = QApplication(sys.argv)
	ex = laserDaemon()
	pid = os.getpid()
	with open('laserPid','w+') as f:
		f.write(str(pid))
	sys.exit(app.exec_())
