import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic, QtCore, QtGui
import visa
import time
import datetime
import re

class laserDaemon(QWidget):
	resMan = visa.ResourceManager()
	Laser = None
	command = ''
	prev_command = ''
	def __init__(self,parent=None):
		QWidget.__init__(self, parent)
		self.ui = uic.loadUi('laser.ui', self)  # Loads all widgets of uifile.ui into self
		self.watchdogTimer = QtCore.QTimer()
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
		self.ui.wavelength.valueChanged[int].connect(self.setWavelength)
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
		laserPower = self.Laser.query("READ:POWer?").replace('\n','')
		self.ui.powerMeter.setValue(float(laserPower))
		self.ui.power.setText(laserPower+' W')
		wavelength = self.Laser.query("READ:WAVelength?").replace('\n','')
		self.ui.wavelength.setValue(float(wavelength))
		shutter = self.Laser.query("SHUTter?").replace('\n','')

		status = self.Laser.query("*STB?").replace('\n','')
		v = int(status)
		shutter_ = v.to_bytes(32,byteorder='big')[2]
		emission = v.to_bytes(32,byteorder='big')[0]
		print(shutter,shutter_,emission)

		val = [time.time(), status, shutter, laserPower,wavelength]
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
					self.setShutter(sh)

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
				warmedup = self.Laser.query("READ:PCTWarmedup?")
				self.statusLog(warmedup,dir='<<')
				warmedup = int(warmedup)
				time.sleep(0.2)
				app.processEvents()
			t0 = time.time()
			on = '0'
			while time.time()-t0<130:
				try:
					on = self.Laser.query("ON")
				except:
					pass
				time.sleep(0.2)
				print(on)
				app.processEvents()
			print('ready')
			try:
				self.statusLog("ON",dir='>>')
				on = self.Laser.query("ON")
				self.statusLog(on,dir='<<')
			except:
				pass
		else:
			self.statusLog("OFF",dir='>>')
			self.Laser.write("OFF")

	def setWavelength(self,val):
		if val>=680 and val<=1300:
			self.statusLog("WAVelength "+str(val),dir='>>')
			try:
				self.Laser.write("WAVelength "+str(val))

			except:
				self.statusLog("err",dir='<<')
	def setShutter(self,state=None):
		if state is None:
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
