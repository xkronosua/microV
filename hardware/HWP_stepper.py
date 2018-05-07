import numpy as np
import time
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *
from threading import Timer, Thread

class HWP_stepper():
	""" Class to create a continuous pulse train on a counter

	Usage:  pulse = ContinuousTrainGeneration(period [s],
				duty_cycle (default = 0.5), counter (default = "dev1/ctr0"),
				reset = True/False)
			pulse.start()
			pulse.stop()
			pulse.clear()
	"""
	Direction = False
	Enable = False
	reset = False
	taskHandle = None
	taskHandleDirEnable = None
	steps_deg = 1.8
	calibr = 8.92*2
	currentAngle = 0
	def __init__(self, freq=10.,duty_cycle=0.5, counter="Dev1/ctr1", reset=False, enableTimeout=10):
		self.reset = reset
		if reset:
			self.currentAngle = 0
			DAQmxResetDevice(counter.split('/')[0])
		taskHandle = TaskHandle(0)
		DAQmxCreateTask("",byref(taskHandle))
		DAQmxCreateCOPulseChanFreq(taskHandle,counter,"",DAQmx_Val_Hz,DAQmx_Val_Low,
																   0.0,freq,duty_cycle)


		self.taskHandle = taskHandle

		taskHandle1 = TaskHandle(1)
		DAQmxCreateTask("",byref(taskHandle1))

		DAQmxCreateDOChan(taskHandle1,"/Dev1/port0/line6:7","",DAQmx_Val_ChanForAllLines)
		self.taskHandleDirEnable = taskHandle1
		self.enableTimeout = enableTimeout


	def onEnableTimeout(self):
		#print('thread_start')
		while (time.time()-self.enableTimeStart) < self.enableTimeout:
			time.sleep(1)
			#print(time.time()-self.enableTimeStart, self.enableTimeout,self.enableTimeStart)
		print('HWP_enableTimeout')
		try:
			self.enable(0)
		except:
			print('err')
	def enable(self,state):
		if state:
			state = 1
		else:
			state = 0
		#print('ebable',self.Enable, state)

		if not self.Enable == state:
			self.Enable = state
			data = np.array([self.Direction, self.Enable], dtype=np.uint8)
			DAQmxStartTask(self.taskHandleDirEnable)
			DAQmxWriteDigitalLines(self.taskHandleDirEnable,1,1,10.0,DAQmx_Val_GroupByChannel,data,None,None)
			DAQmxStopTask(self.taskHandleDirEnable)
			if state == 1:
				enableTimeout_thread = Thread(target=self.onEnableTimeout)
				self.enableTimeStart = time.time()
				enableTimeout_thread.start()

		self.enableTimeStart = time.time()
		#print(self.enableTimeStart)
		#self.Enable = state

	def direction(self,state):
		if state:
			state = 1
		else:
			state = 0
		self.Direction = state
		data = np.array([self.Direction, self.Enable], dtype=np.uint8)
		DAQmxStartTask(self.taskHandleDirEnable)
		DAQmxWriteDigitalLines(self.taskHandleDirEnable,1,1,10.0,DAQmx_Val_GroupByChannel,data,None,None)
		DAQmxStopTask(self.taskHandleDirEnable)

	def start(self,step=1, wait=False,timeout=100):
		if step>0:
			#print(type(step))
			steps = round(step*self.calibr)
			real_angle_step = steps/self.calibr
			if self.Direction:
				self.currentAngle += real_angle_step
			else:
				self.currentAngle -= real_angle_step

			#if not self.Enable:
			self.enable(1)
			DAQmxCfgImplicitTiming(self.taskHandle,DAQmx_Val_FiniteSamps,steps)
			DAQmxStartTask(self.taskHandle)
			if wait:
				DAQmxWaitUntilTaskDone(self.taskHandle,timeout)

				self.stop()


	def moveTo(self,angle,wait=False):
		step = angle-self.currentAngle
		self.direction(step>0)
		self.start(step=abs(step),wait=wait)
		time.sleep(0.1)

	def stop(self):
		DAQmxStopTask(self.taskHandle)
		#self.enable(0)

	def setAngle(self,new_angle):
		self.currentAngle = new_angle
	def getAngle(self):
		return self.currentAngle
	def resetAngel(self):
		self.currentAngle = 0

	def clear(self):
		DAQmxClearTask(self.taskHandle)
		DAQmxClearTask(self.taskHandleDirEnable)


	def close(self):
		self.enable(0)
		#self.clear()


if __name__=="__main__":
	import time
	#HWP_Enable(1)
	pulse_gene1 = HWP_stepper(4000,0.02, "dev1/ctr1", reset=True)
	#for i in range(0,360+1,1):
	#	print(i)
	#	pulse_gene1.moveTo(i,wait=True)
	pulse_gene1.moveTo(360,wait=True)
	#pulse_gene1.moveTo(90,wait=True)
	#pulse_gene1.moveTo(180,wait=True)
	#pulse_gene1.moveTo(45,wait=True)
	#time.sleep(1)
	'''
	pulse_gene1.start(step=180,wait=True)
	#pulse_gene1.stop()
	print(pulse_gene1.getAngle())
	time.sleep(1)
	pulse_gene1.direction(1)
	pulse_gene1.start(step=360,wait=True)
	print(pulse_gene1.getAngle())
	#pulse_gene1.moveTo(45,wait=True)
	print(pulse_gene1.getAngle())
	'''
	pulse_gene1.enable(0)
	pulse_gene1.close()
	#pulse_gene1.enable(0)
	#HWP_Enable(0)
