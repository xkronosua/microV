# -*- coding: utf-8 -*-
from .E727Wrapper import *
import time, re, os, sys
import traceback




class E727():


	szUsbController = create_string_buffer(1024)
	szAxes = create_string_buffer(512)
	bFlags = create_string_buffer(3);
	szErrorMesage = create_string_buffer(1024)
	ID = 0
	iError = None

	def __init__(self):
		pass
	def EnumerateUSB(self):
		#print(self.libDLL)

		c=create_string_buffer(len(b"PI E-727"),b"PI E-727")
		r = PI_EnumerateUSB(self.szUsbController, 1024, c)
		print(c.value,r,self.szUsbController.value)
		usb_id = self.szUsbController.value.split(b'\n')
		pattern = re.compile(b".*PI E-727 Controller SN [0-9]+")
		for i in usb_id:
			if pattern.match(i):
				#print(i.decode())
				id_ = i+b"\n"
				self.szUsbController = c_char_p(id_)

		print("#",self.szUsbController.value)
		return self.szUsbController.value

	def ConnectUSB(self):
		#/////////////////////////////////////////
		#// connect to the controller over USB. //
		#/////////////////////////////////////////
		self.EnumerateUSB()

		self.ID = PI_ConnectUSB(self.szUsbController)
		print(self.ID)
		if self.ID<0:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("ConnectUSB: ERROR ",iError, szErrorMesage)
		return self.ID

	def ConnectUSBWithBaudRate(self,iBaudRate=115200):
		#/////////////////////////////////////////
		#// connect to the controller over USB. //
		#/////////////////////////////////////////
		self.EnumerateUSB()

		self.ID = PI_ConnectUSBWithBaudRate(self.szUsbController,iBaudRate)
		print(self.ID)
		if self.ID<0:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("ConnectUSBWithBaudRate: ERROR ",iError, szErrorMesage)
		return self.ID

	def CloseConnection(self):

		PI_CloseConnection(self.ID)
	def GetError(self):

		self.iError = PI_GetError(self.ID)
		return self.iError
	def TranslateError(self, iError):

		r = PI_TranslateError(iError, self.szErrorMesage, 1024)
		return self.szErrorMesage.value
	'''
	def EAX(self,axis=b'1 2 3',flags=[True,True,True]):
		if type(axis)==bytes:
			bFlags = (c_bool*len(flags))()
			for i in range(len(flags)):
				bFlags[i] = c_bool(flags[i])
		else:
			axis=str(axis).encode()
			bFlags = c_bool(flags)

		r = PI_EAX(self.ID, axis, bFlags)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("EAX> ERROR ",iError, szErrorMesage)
			#self.CloseConnection()
		else:
			pass
		return r

	def qEAX(self):
		val = (c_bool*3)()
		axis = b''
		r = PI_qEAX(self.ID, axis, val)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("qEAX> ERROR ",iError, szErrorMesage)
			#self.CloseConnection()
		else:
			pass
		return [v for v in val]
	'''

	def qSAI(self):
		#/////////////////////////////////////////
		#// Get the name of the connected axis. //
		#/////////////////////////////////////////

		r = PI_qSAI(self.ID, self.szAxes, 512)
		tmp = self.szAxes.value
		self.szAxes = tmp#.replace(b'\n',b'')
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("qSAI> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		else:
			pass
		return self.szAxes#.value

	def SVO(self,axis=b'1 2 3',flags=[True,True,True]):

		ax = axis.split(b' ')
		for i in range(len(flags)):
			r = PI_SVO(self.ID, ax[i], c_bool(flags[i]))
			if not r:
				iError = self.GetError()
				szErrorMesage=self.TranslateError(iError)
				print("SVO> ERROR ",iError, szErrorMesage)
				self.CloseConnection()
			else:
				pass
		return r

	def qSVO(self,axis=b'1 2 3'):
		val = (c_bool*3)()
		val[0] = c_bool(False)
		val[1] = c_bool(False)
		val[2] = c_bool(False)
		out = []
		ax = axis.split(b' ')
		for i in range(len(ax)):
			r = PI_qSVO(self.ID, ax[i], val)
			out.append(val[0])
			if not r:
				iError = self.GetError()
				szErrorMesage=self.TranslateError(iError)
				print("qSVO> ERROR ",iError, szErrorMesage)
				self.CloseConnection()
			else:
				pass
		return out

	def CMO(self,axis=b'1 2 3',mode=[True,True,True]):
		if type(axis)==bytes:
			mode_ = (c_bool*len(mode))()
			for i in range(len(mode)):
				mode_[i] = c_bool(mode[i])
		else:
			axis=str(axis).encode()
			mode_ = c_bool(mode)

		r = PI_CMO(self.ID, axis, mode_)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("CMO> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		else:
			pass
		return r
	'''
	def qCMO(self):
		val = (c_int*3)()
		axis = b''
		r = PI_qCMO(self.ID, axis, val)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("qCMO> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		else:
			pass
		return [v for v in val]

	def BRA(self,axis=b'1 2 3',flags=[True]):

		bFlags = (c_bool*len(flags))()
		for i in range(len(flags)):
			bFlags[i] = c_bool(flags[i])

		r = PI_BRA(self.ID, axis, bFlags)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("BRA> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		else:
			pass
		return r

	def qBRA(self):

		szBuffer = create_string_buffer(17)
		r = PI_qBRA(self.ID, szBuffer, 16)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("qBRA> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		return szBuffer.value
	'''

	def ATZ(self):
		pdLowVoltageArray = (c_double*3)()

		res = 0
		print('PI_ATZ')
		r = PI_ATZ(self.ID, b'', pdLowVoltageArray,c_bool(True))
		time.sleep(5)
		iError = self.GetError()
		k = 0
		while iError==61:
			print('ATZ',iError)
			iError = self.GetError()
			time.sleep(0.5)
			k+=1
			if k==1000:
				self.CloseConnection()
				break
		print('PI_ATZ:Done')
		return r

	def MOV(self, dPos=[50.0,50.0,50.0],axis=b'1 2 3', waitUntilReady=False):
		if type(axis)==bytes:
			dPos_ = (c_double*len(dPos))()
			for i in range(len(dPos)):
				dPos_[i] = c_double(dPos[i])
		else:
			axis=str(axis).encode()
			dPos_ = c_double(dPos)

		r = PI_MOV(self.ID, axis, dPos_)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("MOV> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		else:
			pass

		if waitUntilReady:
			m = self.IsMoving()
			#print(m)
			N_max = 1000000
			n = 0
			while sum(m)!=0 or n>N_max:
				n=n+1
				m = self.IsMoving()
				#print(m)
				time.sleep(0.001)
		return r

	def qPOS(self,axis=b"1 2 3"):
		val = (c_double*3)()
		r = PI_qPOS(self.ID, axis, val)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("qPOS> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		return [v for v in val]

	def IsMoving(self,axis=b""):
		val = (c_bool*3)()

		r = PI_IsMoving(self.ID, axis, val)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("IsMoving> ERROR ",iError, szErrorMesage)
			#self.CloseConnection()
		return [v for v in val]

	def VEL(self, dVel=[1000,1000,1000],axis=b'1 2 3'):
		if type(axis)==bytes:
			dVel_ = (c_double*len(dVel))()
			for i in range(len(dVel)):
				dVel_[i] = c_double(dVel[i])
		else:
			axis=str(axis).encode()
			dVel_ = c_double(dVel)

		r = PI_VEL(self.ID, axis, dVel_)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("VEL> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		else:
			pass

		return r
	def qVEL(self,axis=b"1 2 3"):
		val = (c_double*3)()
		r = PI_qVEL(self.ID, axis, val)
		if not r:
			iError = self.GetError()
			szErrorMesage=self.TranslateError(iError)
			print("qVEL> ERROR ",iError, szErrorMesage)
			self.CloseConnection()
		return [v for v in val]

if __name__ == "__main__":
	e = E727()
	print(e.ConnectUSBWithBaudRate())
	try:
		print(e.qSAI())
		print("SVO:",e.qSVO(b'1 2 3'))
		print(e.SVO(b'1 2 3',[True, True, True]))
		print("SVO:",e.qSVO(b'1 2 3'))
		time.sleep(1)
		#print(e.ATZ())
		print('X')
		#print(e.BRA(b'1 2 3',[True, True, True]))
		#print(e.CMO())
		#print(e.qCMO())
		print(e.VEL([1000,1000,1000],b'1 2 3'))
		print(e.qVEL())
		print(e.MOV(dPos=100,axis=1, waitUntilReady=True))
		print(e.qPOS())
		print(e.MOV(0,axis=1, waitUntilReady=True))
		print(e.qPOS())
		print('Y')
		print(e.MOV(100,axis=2, waitUntilReady=True))
		print(e.qPOS())
		print(e.MOV(0,axis=2, waitUntilReady=True))
		print(e.qPOS())
		print('Z')
		print(e.MOV(100,axis=3, waitUntilReady=True))
		print(e.qPOS())
		print(e.MOV(0,axis=3, waitUntilReady=True))
		print(e.qPOS())


		print(e.MOV(dPos=[50,50,50],axis=b"1 2 3", waitUntilReady=True))
		print(e.qPOS())
		e.SVO(b'1 2 3', flags=[False, False, False])
		print(e.CloseConnection())

	except:
		traceback.print_exc()
		print(e.CloseConnection())
