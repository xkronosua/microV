import numpy as np
from ctypes import CDLL, WinDLL, c_int, c_float, c_char, c_char_p, byref, POINTER, create_string_buffer
from collections import OrderedDict
import logging
from threading import RLock
import os
__all__ = ["ShamRockErrors",
		   "ShamRockController",
		   "ShamRock"]

#===============================================================================
# ShamRockErrors
#===============================================================================

class ShamRockErrors:
	COMMUNICATION_ERROR = "SHAMROCK_COMMUNICATION_ERROR"
	SUCCESS = "SHAMROCK_SUCCESS"
	P1INVALID = "SHAMROCK_P1INVALID"
	P2INVALID = "SHAMROCK_P2INVALID"
	P3INVALID = "SHAMROCK_P3INVALID"
	P4INVALID = "SHAMROCK_P4INVALID"
	P5INVALID = "SHAMROCK_P5INVALID"
	NOT_INITIALIZED = "SHAMROCK_NOT_INITIALIZED"
	NOT_AVAILABLE = "SHAMROCK_NOT_AVAILABLE"

	ERROR_CODE = {
		20201: COMMUNICATION_ERROR,
		20202: SUCCESS,
		20266: P1INVALID,
		20267: P2INVALID,
		20268: P3INVALID,
		20269: P4INVALID,
		20270: P5INVALID,
		20275: NOT_INITIALIZED,
		20292: NOT_AVAILABLE
	}

	@staticmethod
	def FromCode(errorCode):
		if errorCode in ShamRockErrors.ERROR_CODE:
			return ShamRockErrors.ERROR_CODE[errorCode]
		else:
			return "Unknown error (%d)" % (errorCode)

	@staticmethod
	def ProcessErrorCode(errorCode):
		errorStr = ShamRockErrors.FromCode(errorCode)
		#print "ProcessErrorCode", sys._getframe(1).f_code.co_name, errorStr
		if errorStr != ShamRockErrors.SUCCESS:
			raise RuntimeError("ShamRock error: %s" % (errorStr))

#===============================================================================
# ShamRockController
#===============================================================================

class ShamRockController:

	ACCESSORYMIN = 1
	ACCESSORYMAX = 2
	FILTERMIN = 1
	FILTERMAX = 6
	TURRETMIN = 1
	TURRETMAX = 3
	GRATINGMIN = 1
	SLITWIDTHMIN = 10
	SLITWIDTHMAX = 2500
	SHUTTERMODEMIN = 0
	SHUTTERMODEMAX = 1
	DET_OFFSET_MIN = -240000
	DET_OFFSET_MAX = 240000
	GRAT_OFFSET_MIN = -20000
	GRAT_OFFSET_MAX = 20000

	SLIT_INDEX_MIN = 1
	SLIT_INDEX_MAX = 4

	INPUT_SLIT_SIDE = 1
	INPUT_SLIT_DIRECT = 2
	OUTPUT_SLIT_SIDE = 3
	OUTPUT_SLIT_DIRECT = 4

	FLIPPER_INDEX_MIN = 1
	FLIPPER_INDEX_MAX = 2
	PORTMIN = 0
	PORTMAX = 1

	INPUT_FLIPPER = 1
	OUTPUT_FLIPPER = 2
	DIRECT_PORT = 0
	SIDE_PORT = 1

	ERRORLENGTH = 64

	SHUTTER_CLOSED = 0
	SHUTTER_OPENED = 1


	def __init__(self):
		self.logger = logging.getLogger("ShamRockController")
		self.lock = RLock()
		self.dll = CDLL( os.path.abspath(__file__).split('Shamrock.py')[0]+"ShamrockCIF.dll")

		self.dll.ShamrockInitialize.argtypes = [c_char_p]
		self.dll.ShamrockClose.argtypes = []
		self.dll.ShamrockGetNumberDevices.argtypes = [POINTER(c_int)]
		self.dll.ShamrockGetFunctionReturnDescription.argtypes = [c_int, c_char_p, c_int]
		self.dll.ShamrockGetSerialNumber.argtypes = [c_int, c_char_p]
		self.dll.ShamrockEepromGetOpticalParams.argtypes = [c_int, POINTER(c_float), POINTER(c_float), POINTER(c_float)]

		self.dll.ShamrockSetGrating.argtypes = [c_int, c_int]
		self.dll.ShamrockGetGrating.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockWavelengthReset.argtypes = [c_int]
		self.dll.ShamrockGetNumberGratings.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockGetGratingInfo.argtypes = [c_int, c_int, POINTER(c_float), c_char_p, POINTER(c_int), POINTER(c_int)]
		self.dll.ShamrockSetDetectorOffset.argtypes = [c_int, c_int]
		self.dll.ShamrockGetDetectorOffset.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockSetDetectorOffsetPort2.argtypes = [c_int, c_int]
		self.dll.ShamrockGetDetectorOffsetPort2.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockSetGratingOffset.argtypes = [c_int, c_int, c_int]
		self.dll.ShamrockGetGratingOffset.argtypes = [c_int, c_int, POINTER(c_int)]
		self.dll.ShamrockGratingIsPresent.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockSetTurret.argtypes = [c_int, c_int]
		self.dll.ShamrockGetTurret.argtypes = [c_int, POINTER(c_int)]

		self.dll.ShamrockSetWavelength.argtypes = [c_int, c_float]
		self.dll.ShamrockGetWavelength.argtypes = [c_int, POINTER(c_float)]
		self.dll.ShamrockGotoZeroOrder.argtypes = [c_int]
		self.dll.ShamrockAtZeroOrder.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockGetWavelengthLimits.argtypes = [c_int, c_int, POINTER(c_float), POINTER(c_float)]
		self.dll.ShamrockWavelengthIsPresent.argtypes = [c_int, POINTER(c_int)]

		self.dll.ShamrockSetAutoSlitWidth.argtypes = [c_int, c_int, c_float]
		self.dll.ShamrockGetAutoSlitWidth.argtypes = [c_int, c_int, POINTER(c_float)]
		self.dll.ShamrockAutoSlitReset.argtypes = [c_int, c_int]
		self.dll.ShamrockAutoSlitIsPresent.argtypes = [c_int, c_int, POINTER(c_int)]
		self.dll.ShamrockSetAutoSlitCoefficients.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int]
		self.dll.ShamrockGetAutoSlitCoefficients.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)]

		self.dll.ShamrockSetShutter.argtypes = [c_int, c_int]
		self.dll.ShamrockGetShutter.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockIsModePossible.argtypes = [c_int, c_int, POINTER(c_int)]
		self.dll.ShamrockShutterIsPresent.argtypes = [c_int, POINTER(c_int)]

		self.dll.ShamrockSetFilter.argtypes = [c_int, c_int]
		self.dll.ShamrockGetFilter.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockGetFilterInfo.argtypes = [c_int, c_int, c_char_p]
		self.dll.ShamrockSetFilterInfo.argtypes = [c_int, c_int, c_char_p]
		self.dll.ShamrockFilterReset.argtypes = [c_int]
		self.dll.ShamrockFilterIsPresent.argtypes = [c_int, POINTER(c_int)]

		self.dll.ShamrockSetPixelWidth.argtypes = [c_int, c_float]
		self.dll.ShamrockSetNumberPixels.argtypes = [c_int, c_int]
		self.dll.ShamrockGetPixelWidth.argtypes = [c_int, POINTER(c_float)]
		self.dll.ShamrockGetNumberPixels.argtypes = [c_int, POINTER(c_int)]
		self.dll.ShamrockGetCalibration.argtypes = [c_int, POINTER(c_float), c_int]

		self.dll.ShamrockSetPort.argtypes = [c_int, c_int]
		self.dll.ShamrockGetPort.argtypes = [c_int, POINTER(c_int)]


	# Basic Shamrock features --------------------------------------------------

	def Initialize(self):
		self.logger.debug("Initialize")
		iniPath = c_char_p(b"")
		with self.lock:
			error = self.dll.ShamrockInitialize(iniPath)
			print(self.GetFunctionReturnDescription(error))
		#ShamRockErrors.ProcessErrorCode(error)

	def Close(self):
		self.logger.debug("Close")
		with self.lock:
			error = self.dll.ShamrockClose()
		ShamRockErrors.ProcessErrorCode(error)

	def GetNumberDevices(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetNumberDevices(byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetNumberDevices -> %d" % (res.value))
		return res.value

	def GetFunctionReturnDescription(self, error):
		errorC = c_int(error)
		res = c_char_p(b" " * ShamRockController.ERRORLENGTH)
		with self.lock:
			self.dll.ShamrockGetFunctionReturnDescription(errorC, res, ShamRockController.ERRORLENGTH)
		self.logger.debug("GetFunctionReturnDescription %d -> %s" % (error, res.value))
		return res.value

	def Connect(self, nr = 0):
		self.logger.debug("Connect %d" % (nr))
		self.shamrock = ShamRock(nr, self)
		return self.shamrock

#===============================================================================
# ShamRock
#===============================================================================

class ShamRock():

	def __init__(self, currentShamrock, controller):
		self.logger = logging.getLogger("ShamRock")
		self.lock = RLock()
		self.logger.debug("__init__ %d" % (currentShamrock))
		self.controller = controller
		self.dll = controller.dll
		self.curId = currentShamrock
		#self.SetShutter(ShamRockController.SHUTTER_CLOSED)

	# EEPROM functions ---------------------------------------------------------

	def GetSerialNumber(self):
		res = create_string_buffer(101)
		with self.lock:
			error = self.dll.ShamrockGetSerialNumber(self.curId, res)
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetSerialNumber -> %s" % (res.value))
		return res.value

	def EepromGetOpticalParams(self):
		focalLength = c_float()
		angularDeviation = c_float()
		focalTilt = c_float()
		with self.lock:
			error = self.dll.ShamrockEepromGetOpticalParams(\
					self.curId, byref(focalLength), \
					byref(angularDeviation), byref(focalTilt))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("EepromGetOpticalParams %f %f %f" % \
				(focalLength.value, angularDeviation.value, focalTilt.value))
		return focalLength.value, angularDeviation.value, focalTilt.value

	# Basic Grating features ---------------------------------------------------

	def SetGrating(self, grating):
		self.logger.debug("SetGrating %d" % (grating))
		with self.lock:
			error = self.dll.ShamrockSetGrating(self.curId, grating)
		ShamRockErrors.ProcessErrorCode(error)

	def GetGrating(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetGrating(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetGrating %d" % (res.value))
		return res.value

	def WavelengthReset(self):
		self.logger.debug("WavelengthReset")
		with self.lock:
			error = self.dll.ShamrockWavelengthReset(self.curId)
		ShamRockErrors.ProcessErrorCode(error)

	def GetNumberGratings(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetNumberGratings(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetNumberGratings %d" % (res.value))
		return res.value

	def GetGratingInfo(self, grating):
		lines = c_float()
		blaze = c_char_p(b" " * 100)
		home = c_int()
		offset = c_int()

		with self.lock:
			error = self.dll.ShamrockGetGratingInfo(self.curId, grating, byref(lines), \
												blaze, byref(home), byref(offset))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetGratingInfo %s" % \
				(str((lines.value, blaze.value, home.value, offset.value))))
		return lines.value, blaze.value, home.value, offset.value

	def SetDetectorOffset(self, offset):
		self.logger.debug("SetDetectorOffset %d" % (offset))
		with self.lock:
			error = self.dll.ShamrockSetDetectorOffset(self.curId, offset)
		ShamRockErrors.ProcessErrorCode(error)

	def GetDetectorOffset(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetDetectorOffset(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetDetectorOffset -> %d" % (res.value))
		return res.value

	def SetDetectorOffsetPort2(self, offset):
		self.logger.debug("SetDetectorOffsetPort2 %d" % (offset))
		with self.lock:
			error = self.dll.ShamrockSetDetectorOffsetPort2(self.curId, offset)
		ShamRockErrors.ProcessErrorCode(error)

	def GetDetectorOffsetPort2(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetDetectorOffsetPort2(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetDetectorOffsetPort2 -> %d" % (res.value))
		return res.value

	def SetGratingOffset(self, grating, offset):
		self.logger.debug("SetGratingOffset %d %d" % (grating, offset))
		with self.lock:
			error = self.dll.ShamrockSetGratingOffset(self.curId, grating, offset)
		ShamRockErrors.ProcessErrorCode(error)

	def GetGratingOffset(self, grating):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetGratingOffset(self.curId, grating, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetGratingOffset %d -> %d" % (grating, res.value))
		return res.value

	def GratingIsPresent(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGratingIsPresent(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GratingIsPresent %d" % (res.value))
		return res.value != 0

	def SetTurret(self, turret):
		self.logger.debug("SetTurret %d" % (turret))
		with self.lock:
			error = self.dll.ShamrockSetTurret(self.curId, turret)
		ShamRockErrors.ProcessErrorCode(error)

	def GetTurret(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetTurret(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetTurret -> %d" % (res.value))
		return res.value

	# Wavelength ---------------------------------------------------------------

	def SetWavelength(self, wl):
		self.logger.debug("SetWavelength %f" % (wl))
		with self.lock:
			error = self.dll.ShamrockSetWavelength(self.curId, wl)
		ShamRockErrors.ProcessErrorCode(error)

	def GetWavelength(self):
		res = c_float()
		with self.lock:
			error = self.dll.ShamrockGetWavelength(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetWavelength -> %f" % (res.value))
		return res.value

	def GotoZeroOrder(self):
		self.logger.debug("GotoZeroOrder")
		with self.lock:
			error = self.dll.ShamrockGotoZeroOrder(self.curId)
		ShamRockErrors.ProcessErrorCode(error)

	def AtZeroOrder(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockAtZeroOrder(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("AtZeroOrder %d" % (res.value))
		return res.value != 0

	def GetWavelengthLimits(self, grating):
		wlmin = c_float()
		wlmax = c_float()
		with self.lock:
			error = self.dll.ShamrockGetWavelengthLimits(self.curId, grating, \
													 byref(wlmin), byref(wlmax))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetWavelengthLimits -> %f %f" % \
						  (wlmin.value, wlmax.value))
		return wlmin.value, wlmax.value

	def WavelengthIsPresent(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockWavelengthIsPresent(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("WavelengthIsPresent %d" % (res.value))
		return res.value != 0

	# Slit functions -----------------------------------------------------------

	def SetAutoSlitWidth(self, index, width):
		self.logger.debug("SetAutoSlitWidth %d %f" % (index, width))
		with self.lock:
			error = self.dll.ShamrockSetAutoSlitWidth(self.curId, index, width)
		ShamRockErrors.ProcessErrorCode(error)

	def GetAutoSlitWidth(self, index):
		res = c_float()
		with self.lock:
			error = self.dll.ShamrockGetAutoSlitWidth(self.curId, index, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetAutoSlitWidth %d -> %f" % (index, res.value))
		return res.value

	def AutoSlitReset(self, index):
		self.logger.debug("AutoSlitReset %d" % (index))
		with self.lock:
			error = self.dll.ShamrockAutoSlitReset(self.curId, index)
		ShamRockErrors.ProcessErrorCode(error)

	def AutoSlitIsPresent(self, index):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockAutoSlitIsPresent(self.curId, index, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("AutoSlitIsPresent %d -> %d" % (index, res.value))
		return res.value != 0

	def SetAutoSlitCoefficients(self, index, x1, y1, x2, y2):
		with self.lock:
			error = self.dll.ShamrockSetAutoSlitCoefficients(self.curId, index, x1, y1, x2, y2)
		self.logger.debug("SetAutoSlitCoefficients %d %d %d %d %d" % \
						  (index, x1, y1, x2, y2))
		ShamRockErrors.ProcessErrorCode(error)

	def GetAutoSlitCoefficients(self, index):
		x1 = c_int()
		y1 = c_int()
		x2 = c_int()
		y2 = c_int()
		with self.lock:
			error = self.dll.ShamrockGetAutoSlitCoefficients(self.curId, index, byref(x1), \
														 byref(y1), byref(x2), byref(y2))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetAutoSlitCoefficients %d -> %d %d %d %d" % \
						  (index, x1.value, y1.value, x2.value, y2.value))
		return x1.value, y1.value, x2.value, y2.value

	# Shutter ------------------------------------------------------------------

	def SetShutter(self, mode):
		self.logger.debug("SetShutter %d" % (mode))
		with self.lock:
			error = self.dll.ShamrockSetShutter(self.curId, mode)
		ShamRockErrors.ProcessErrorCode(error)

	def GetShutter(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetShutter(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetShutter -> %d" % (res.value))
		return res.value

	def IsModePossible(self, mode):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockIsModePossible(self.curId, mode, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("IsModePossible -> %d" % (res.value))
		return res.value != 0

	def ShutterIsPresent(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockShutterIsPresent(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("ShutterIsPresent -> %d" % (res.value))
		return res.value != 0

	# Filters ------------------------------------------------------------------

	def SetFilter(self, filterId):
		self.logger.debug("SetFilter %d" % (filterId))
		with self.lock:
			error = self.dll.ShamrockSetFilter(self.curId, filterId)
		ShamRockErrors.ProcessErrorCode(error)

	def GetFilter(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetFilter(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetFilter -> %d" % (res.value))
		return res.value

	def GetFilterInfo(self, filterId):
		res = c_char_p(b" " * 100)
		with self.lock:
			error = self.dll.ShamrockGetFilterInfo(self.curId, filterId, res)
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetFilterInfo %d -> %s" % (filterId, res.value))
		return res.value

	def SetFilterInfo(self, filterId, filterInfo):
		self.logger.debug("SetFilterInfo %d %s" % (filterId, filterInfo))
		with self.lock:
			error = self.dll.ShamrockSetFilter(self.curId, filterId, filterInfo)
		ShamRockErrors.ProcessErrorCode(error)

	def FilterReset(self):
		self.logger.debug("FilterReset")
		with self.lock:
			error = self.dll.ShamrockFilterReset(self.curId)
		ShamRockErrors.ProcessErrorCode(error)

	def FilterIsPresent(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockFilterIsPresent(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("FilterIsPresent -> %d" % (res.value))
		return res.value != 0


	# Calibration --------------------------------------------------------------

	def SetPixelWidth(self, width):
		self.logger.debug("SetPixelWidth %f" % (width))
		with self.lock:
			error = self.dll.ShamrockSetPixelWidth(self.curId, width)
		ShamRockErrors.ProcessErrorCode(error)

	def SetNumberPixels(self, numberPixels):
		self.logger.debug("SetNumberPixels %d" % (numberPixels))
		with self.lock:
			error = self.dll.ShamrockSetNumberPixels(self.curId, numberPixels)
		ShamRockErrors.ProcessErrorCode(error)

	def GetPixelWidth(self):
		res = c_float()
		with self.lock:
			error = self.dll.ShamrockGetPixelWidth(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetPixelWidth -> %f" % (res.value))
		return res.value

	def GetNumberPixels(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetNumberPixels(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetNumberPixels -> %d" % (res.value))
		return res.value

	def GetCalibration(self):
		nrPixels = self.GetNumberPixels()
		self.logger.debug("GetCalibration %d" % (nrPixels))
		calibValues = (c_float * nrPixels)()
		with self.lock:
			error = self.dll.ShamrockGetCalibration(self.curId, calibValues, nrPixels)
		ShamRockErrors.ProcessErrorCode(error)
		res = np.array(np.frombuffer(calibValues, dtype = np.float32), dtype = np.float)
		return res

	def GetWavelengths(self):
		return 1e-9 * self.GetCalibration()

	def SetPort(self, port):
		self.logger.debug("SetPort %d" % (port))
		with self.lock:
			error = self.dll.ShamrockSetGrating(self.curId, port)
		ShamRockErrors.ProcessErrorCode(error)

	def GetPort(self):
		res = c_int()
		with self.lock:
			error = self.dll.ShamrockGetGrating(self.curId, byref(res))
		ShamRockErrors.ProcessErrorCode(error)
		self.logger.debug("GetPort %d" % (res.value))
		return res.value
	# Info ---------------------------------------------------------------------

	def GetInfo(self):
		res = OrderedDict()
		methods = [(self.GetSerialNumber, []),
				   (self.EepromGetOpticalParams, []),
				   (self.GetGrating, []),
				   (self.GetNumberGratings, []),
				   (self.GetGratingInfo, [(1,), (2,), (3,)]),
				   #(self.GetDetectorOffset, []),
				   #(self.GetDetectorOffsetPort2, []),
				   (self.GratingIsPresent, []),
				   (self.GetTurret, []),
				   (self.GetWavelength, []),
				   (self.AtZeroOrder, []),
				   (self.GetWavelengthLimits, [(1,), (2,), (3,)]),
				   (self.WavelengthIsPresent, []),
				   (self.GetAutoSlitWidth, [(1,)]),
				   #(self.GetAutoSlitCoefficients, [(1,)]),
				   (self.GetShutter, []),
				   #(self.ShutterIsPresent, []),
				   #(self.GetFilter, []),
				   #(self.GetFilterInfo, [(1,), (2,), (3,), (4,), (5,), (6,)]),
				   (self.FilterIsPresent, []),
				   (self.GetPixelWidth, []),
				   (self.GetNumberPixels, []),
				   #(self.GetCalibration, [])
				   ]

		for m, args in methods:
			if len(args) == 0:
				res[m.__name__] = m()
			else:
				for arg in args:
					res[m.__name__ + str(arg)] = m(*arg)
		return res

if __name__ == "__main__":
	pass
