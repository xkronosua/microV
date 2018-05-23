import numpy as np
import logging
import ctypes
from ctypes import byref, c_int, c_float, c_char_p, c_long, c_ulong, POINTER
from .AndorUtils import ANDOR_STATUS, ANDOR_CODES
from collections import OrderedDict
from threading import RLock
import os

__all__ = ["AndorCamera"]


class AndorCamera(object):

	READ_MODE_CODES = {"FVB": 0,
					  "Image": 4}

	TRIGGER_MODE_CODES = {"internal": 0,
						  "external": 1,
						  "softwareTrigger": 10}

	ACQUISITION_MODE_CODES = {"single": 1,
							  "accumulate": 2}

	OUTPUT_AMPLIFIERS = {"EMCCD": 0,
						 "CCD": 1}

	FAN_MODES = {"full": 0,
				 "low": 1,
				 "off": 2}

	def __init__(self):
		self.logger = logging.getLogger("AndorCamera")
		self.lock = RLock()
		path = os.path.abspath(__file__).split('AndorCamera.py')[0]+"atmcd64d.dll"
		print(path)
		self.dll = ctypes.WinDLL(path)

		# Variables
		self._vBin = 1
		self._hBin = 1

		# Init and close
		self.dll.Initialize.argtypes = [c_char_p]
		self.dll.ShutDown.argtypes = []

		# Cooler
		self.dll.CoolerON.argtypes = []
		self.dll.CoolerOFF.argtypes = []
		self.dll.IsCoolerOn.argtypes = [POINTER(c_int)]
		self.dll.GetTemperatureF.argtypes = [POINTER(c_float)]
		self.dll.SetTemperature.argtypes = [c_int]

		# Acquisition
		self.dll.SetReadMode.argtypes = [c_int]
		self.dll.SetImage.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int]
		self.dll.SetTriggerMode.argtypes = [c_int]
		self.dll.SetAcquisitionMode.argtypes = [c_int]

		self.dll.SetExposureTime.argtypes = [c_float]
		self.dll.SetAccumulationCycleTime.argtypes = [c_float]
		self.dll.SetNumberAccumulations.argtypes = [c_int]
		self.dll.GetAcquisitionTimings.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]

		self.dll.StartAcquisition.argtypes = []
		self.dll.WaitForAcquisition.argtypes = []
		self.dll.CancelWait.argtypes = []
		self.dll.AbortAcquisition.argtypes = []
		self.dll.GetStatus.argtypes = [POINTER(c_int)]
		self.dll.GetMostRecentImage.argtypes = [POINTER(c_long), c_ulong]

		# Get other information
		self.dll.GetBaselineClamp.argtypes = [POINTER(c_int)]
		self.dll.GetBitDepth.argtypes = [c_int, POINTER(c_int)]
		self.dll.GetCameraSerialNumber.argtypes = [POINTER(c_int)]
		self.dll.GetDetector.argtypes = [POINTER(c_int), POINTER(c_int)]
		self.dll.GetPixelSize.argtypes = [POINTER(c_float), POINTER(c_float)]

		self.dll.GetFastestRecommendedVSSpeed.argtypes = [POINTER(c_int), POINTER(c_float)]
		self.dll.GetVSSpeed.argtypes = [c_int, POINTER(c_float)]
		self.dll.GetNumberVSSpeeds.argtypes = [POINTER(c_int)]
		self.dll.SetVSSpeed.argtypes = [c_int]

		self.dll.GetNumberADChannels.argtypes = [POINTER(c_int)]
		self.dll.SetADChannel.argtypes = [c_int]
		self.dll.GetNumberAmp.argtypes = [POINTER(c_int)]
		self.dll.SetOutputAmplifier.argtypes = [c_int]

		self.dll.GetHSSpeed.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
		self.dll.GetNumberHSSpeeds.argtypes = [c_int, c_int, POINTER(c_int)]
		self.dll.SetHSSpeed.argtypes = [c_int, c_int]

		self.dll.GetNumberPreAmpGains.argtypes = [POINTER(c_int)]
		self.dll.SetPreAmpGain.argtypes = [c_int]
		self.dll.GetPreAmpGain.argtypes = [c_int, POINTER(c_float)]

		self.dll.GetEMCCDGain.argtypes = [POINTER(c_int)]

	def _chk(self, statusCode):

		if statusCode == ANDOR_STATUS["DRV_ACQUIRING"]:
			self.logger.warn("Data acquisition in process")
		elif statusCode == ANDOR_STATUS["DRV_TEMPERATURE_NOT_REACHED"]:
			self.logger.warn("Temperature not reached.")
		elif statusCode == ANDOR_STATUS["DRV_TEMPERATURE_DRIFT"]:
			self.logger.warn("Temperature drifting")
		elif statusCode == ANDOR_STATUS["DRV_TEMP_NOT_STABILIZED"]:
			self.logger.warn("Temperature not stabilized")
		elif statusCode == ANDOR_STATUS["DRV_IDLE"]:
			self.logger.warn("Function call resulted in IDLE")
		elif statusCode == ANDOR_STATUS["DRV_TEMPERATURE_STABILIZED"]:
			pass
		elif statusCode != ANDOR_STATUS["DRV_SUCCESS"]:
			raise RuntimeError("Andor returned error %s" % ANDOR_CODES[statusCode])

	# Init ---------------------------------------------------------------------

	def Initialize(self, acquisitionMode = "accumulate", triggerMode = "internal", \
				   temperature = -85,):
		self.logger.debug("Initialize")
		with self.lock:
			self._chk(self.dll.Initialize(b"."))
		self._detShape = self.GetDetector()
		self._pixelSize = self.GetPixelSize()

		# Start cooling

		self.CoolerON()
		self.SetTemperature(temperature)
		self.SetFanMode("full")

		self.SetAcquisitionMode(acquisitionMode)
		self.SetTriggerMode(triggerMode)
		#self.SetOutputAmplifier("CCD")
		self.SetPreAmpGain(0)

		# Set speeds
		self.SetVSSpeed(3)
		self.SetADChannel(0)
		#self.SetHSSpeed("CCD", 2)

		self.SetReadMode("FVB")
		#self.SetImage(1, 1)

	def ShutDown(self):
		self.logger.debug("ShutDown")
		with self.lock:
			self._chk(self.dll.ShutDown())

	# Cooler -------------------------------------------------------------------

	def CoolerON(self):
		self.logger.debug("CoolerON")
		with self.lock:
			self._chk(self.dll.CoolerON())

	def CoolerOFF(self):
		self.logger.debug("CoolerOFF")
		with self.lock:
			self._chk(self.dll.CoolerOFF())

	def IsCoolerOn(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.IsCoolerOn(byref(res)))
		self.logger.debug("IsCoolerOn %d" % (res.value))
		return res.value != 0

	def GetTemperature(self):
		res = c_float()
		with self.lock:
			self._chk(self.dll.GetTemperatureF(byref(res)))
		self.logger.debug("GetTemperature %.2f" % (res.value))
		return res.value

	def SetTemperature(self, temperature):
		self.logger.debug("SetTemperature %.1f" % (temperature))
		with self.lock:
			self._chk(self.dll.SetTemperature(int(temperature)))

	def SetFanMode(self, mode):
		self.logger.debug("SetFanMode %s" % (mode))
		with self.lock:
			self._chk(self.dll.SetFanMode(AndorCamera.FAN_MODES[mode]))

	# Acquisition --------------------------------------------------------------

	def SetReadMode(self, mode):
		self.logger.debug("SetReadMode %s %d" % (mode, AndorCamera.READ_MODE_CODES[mode]))
		with self.lock:
			self._chk(self.dll.SetReadMode(AndorCamera.READ_MODE_CODES[mode]))

			if mode == "FVB":
				self._imageShape = (int(self._detShape[0]),)
				self._hBin, self._vBin = 1, 1
			elif mode == "Image":
				pass
			else:
				raise ValueError("Unknown read mode")
			self._mode = mode

	def SetImage(self, hBin, vBin, hStart = 1, hEnd = None, vStart = 1, vEnd = None):
		if self._mode != "Image":
			raise RuntimeError("Only available in image mode")
		with self.lock:
			hEnd = self._detShape[0] if hEnd is None else hEnd
			vEnd = self._detShape[1] if vEnd is None else vEnd
			self.logger.debug("SetImage %d %d %d %d %d %d" % (hBin, vBin, hStart, hEnd, vStart, vEnd))
			self._chk(self.dll.SetImage(hBin, vBin, hStart, hEnd, vStart, vEnd))
			self._hBin, self._vBin = hBin, vBin
			self._imageShape = (int((hEnd - hStart + 1) / hBin), int((vEnd - vStart + 1) / vBin))
			self.SetReadMode(self._mode)

	def SetTriggerMode(self, mode):
		self.logger.debug("SetTriggerMode %s" % (mode))
		with self.lock:
			self._chk(self.dll.SetTriggerMode(AndorCamera.TRIGGER_MODE_CODES[mode]))

	def SetAcquisitionMode(self, mode):
		self.logger.debug("SetAcquisitionMode %s" % (mode))
		with self.lock:
			self._chk(self.dll.SetAcquisitionMode(AndorCamera.ACQUISITION_MODE_CODES[mode]))

	def SetExposureTime(self, value):
		self.logger.debug("SetExposureTime %.4f" % (value))
		with self.lock:
			self._chk(self.dll.SetExposureTime(value))

	def SetAccumulationCycleTime(self, value):
		self.logger.debug("SetAccumulationCycleTime %.4f" % (value))
		with self.lock:
			self._chk(self.dll.SetAccumulationCycleTime(value))

	def SetNumberAccumulations(self, value):
		self.logger.debug("SetNumberAccumulations %d" % (value))
		with self.lock:
			self._chk(self.dll.SetNumberAccumulations(value))

	def GetAcquisitionTimings(self):
		with self.lock:
			exposure, accumulate, kinetic = c_float(), c_float(), c_float()
			self._chk(self.dll.GetAcquisitionTimings(byref(exposure), \
													 byref(accumulate), \
													 byref(kinetic)))
		self.logger.debug("GetAcquisitionTimings %.4f %.4f %.4f" % \
						  (exposure.value, accumulate.value, kinetic.value))
		return exposure.value, accumulate.value, kinetic.value

	def StartAcquisition(self):
		self.logger.debug("StartAcquisition")
		with self.lock:
			self._chk(self.dll.StartAcquisition())

	def WaitForAcquisition(self):
		self.logger.debug("WaitForAcquisition")
		with self.lock:
			self._chk(self.dll.WaitForAcquisition())

	def CancelWait(self):
		self.logger.debug("CancelWait")
		with self.lock:
			self._chk(self.dll.CancelWait())

	def AbortAcquisition(self):
		self.logger.debug("AbortAcquisition")
		with self.lock:
			self._chk(self.dll.AbortAcquisition())

	def GetStatus(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetStatus(byref(res)))
		self.logger.debug("GetStatus %s" % (ANDOR_CODES[res.value]))
		return ANDOR_CODES[res.value]

	def GetMostRecentImage(self):
		with self.lock:
			size = self._imageShape[0]
			if len(self._imageShape) > 1:
				size *= self._imageShape[1]

			imgArray = c_long * size
			imgBuffer = imgArray()

			# self.WaitForAcquisition()
			self.logger.debug("size %d" % (size))
			self._chk(self.dll.GetMostRecentImage(imgBuffer, size))

			res = np.frombuffer(imgBuffer, dtype = c_long).reshape(self._imageShape, order = "F")
		self.logger.debug("GetMostRecentImage %s" % (str(res.shape)))
		return res

	# Info ---------------------------------------------------------------------

	def GetBaselineClamp(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetBaselineClamp(byref(res)))
		self.logger.debug("GetBaselineClamp %d" % (res.value))
		return res.value != 0

	def GetBitDepth(self, channel):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetBitDepth(channel, byref(res)))
		self.logger.debug("GetBitDepth %d" % (res.value))
		return res.value

	def GetCameraSerialNumber(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetCameraSerialNumber(byref(res)))
		self.logger.debug("GetCameraSerialNumber %d" % (res.value))
		return res.value

	def GetDetector(self):
		with self.lock:
			xS, yS = c_int(), c_int()
			self._chk(self.dll.GetDetector(byref(xS), byref(yS)))
		self.logger.debug("GetDetector %d %d" % (xS.value, yS.value))
		return xS.value, yS.value

	def GetPixelSize(self):
		with self.lock:
			xS, yS = c_float(), c_float()
			self._chk(self.dll.GetPixelSize(byref(xS), byref(yS)))
		self.logger.debug("GetPixelSize %f %f" % (xS.value, yS.value))
		return xS.value, yS.value

	# VSS

	def GetFastestRecommendedVSSpeed(self):
		index, speed = c_int(), c_float()
		with self.lock:
			self._chk(self.dll.GetFastestRecommendedVSSpeed(byref(index), byref(speed)))
		self.logger.debug("GetFastestRecommendedVSSpeed %d %f" % (index.value, speed.value))
		return index.value, speed.value

	def GetVSSpeed(self, index):
		res = c_float()
		with self.lock:
			self._chk(self.dll.GetVSSpeed(index, byref(res)))
		self.logger.debug("GetVSSpeed %f" % (res.value))
		return res.value

	def GetNumberVSSpeeds(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetNumberVSSpeeds(byref(res)))
		self.logger.debug("GetNumberVSSpeeds %d" % (res.value))
		return res.value

	def SetVSSpeed(self, index):
		self.logger.debug("SetVSSpeed %d" % (index))
		with self.lock:
			self._chk(self.dll.SetVSSpeed(index))

	# HSS

	def GetHSSpeed(self, channel, typ, index):
		res = c_float()
		with self.lock:
			self._chk(self.dll.GetHSSpeed(channel, AndorCamera.OUTPUT_AMPLIFIERS[typ], index, byref(res)))
		self.logger.debug("GetHSSpeed %d %s, %s-> %f" % (channel, typ, index, res.value))
		return res.value

	def GetNumberHSSpeeds(self, channel, typ):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetNumberHSSpeeds(channel, AndorCamera.OUTPUT_AMPLIFIERS[typ], byref(res)))
		self.logger.debug("GetNumberHSSpeeds %d, %s -> %d" % (channel, typ, res.value))
		return res.value

	def SetHSSpeed(self, typ, index):
		self.logger.debug("SetHSSpeed %s %d" % (typ, index))
		with self.lock:
			self._chk(self.dll.SetHSSpeed(AndorCamera.OUTPUT_AMPLIFIERS[typ], index))

	# ADC

	def GetNumberADChannels(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetNumberADChannels(byref(res)))
		self.logger.debug("GetNumberADChannels %d" % (res.value))
		return res.value

	def SetADChannel(self, index):
		self.logger.debug("SetADChannel %d" % (index))
		with self.lock:
			self._chk(self.dll.SetADChannel(index))

	def GetNumberAmp(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetNumberAmp(byref(res)))
		self.logger.debug("GetNumberAmp %d" % (res.value))
		return res.value

	def SetOutputAmplifier(self, amplifier):
		self.logger.debug("SetOutputAmplifier %s" % (amplifier))
		with self.lock:
			self._chk(self.dll.SetOutputAmplifier(AndorCamera.OUTPUT_AMPLIFIERS[amplifier]))

	# Gains

	def GetNumberPreAmpGains(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetNumberPreAmpGains(byref(res)))
		self.logger.debug("GetNumberPreAmpGains %d" % (res.value))
		return res.value

	def SetPreAmpGain(self, index):
		self.logger.debug("SetPreAmpGain %d" % (index))
		with self.lock:
			self._chk(self.dll.SetPreAmpGain(index))

	def GetPreAmpGain(self, index):
		res = c_float()
		with self.lock:
			self._chk(self.dll.GetPreAmpGain(index, byref(res)))
		self.logger.debug("GetPreAmpGain %d -> %f" % (index, res.value))
		return res.value

	def GetEMCCDGain(self):
		res = c_int()
		with self.lock:
			self._chk(self.dll.GetEMCCDGain(byref(res)))
		self.logger.debug("GetEMCCDGain %d" % (res.value))
		return res.value

	# Info ---------------------------------------------------------------------

	def GetInfo(self):
		res = OrderedDict()
		methods = [(self.IsCoolerOn, []),
				   (self.GetTemperature, []),
				   (self.GetAcquisitionTimings, []),
				   (self.GetStatus, []),
				   (self.GetBaselineClamp, []),
				   (self.GetCameraSerialNumber, []),
				   (self.GetDetector, []),
				   (self.GetPixelSize, []),
				   (self.GetFastestRecommendedVSSpeed, []),
				   (self.GetBitDepth, [(0,)]),
				   (self.GetNumberVSSpeeds, []),
				   (self.GetNumberADChannels, []),
				   (self.GetNumberAmp, []),
				   (self.GetNumberPreAmpGains, []),
				   (self.GetPreAmpGain, [(0,), (1,), (2,)]),
				   (self.GetVSSpeed, [(0,), (1,), (2,), (3,)]),
				   (self.GetNumberHSSpeeds, [(0, "EMCCD"), (0, "CCD")]),  # (0, "EMCCD"), (0, "CCD") (1, "EMCCD"), (1, "CCD")
				   (self.GetHSSpeed, [(0, "EMCCD", 0), (0, "CCD", 0), (0, "EMCCD", 1), (0, "CCD", 1), (0, "EMCCD", 2), (0, "CCD", 2)]),  # (0, "EMCCD", 0), (0, "CCD", 0), (1, "EMCCD", 0), (1, "CCD", 0), (1, "EMCCD", 1), (1, "CCD", 1)
				   (self.GetEMCCDGain, [])
				   ]

		for m, args in methods:
			if len(args) == 0:
				res[m.__name__] = m()
			else:
				for arg in args:
					res[m.__name__ + str(arg)] = m(*arg)
		return res

	@property
	def sensorWidth(self):
		# In um
		return self._pixelSize[0] * self._detShape[0]

	@property
	def imageShape(self):
		return self._imageShape


if __name__ == "__main__":
	pass
