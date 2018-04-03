from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import time
import sys

import multiprocessing
from picoscope import ps3000a
import time


class Pico_recorder(multiprocessing.Process):
	ps = None
	def __init__(self):
		super(Pico_recorder, self).__init__()


	def run(self):
		ps = ps3000a.PS3000a(connect=False)
		ps.open()
		n_captures = 1
		ps.setChannel("A", coupling="DC", VRange=0.5)
		ps.setChannel("B", coupling="DC", VRange=0.5)
		capture_duration = 0.003
		sample_interval = 0.00001
		ps.setSamplingInterval(sample_interval,capture_duration)
		ps.setSimpleTrigger(trigSrc="B", threshold_V=-0.320, direction='Falling',
								 timeout_ms=10, enabled=True,delay=120)
		max_samples_per_segment = ps.memorySegments(n_captures)
		samples_per_segment = int(capture_duration / sample_interval)
		ps.setNoOfCaptures(n_captures)

		print('start')
		for i in range(10):
			ps.flashLed(1)
			time.sleep(1)
		print('End')

if __name__ == "__main__":
	__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

	number = 7
	result = None

	#p1 = multiprocessing.Process(target=pico_proc, args=(ps,))
	p1 = Pico_recorder()
	p1.start()
	p1.join()
