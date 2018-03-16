import pprint
import nidaqmx
import time
from nidaqmx.constants import AcquisitionType, TaskMode

pp = pprint.PrettyPrinter(indent=4)


#with nidaqmx.Task() as master_task:
master_task = nidaqmx.Task()
master_task.ai_channels.add_ai_voltage_chan("Dev1/ai0,Dev1/ai2")

master_task.timing.cfg_samp_clk_timing(
	10000, sample_mode=AcquisitionType.CONTINUOUS)

master_task.control(TaskMode.TASK_COMMIT)

master_task.triggers.start_trigger.cfg_dig_edge_start_trig("PFI0")


master_task.start()
start = time.time()
for i in range(100):
	master_data = master_task.read(number_of_samples_per_channel=200)

r,d = master_data
#pp.pprint(master_data)
print(time.time()-start)
master_task.close()
from pylab import *
plot(r[:-6],'-b')
plot(d[6:],'-r')
show(0)
