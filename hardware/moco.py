import visa
import time

class MOCO():
	inst = None
	position = 0
	def __init__(self, id='ASRL7::INSTR'):
		rm = visa.ResourceManager()
		self.inst =  rm.open_resource(id,baud_rate=9600)
		self.getPosition()

	def close(self):
		self.inst.close()

	def getPosition(self):
		p = self.inst.query("'")
		self.position = int(p.split(':')[1])
		return self.position

	def moveRel(self, incr, waitUntilReady=False):
		self.inst.write('mr'+str(incr))
		if waitUntilReady:
			while self.isMoving():
				time.sleep(0.01)
		pos = self.getPosition()
		return pos
	def moveAbs(self, pos, waitUntilReady=False):
		self.inst.write('ma'+str(pos))
		if waitUntilReady:
			while self.isMoving():
				time.sleep(0.01)
				#print(moco.getPosition())
		pos = self.getPosition()
		return pos

	def isMoving(self):
		p = self.inst.query("%")
		try:
			val = int(p.split(':')[1].split(' ')[0])
		except:
			return True
		if val==0:
			return True
		else:
			return False

	def calibr(self,waitUntilReady=False):
		self.inst.write('mc2')
		if waitUntilReady:
			while self.isMoving():
				time.sleep(0.01)
		pos = self.getPosition()
		return pos

	def __del__(self):
		self.close()

if __name__=='__main__':
	moco = MOCO()
	print(moco.getPosition())
	print(moco.calibr(True))
	print(moco.moveAbs(100000,True))
	#moco.close()
