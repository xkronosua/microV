import visa
import time

class MOCO():
	inst = None
	position = 0
	rm = None
	id = None
	def __init__(self, id='ASRL7::INSTR'):
		self.rm = visa.ResourceManager()
		self.id = id
		self.init()
	def init(self):
		self.inst =  self.rm.open_resource(self.id,baud_rate=9600)
		p = self.inst.query("*IDN?")
		print(p)
		self.getPosition()

	def close(self):
		self.inst.close()

	def getPosition(self):
		p = self.inst.query("'")
		#p = self.inst.query("'")
		#print(p)
		try:
			self.position = int(p.split(':')[1])
		except:
			print('PosErr:',p)
			return None
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
			print(":",p)
			x = p.split(':')[1].split(' ')[0]
			if x =='4C':
				return True
			elif x == '0C':
				return True
			elif x == '40':
				return True
			#elif x == '04':
			#	return True
			val = int(x)
			#print(p,val)
		except:
			print('isMovingErr:',p)
			return False
		#p = self.inst.query("%")
		#print(p)
		if val==0:
			return True
		else:
			return False
	def reset(self):
		self.inst.close()
		del self.rm
		self.rm = visa.ResourceManager()
		self.init()

	def calibr(self,waitUntilReady=False):
		try:
			r = self.inst.query('mc2')
		except:
			print('Timeout')
		time.sleep(3)
		if waitUntilReady:
			while True:
				s = self.isMoving()
				if not s:
					s = self.isMoving()
					print(s)
					try:
						r = self.inst.query('ab2')
					except:
						print('Timeout')
					break
				time.sleep(0.01)
				#pos = self.getPosition()
				#print(pos)

		pos = self.getPosition()
		#self.reset()
		return pos

	def __del__(self):
		self.close()

if __name__=='__main__':
	try: del moco
	except: pass
	moco = MOCO()
	print(moco.getPosition())
	print(moco.getPosition())
	#print(moco.calibr(True))
	print(moco.moveAbs(200000,True))
	time.sleep(3)
	#print(moco.calibr(True))
	print('='*10)
	print(moco.moveAbs(100000,True))
	time.sleep(3)
	print('---'*10)

	print(moco.calibr(True))
	del moco
	moco = MOCO()
	print(moco.moveAbs(150000,True))
	moco.close()
