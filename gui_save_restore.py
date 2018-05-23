from PyQt5.QtCore import QFileInfo, QSettings
from PyQt5.QtWidgets import qApp
from PyQt5 import QtGui#,QtCore

list_gui_save_restore_classes = [QtGui.QDoubleSpinBox,QtGui.QSpinBox,QtGui.QCheckBox,QtGui.QLineEdit,QtGui.QComboBox]
list_gui_save_restore_properties = ['value','text','currentIndex','checked']
def restore_gui(settings):
	finfo = QFileInfo(settings.fileName())

	if finfo.exists() and finfo.isFile():
		for w in qApp.allWidgets():
			mo = w.metaObject()
			if type(w) in list_gui_save_restore_classes:
				#print(w.objectName())

				for i in range(mo.propertyCount()):
					name = mo.property(i).name()
					if name in list_gui_save_restore_properties:
						val = settings.value("{}/{}".format(w.objectName(), name), w.property(name))
						w.blockSignals(True)
						w.setProperty(name, val)
						w.blockSignals(False)


def save_gui(settings):

	for w in qApp.allWidgets():
		mo = w.metaObject()
		if type(w) in list_gui_save_restore_classes:
			for i in range(mo.propertyCount()):
				name = mo.property(i).name()
				if name in list_gui_save_restore_properties:
					settings.setValue("{}/{}".format(w.objectName(), name), w.property(name))
