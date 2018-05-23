"""
Status codes for Andor cameras.

"""

__all__ = ["ANDOR_CODES",
           "ANDOR_STATUS"]

# Status code -> status message
ANDOR_CODES = {
    20001: "DRV_ERROR_CODES",
    20002: "DRV_SUCCESS",
    20003: "DRV_VXDNOTINSTALLED",
    20004: "DRV_ERROR_SCAN",
    20005: "DRV_ERROR_CHECK_SUM",
    20006: "DRV_ERROR_FILELOAD",
    20007: "DRV_UNKNOWN_FUNCTION",
    20008: "DRV_ERROR_VXD_INIT",
    20009: "DRV_ERROR_ADDRESS",
    20010: "DRV_ERROR_PAGELOCK",
    20011: "DRV_ERROR_PAGE_UNLOCK",
    20012: "DRV_ERROR_BOARDTEST",
    20013: "DRV_ERROR_ACK",
    20014: "DRV_ERROR_UP_FIFO",
    20015: "DRV_ERROR_PATTERN",
    20017: "DRV_ACQUISITION_ERRORS",
    20018: "DRV_ACQ_BUFFER",
    20019: "DRV_ACQ_DOWNFIFO_FULL",
    20020: "DRV_PROC_UNKNOWN_INSTRUCTION",
    20021: "DRV_ILLEGAL_OP_CODE",
    20022: "DRV_KINETIC_TIME_NOT_MET",
    20022: "DRV_KINETIC_TIME_NOT_MET",
    20023: "DRV_ACCUM_TIME_NOT_MET",
    20024: "DRV_NO_NEW_DATA",
    20026: "DRV_SPOOLERROR",
    20027: "DRV_SPOOLSETUPERROR",
    20033: "DRV_TEMPERATURE_CODES",
    20034: "DRV_TEMPERATURE_OFF",
    20035: "DRV_TEMP_NOT_STABILIZED",
    20036: "DRV_TEMPERATURE_STABILIZED",
    20037: "DRV_TEMPERATURE_NOT_REACHED",
    20038: "DRV_TEMPERATURE_OUT_RANGE",
    20039: "DRV_TEMPERATURE_NOT_SUPPORTED",
    20040: "DRV_TEMPERATURE_DRIFT",
    20049: "DRV_GENERAL_ERRORS",
    20050: "DRV_INVALID_AUX",
    20051: "DRV_COF_NOTLOADED",
    20052: "DRV_FPGAPROG",
    20053: "DRV_FLEXERROR",
    20054: "DRV_GPIBERROR",
    20064: "DRV_DATATYPE",
    20065: "DRV_DRIVER_ERRORS",
    20066: "DRV_P1INVALID",
    20067: "DRV_P2INVALID",
    20068: "DRV_P3INVALID",
    20069: "DRV_P4INVALID",
    20070: "DRV_INIERROR",
    20071: "DRV_COFERROR",
    20072: "DRV_ACQUIRING",
    20073: "DRV_IDLE",
    20074: "DRV_TEMPCYCLE",
    20075: "DRV_NOT_INITIALIZED",
    20076: "DRV_P5INVALID",
    20077: "DRV_P6INVALID",
    20078: "DRV_INVALID_MODE",
    20079: "DRV_INVALID_FILTER",
    20080: "DRV_I2CERRORS",
    20081: "DRV_DRV_I2CDEVNOTFOUND",
    20082: "DRV_I2CTIMEOUT",
    20083: "DRV_P7INVALID",
    20089: "DRV_USBERROR",
    20090: "DRV_IOCERROR",
    20091: "DRV_VRMVERSIONERROR",
    20093: "DRV_USB_INTERRUPT_ENDPOINT_ERROR",
    20094: "DRV_RANDOM_TRACK_ERROR",
    20095: "DRV_INVALID_TRIGGER_MODE",
    20096: "DRV_LOAD_FIRMWARE_ERROR",
    20097: "DRV_DIVIDE_BY_ZERO_ERROR",
    20098: "DRV_INVALID_RINGEXPOSURES",
    20099: "DRV_BINNING_ERROR",
    20990: "DRV_ERROR_NOCAMERA",
    20991: "DRV_NOT_SUPPORTED",
    20992: "DRV_NOT_AVAILABLE",
    20115: "DRV_ERROR_MAP",
    20116: "DRV_ERROR_UNMAP",
    20117: "DRV_ERROR_MDL",
    20118: "DRV_ERROR_UNMDL",
    20119: "DRV_ERROR_BUFFSIZE",
    20121: "DRV_ERROR_NOHANDLE",
    20130: "DRV_GATING_NOT_AVAILABLE",
    20131: "DRV_FPGA_VOLTAGE_ERROR",
    20099: "DRV_BINNING_ERROR",
    20100: "DRV_INVALID_AMPLIFIER",
    20101: "DRV_INVALID_COUNTCONVERT_MODE"}
    
# Status message -> status code
ANDOR_STATUS = {
    "DRV_LOAD_FIRMWARE_ERROR": 20096,
    "DRV_DIVIDE_BY_ZERO_ERROR": 20097,
    "DRV_INVALID_RINGEXPOSURES": 20098,
    "DRV_BINNING_ERROR": 20099,
    "DRV_INVALID_AMPLIFIER": 20100,
    "DRV_ERROR_UNMDL": 20118,
    "DRV_ERROR_MAP": 20115,
    "DRV_ERROR_UNMAP": 20116,
    "DRV_ERROR_MDL": 20117,
    "DRV_NOT_AVAILABLE": 20992,
    "DRV_ERROR_BUFFSIZE": 20119,
    "DRV_ERROR_NOHANDLE": 20121,
    "DRV_INVALID_COUNTCONVERT_MODE": 20101,
    "DRV_ERROR_CODES": 20001,
    "DRV_SUCCESS": 20002,
    "DRV_VXDNOTINSTALLED": 20003,
    "DRV_ERROR_SCAN": 20004,
    "DRV_ERROR_CHECK_SUM": 20005,
    "DRV_ERROR_FILELOAD": 20006,
    "DRV_UNKNOWN_FUNCTION": 20007,
    "DRV_ERROR_VXD_INIT": 20008,
    "DRV_ERROR_ADDRESS": 20009,
    "DRV_ERROR_PAGELOCK": 20010,
    "DRV_ERROR_PAGE_UNLOCK": 20011,
    "DRV_ERROR_BOARDTEST": 20012,
    "DRV_ERROR_ACK": 20013,
    "DRV_ERROR_UP_FIFO": 20014,
    "DRV_ERROR_PATTERN": 20015,
    "DRV_ACQUISITION_ERRORS": 20017,
    "DRV_ACQ_BUFFER": 20018,
    "DRV_ACQ_DOWNFIFO_FULL": 20019,
    "DRV_PROC_UNKNOWN_INSTRUCTION": 20020,
    "DRV_ILLEGAL_OP_CODE": 20021,
    "DRV_KINETIC_TIME_NOT_MET": 20022,
    "DRV_ACCUM_TIME_NOT_MET": 20023,
    "DRV_NO_NEW_DATA": 20024,
    "DRV_SPOOLERROR": 20026,
    "DRV_SPOOLSETUPERROR": 20027,
    "DRV_TEMPERATURE_CODES": 20033,
    "DRV_TEMPERATURE_OFF": 20034,
    "DRV_TEMP_NOT_STABILIZED": 20035,
    "DRV_TEMPERATURE_STABILIZED": 20036,
    "DRV_TEMPERATURE_NOT_REACHED": 20037,
    "DRV_TEMPERATURE_OUT_RANGE": 20038,
    "DRV_TEMPERATURE_NOT_SUPPORTED": 20039,
    "DRV_TEMPERATURE_DRIFT": 20040,
    "DRV_FPGA_VOLTAGE_ERROR": 20131,
    "DRV_GATING_NOT_AVAILABLE": 20130,
    "DRV_GENERAL_ERRORS": 20049,
    "DRV_INVALID_AUX": 20050,
    "DRV_COF_NOTLOADED": 20051,
    "DRV_FPGAPROG": 20052,
    "DRV_FLEXERROR": 20053,
    "DRV_GPIBERROR": 20054,
    "DRV_DATATYPE": 20064,
    "DRV_DRIVER_ERRORS": 20065,
    "DRV_P1INVALID": 20066,
    "DRV_P2INVALID": 20067,
    "DRV_P3INVALID": 20068,
    "DRV_P4INVALID": 20069,
    "DRV_INIERROR": 20070,
    "DRV_COFERROR": 20071,
    "DRV_ACQUIRING": 20072,
    "DRV_IDLE": 20073,
    "DRV_TEMPCYCLE": 20074,
    "DRV_NOT_INITIALIZED": 20075,
    "DRV_P5INVALID": 20076,
    "DRV_P6INVALID": 20077,
    "DRV_INVALID_MODE": 20078,
    "DRV_INVALID_FILTER": 20079,
    "DRV_I2CERRORS": 20080,
    "DRV_DRV_I2CDEVNOTFOUND": 20081,
    "DRV_I2CTIMEOUT": 20082,
    "DRV_P7INVALID": 20083,
    "DRV_ERROR_NOCAMERA": 20990,
    "DRV_NOT_SUPPORTED": 20991,
    "DRV_USBERROR": 20089,
    "DRV_IOCERROR": 20090,
    "DRV_VRMVERSIONERROR": 20091,
    "DRV_USB_INTERRUPT_ENDPOINT_ERROR": 20093,
    "DRV_RANDOM_TRACK_ERROR": 20094,
    "DRV_INVALID_TRIGGER_MODE": 20095}

