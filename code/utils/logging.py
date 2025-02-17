# coding: utf-8

from logging import getLogger,FileHandler,StreamHandler,DEBUG,Formatter
import os.path,sys

def get_logger(log_path=None):
	logger = getLogger("__main__")
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	if log_path is None:
		handler = StreamHandler(sys.stdout)
	else:
		handler = FileHandler(filename=log_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	logger.info("Logger set up.")
	return logger