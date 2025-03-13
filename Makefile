all: 
	python maldetect.py cnnbigru-functions
	python maldetect.py cnnbigru-dlls
	python maldetect.py cnnbigru-sections

CNN_BiGRU_func:
	python maldetect.py cnnbigru-functions

CNN_BiGRU_dlls:
	python maldetect.py cnnbigru-dlls

CNN_BiGRU_sections:
	python maldetect.py cnnbigru-sections
