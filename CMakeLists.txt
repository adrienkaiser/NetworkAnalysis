CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(NetworkAnalysis)  # name of the project

FIND_PACKAGE(GenerateCLP REQUIRED)
IF(GenerateCLP_FOUND)
	INCLUDE(${GenerateCLP_USE_FILE})
ELSE(GenerateCLP_FOUND)
	MESSAGE(FATAL_ERROR "GenerateCLP not found. Please set GenerateCLP_DIR")
ENDIF(GenerateCLP_FOUND)

SET(sources NetworkAnalysis.cxx) #define the variable "sources" that contains the names of files

GENERATECLP(sources NetworkAnalysis.xml) # include the GCLP file to the project

ADD_EXECUTABLE(NetworkAnalysis ${sources})  # add the files contained by "sources" to the project
