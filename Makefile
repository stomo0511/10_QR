UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
	# CXX = g++-7
	CXX = icpc
	CXXFLAGS = -m64 -openmp -O3
	LIB_DIR = /opt/intel/compilers_and_libraries/linux/lib/intel64
	MKL_ROOT = /opt/intel/compilers_and_libraries/linux/mkl
	MKL_INC_DIR = $(MKL_ROOT)/include
	MKL_LIB_DIR = $(MKL_ROOT)/lib/intel64
endif
ifeq ($(UNAME),Darwin)
	CXX = g++-10
	CXXFLAGS = -m64 -fopenmp -O3
	OMP_LIB_DIR = /opt/intel/oneapi/compiler/latest/mac/compiler/lib
	OMP_LIB_ADD = -Wl,-rpath,$(OMP_LIB_DIR)
	OMP_LIBS = -liomp5
	MKL_LIB_ADD = -Wl,-rpath,$(MKLROOT)/lib
	MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core 
	MYROOT = /Users/stomo/WorkSpace/C++
	MY_UTIL_DIR = $(MYROOT)/00_Utils
endif

LIBS = -pthread -lm -ldl

OBJS =	BlockQR.o $(MY_UTIL_DIR)/Utils.o

TARGET = BlockQR

all:	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) \
	-L$(MKLROOT)/lib $(MKL_LIB_ADD) $(MKL_LIBS) \
	-L$(OMP_LIB_DIR) $(OMP_LIB_ADD) $(OMP_LIBS) \
	$(LIBS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(MKLROOT)/include  -I$(MY_UTIL_DIR) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
