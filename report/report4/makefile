CXX := clang++
CXXFLAGS := -O3 --std=c++14

QP = cpp/quadProg/QuadProg++.cc cpp/quadProg/Array.cc

SVRDEPENDS := $(shell echo cpp/{main,svr,svm,kernel,util}.cpp) $(QP)

.PHONY: svr  clean all
all: svr

clean:
	rm svr

svr: $(SVRDEPENDS)
	$(CXX) $(CXXFLAGS) -o svr $(SVRDEPENDS)