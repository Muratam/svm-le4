CXX := clang++
CXXFLAGS := -O3 --std=c++14

QP = cpp/quadProg/QuadProg++.cc cpp/quadProg/Array.cc

SVRDEPENDS := $(shell echo cpp/{main,svr,svm,kernel,util}.cpp) $(QP)
SVMDEPENDS := $(shell echo cpp/{svm_main,svm,kernel,util}.cpp) $(QP)

.PHONY: svr svm clean all
all: svr svm

clean:
	rm svr svm

svr: $(SVRDEPENDS)
	$(CXX) $(CXXFLAGS) -o svr $(SVRDEPENDS)

svm: $(SVMDEPENDS)
	$(CXX) $(CXXFLAGS) -o svm $(SVMDEPENDS)

