svr: cpp/main.cpp cpp/quadProg/QuadProg++.cc cpp/svr.h cpp/svr.cpp cpp/svm.h cpp/svm.h cpp/svm.h cpp/svm.cpp cpp/kernel.cpp cpp/kernel.h
	g++ -O3 --std=c++14 -o svr cpp/main.cpp cpp/quadProg/QuadProg++.cc cpp/svr.cpp cpp/svm.cpp cpp/kernel.cpp
svm: cpp/svm_main.cpp cpp/quadProg/QuadProg++.cc cpp/svm.h cpp/svm.cpp cpp/kernel.cpp cpp/kernel.h
	g++ -O3 --std=c++14 -o svm cpp/svm_main.cpp cpp/svm.cpp cpp/quadProg/QuadProg++.cc cpp/kernel.cpp