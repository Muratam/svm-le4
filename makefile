svr: cpp/main.cpp cpp/quadProg/QuadProg++.cc
	g++ -O3 --std=c++14 -o svr cpp/main.cpp cpp/quadProg/QuadProg++.cc
svm: cpp/svm_main.cpp cpp/quadProg/QuadProg++.cc
	g++ -O3 --std=c++14 -o svm cpp/svm_main.cpp cpp/quadProg/QuadProg++.cc