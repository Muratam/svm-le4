all:
	g++ -O3 --std=c++14 -o svm cpp/main.cpp cpp/quadProg/QuadProg++.cc