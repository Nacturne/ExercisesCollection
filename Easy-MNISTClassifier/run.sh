cp ../../test/main.cpp main.cpp
cp ../../test/MNISTClassifier.cpp MNISTClassifier.cpp
cp ../../test/MNISTClassifier.h MNISTClassifier.h
make clean
make 
./output/MNISTClassifier
