/*
 * main.cpp
 *
 *  Created on: Aug 30, 2016
 *      Author: roy_shilkrot
 */



#include "MNISTClassifier.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>



using namespace std;

int main(int argc, char** argv) {

    MNISTDataset trainDataset = MNISTClassifier::loadDatasetFromFiles("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte");
    MNISTDataset testDataset = MNISTClassifier::loadDatasetFromFiles("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte");

    //MNISTClassifier classifier(trainDataset);
    //classifier.runTestDatasetAndPrintStats(testDataset);


/*
    cout<< trainDataset.imageMagicNumber << endl;
    cout<< trainDataset.numberOfImages << endl;
    cout<< trainDataset.rowsOfImage << endl;
    cout<< trainDataset.ColsOfImage << endl;
    cout<< trainDataset.labelMagicNumber << endl;
    cout<< trainDataset.numberOfLabels << endl;
*/



    MNISTClassifier test(testDataset, false);
    test.load("models/test.xml");
    test.runTestDatasetAndPrintStats(testDataset);

    test.load("models/test-300-01-500.xml");
    test.runTestDatasetAndPrintStats(testDataset);

    test.load("models/test-5-05-10.xml");
    test.runTestDatasetAndPrintStats(testDataset);


/*
    MNISTClassifier classifier(trainDataset,false);
    classifier.softmaxTrain(trainDataset, 300, 0.01, 500, -1);
    classifier.runTestDatasetAndPrintStats(testDataset);
    classifier.save("test-300-01-500.xml");
*/

}



/*
using namespace std;

int main(int argc, char** argv) {
	MNISTDataset trainDataset = MNISTClassifier::loadDatasetFromFiles("train-labels-idx1-ubyte", "train-images-idx3-ubyte");
	MNISTDataset testDataset  = MNISTClassifier::loadDatasetFromFiles("t10k-labels-idx1-ubyte",  "t10k-images-idx3-ubyte");

	MNISTClassifier classifier(trainDataset);
	classifier.runTestDatasetAndPrintStats(testDataset);
}


*/










