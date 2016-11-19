/*
 * main.cpp
 *
 *  Created on: Aug 30, 2016
 *      Author: roy_shilkrot
 */

#include "MNISTClassifier.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char** argv) {

    MNISTDataset trainDataset = MNISTClassifier::loadDatasetFromFiles("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte");
    MNISTDataset testDataset = MNISTClassifier::loadDatasetFromFiles("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte");

/*
    //output the attribute date to check whether it works
    cout<< trainDataset.imageMagicNumber << endl;
    cout<< trainDataset.numberOfImages << endl;
    cout<< trainDataset.rowsOfImage << endl;
    cout<< trainDataset.ColsOfImage << endl;
    cout<< trainDataset.labelMagicNumber << endl;
    cout<< trainDataset.numberOfLabels << endl;
*/

    //MNISTClassifier classifier(trainDataset);
    //classifier.runTestDatasetAndPrintStats(testDataset);




    MNISTClassifier test(testDataset, false);
    test.load("models/test.xml");
    test.runTestDatasetAndPrintStats(testDataset);

    test.load("models/test-300-01-500.xml");
    test.runTestDatasetAndPrintStats(testDataset);

    test.load("models/test-5-05-10.xml");
    test.runTestDatasetAndPrintStats(testDataset);

    cout << test.classifyImage(trainDataset.images[0]) << endl;



    MNISTClassifier classifier(trainDataset,false);
    classifier.softmaxTrain(trainDataset, 3, 0.01, 500, -1);
    classifier.runTestDatasetAndPrintStats(testDataset);
    classifier.save("models/temp.xml");

    return 0;

}


