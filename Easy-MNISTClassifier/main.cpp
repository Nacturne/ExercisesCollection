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

    // Read in the training set.
    MNISTDataset trainDataset = MNISTClassifier::loadDatasetFromFiles("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte");
    // Read in the test set.
    MNISTDataset testDataset = MNISTClassifier::loadDatasetFromFiles("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte");


    //output the attribute date to check whether it works.
    cout << "---------------------------------------------------" << endl;
    cout << "Information for images in training set" << endl;
    cout << "---------------------------------------------------" << endl;
    cout << "magic number:      " << trainDataset.imageMagicNumber << endl;
    cout << "number of images:  " << trainDataset.numberOfImages << endl;
    cout << "number of rows:    " << trainDataset.rowsOfImage << endl;
    cout << "number of columns: " << trainDataset.ColsOfImage << endl;
    cout << "---------------------------------------------------\n\n" << endl;

    cout << "---------------------------------------------------" << endl;
    cout << "Information for labels in training set" << endl;
    cout << "---------------------------------------------------" << endl;
    cout << "magic number:      " << trainDataset.labelMagicNumber << endl;
    cout << "number of labels:  " <<trainDataset.numberOfLabels << endl;
    cout << "---------------------------------------------------\n\n" << endl;



    // Load a pre-trained model and test its performance on test dataset:
    // Initialize a MNISTClassifer so that it wn't train the model automatically.
    MNISTClassifier test1(testDataset, false);
    // Load the pre-trained model.
    test1.load("models/model_defalt.xml");
    // Run the model and print out the statistics for performance.
    test1.runTestDatasetAndPrintStats(testDataset);


/*
    //train a model with the default settings and save it to "models/model_default.xml"
    MNISTClassifier classifier(trainDataset);
    classifier.runTestDatasetAndPrintStats(testDataset);
    classifier.save("models/model_defalt_2.xml");
*/


    // Train a  model with user-defined settings for parameters:
    // Initialize a MNISTClassifer so that it wn't train the model automatically.
    MNISTClassifier test2(trainDataset,false);
    // Train the model use our own settings:
    // Iteration = 10
    // Learning Rate = 0.01
    // MiniBatch size = 500
    // Regularization = disabled
    test2.softmaxTrain(trainDataset, 3, 0.01, 500, -1);
    // Run the model and print out the statistics for performance.
    test2.runTestDatasetAndPrintStats(testDataset);
    // Save the model to "models/test-3-01-500.xml".
    test2.save("models/test-3-01-500.xml");


    return 0;

}
