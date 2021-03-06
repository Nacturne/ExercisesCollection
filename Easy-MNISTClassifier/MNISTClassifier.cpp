/**
 * @file MNISTClassifier.cpp
 *
 * @date 2016/11/19
 * @author roy_shilkrot, Xiao Liang
 */

#include "MNISTClassifier.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace cv;

MNISTClassifier::MNISTClassifier(const MNISTDataset& trainDataset, const bool train) {

	dataSet = trainDataset;

    // If 'train' flag is 'true', train the model use default parameters (see softmaxTrain() for default values)
    if (train) {
        cout << "Training with default parameters: " << endl;
        softmaxTrain(dataSet);
        cout << "Training procedure finised!"        << endl;
    }

}


MNISTClassifier::~MNISTClassifier() {
}


void MNISTClassifier::softmaxTrain(const MNISTDataset& dataSet,
                                   const int           iteration,
                                   const double        learningRate,
                                   const int           MiniBatchSize,
                                   const int           regularization) {

    // Iterate the image dataset.
    // Reshape the 28x28 image to 1x784 vector,
    // so that it can be used as the input to the softmax classifier
    // Then convert it from int to float to satisfy the requirement of the softmax classifier.
    Mat imagesMat;
    for (const Mat& m : dataSet.images) {
        imagesMat.push_back(m.reshape(1,1));
    }
    imagesMat.convertTo(imagesMat, CV_32F);

    // Create a Mat instance to hold the lables
    // Feed Mat constructor a vector<int>, it will construct a one-column matrix automatically.
    // Then convert it from int to float to satisfy the requirement of the softmax classifier.
    Mat labelsMat(dataSet.labels);
    labelsMat.convertTo(labelsMat, CV_32F);


    // Find the regularization method specified by the caller.
    string regMethod;
    if        (regularization == -1) {
        regMethod = "Disabled";
    } else if (regularization == 0)  {
        regMethod = "L1 regularization";
    } else if (regularization == 1)  {
        regMethod = "L2 regularizatoin";
    }

    // Print out the parameters used to train the modle.
    cout << "A new model is initialized ..."                   << endl
         << "++++++++++++++++++++++++++++++++++++++++++++++++" << endl
         << "Seting for training parameters:"                  << endl
         << "------------------------------------------------" << endl
         << "Training Method: \t" << "MiniBatch Gradient"      << endl
         << "Iteration: \t\t" << iteration                     << endl
         << "Learning Rate: \t\t" << learningRate              << endl
         << "MiniBatch Size: \t" << MiniBatchSize              << endl
         << "Regularization: \t" << regMethod                  << endl
         << "++++++++++++++++++++++++++++++++++++++++++++++++" << endl
         << "Please wait for training ...\n\n"                 << endl;

    // Initialize the classifier with the parameters specified by the caller.
    softmaxClassifier = ml::LogisticRegression::create();

    softmaxClassifier->setLearningRate  (learningRate);
    softmaxClassifier->setIterations    (iteration);
    softmaxClassifier->setRegularization(regularization);
    softmaxClassifier->setTrainMethod   (ml::LogisticRegression::BATCH);
    softmaxClassifier->setMiniBatchSize (MiniBatchSize);

    // Train the model.
    // imagesMat holds the images in each row.
    // labelsMat holds the labels in each row, corresponding to the order of images in the imagesMat.
    softmaxClassifier->train(imagesMat, ml::ROW_SAMPLE, labelsMat);

}


int MNISTClassifier::classifyImage(const Mat& sample) const {
    // Reshape the 28x28 image to 1x784 vector,
    // so that it can be used as the input to the softmax classifier.
    // Then convert it from int to float.
    Mat inputMat = sample.reshape(1,1);
    inputMat.convertTo(inputMat, CV_32F);

    // Make the prediction and save the predicted label in a Mat called 'classID'.
    Mat classID;
    softmaxClassifier->predict(inputMat, classID);

    // Return the predicted label.
	return classID.at<int>(0);
}


MNISTDataset MNISTClassifier::loadDatasetFromFiles(const string& labelsFile, const string& imagesFile) {
    // Initialize a MNISTDataset instance to store the returned value of this method.
	MNISTDataset data;

    // Initialize a file stream to read the binary file of labels.
	ifstream labelStream(labelsFile, ios::binary);


	// Read in the magic number in high-endian format.
	// Then reverse it to get the correct value.
	labelStream >> data.labelMagicNumber;
    data.labelMagicNumber = reverseInt(data.labelMagicNumber);

    // Read in the total number of labels in high-endian format.
	// Then reverse it to get the correct value.
    labelStream.read((char *) &data.numberOfLabels, sizeof(data.numberOfLabels));
    data.numberOfLabels = reverseInt(data.numberOfLabels);

    // After the field for magic number and total number, the remainings are labels.
    // Read in labels using a for loop
    for (int i = 0; i < data.numberOfLabels; i++) {
        unsigned char tempLabel = 0;
        labelStream.read((char *) &tempLabel, sizeof(tempLabel));
        (data.labels).push_back((int) tempLabel);
    }

    // Close the stream after using.
    labelStream.close();


    // Initialize a file stream to read the binary file of images.
    ifstream imageStream (imagesFile, ios::binary);

    // Read in attributes of this dataset in the first 4 fields.
    // This is similiar as what I did for label set in the above.
	imageStream.read((char *) &data.imageMagicNumber, sizeof(data.imageMagicNumber));
    data.imageMagicNumber = reverseInt(data.imageMagicNumber);
    imageStream.read((char *) &data.numberOfImages, sizeof(data.numberOfImages));
    data.numberOfImages = reverseInt(data.numberOfImages);
	imageStream.read((char *) &data.rowsOfImage, sizeof(data.rowsOfImage));
    data.rowsOfImage = reverseInt(data.rowsOfImage);
    imageStream.read((char *) &data.ColsOfImage, sizeof(data.ColsOfImage));
    data.ColsOfImage = reverseInt(data.ColsOfImage);

    // After the first 4 fields the remainings are image pixels.
    // Read in iamges using a for loop
    for (int i = 0; i < data.numberOfImages; i++) {

        // Initialize a temporary Mat to store the i-th image.
        Mat tempMat(data.rowsOfImage, data.ColsOfImage, CV_8UC1);

        for (int row = 0; row < data.rowsOfImage; row++) {
            for (int col = 0; col < data.ColsOfImage; col++) {

                // Read the pixel (row,col) for the i-th image.
                unsigned char tempPixel = 0;
                imageStream.read((char *) &tempPixel, sizeof(tempPixel));

                // Assign it to the corresponding position in the image matrix.
                tempMat.at<uchar>(row, col) = (int) tempPixel;
            }
        }

        // add the newly-read image to the vector of images.
        data.images.push_back(tempMat);
    }

    // Close the stream after using.
    imageStream.close();

	return data;
}


void MNISTClassifier::runTestDatasetAndPrintStats(const MNISTDataset& inputDataset) {
    // Iterate the vector of images to reshape it from 28x28 to 1x784,
    // so that it can be used as the input to the softmax classifier.
    // Then convert it from int to float.
	Mat imagesMat;
    for (const Mat& image : inputDataset.images) {
        imagesMat.push_back(image.reshape(1,1));
    }
    imagesMat.convertTo(imagesMat, CV_32F);

    // Create a Mat instance to hold the lables
    // Feed Mat constructor a vector<int>, it will construct a one-column matrix automatically.
    Mat labelsMat(inputDataset.labels);

    // Initialized a Mat instance to store the predicted labels.
    // Then make predicition use the softmax classifier.
    Mat predictions;
    softmaxClassifier->predict(imagesMat, predictions);

    // Print out the statistics for the performance of the classifier.
    evaluation(labelsMat, predictions);
}


void MNISTClassifier::save(const string &fileName) {

    // save the model to '.xml' format, using the save() method of ml::LogisticRegression.
    softmaxClassifier->save(fileName);
    cout << "The model is saved to: " << fileName << endl;
}


void MNISTClassifier::load(const string& fileName) {
    cout << "Loading a model from: " << fileName << endl;

    // load the model from '.xml' format, using the load() method of ml::StatModel
    softmaxClassifier = ml::StatModel::load<ml::LogisticRegression>(fileName);
}



void MNISTClassifier::evaluation(const Mat &labelsMat, const Mat &predictions) {

    // Initialize three int arrays to store the ture positive, false positive and false negative for each class
    // For example, truePositive[i] stores the true positive value for label 'i'
    // We fill in these arrays when we traverse the prediction result.
    // In this way, all the three values can be calculated by one pass, with a time complexity of O(n).
    int truePositive [10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int falsePositive[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int falseNegative[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    // Save the number of each label.
    int counter[10] =       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    // Traverse the predicted results.
    for (int i = 0; i < labelsMat.rows; i++) {

        // Get the ground truth and the predicted label
        const int original = labelsMat.at<int>(i);
        const int predicted = predictions.at<int>(i);

        // Increment the counter for the true label
        counter[original] += 1;

        if (original == predicted) {
            // When the predicted label equals the true label, it is a true positive for the class of true label.
            truePositive[original] += 1;
        } else {
            // If the predicted label is wrong, it is a false positive for the class of the predicted label.
            // Also, it is a false negative for the class of the true label.
            falsePositive[predicted] += 1;
            falseNegative[original] += 1;
        }
    }

    // Print out the precision, racall and F1 score in a well-formatted way.
    const int width = 15;
    cout << "Statistics for Classifier Performance:"                                       << endl
         << "----------------------------------------------------------------------------" << endl
         << setw(5)     << "Label"
         << setw(width) << "Precision"
         << setw(width) << "Recall"
         << setw(width) << "F1 Score"
         << setw(width) << "Support"                                                       << endl
         << "----------------------------------------------------------------------------" << endl;

    // Iterate the 9 classes to calculate the precision, recall and F1 score;
    for (int i = 0; i < 10; i++) {
        const float precision = (float)truePositive[i] / (truePositive[i] + falsePositive[i]);
        const float recall    = (float)truePositive[i] / (truePositive[i] + falseNegative[i]);
        const float f1Score   = 2 * precision * recall / (precision + recall);

        cout << setw(5) << i;
        cout << setiosflags(ios::fixed) << setprecision(4);
        cout << setw(width) << precision;
        cout << setw(width) << recall;
        cout << setw(width) << f1Score;
        cout << setw(width) << counter[i] << endl;
    }

    // Calculate total accuracy.
    float totalAccuracy = (float)countNonZero(labelsMat == predictions) / predictions.rows;
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "Total Accuracy: \t" << totalAccuracy << endl;
    cout << "Total ErrorRate:\t" << (1-totalAccuracy) << endl;
    cout << "----------------------------------------------------------------------------\n\n" << endl;
}

int MNISTClassifier::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i         & 255;
    c2 = (i >> 8 ) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}
