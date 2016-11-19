/*
 * MNISTClassifier.cpp
 *
 *  Created on: Aug 30, 2016
 *      Author: roy_shilkrot
 */

#include "MNISTClassifier.h"
#include <iostream>
#include <iomanip>
#include <fstream>

MNISTClassifier::MNISTClassifier(const MNISTDataset& trainDataset, bool train) {

	dataSet = trainDataset;

    // If 'train' flag is 'true', train the model use default parameters (see softmaxTrain() for default values)
    if (train) {
        std::cout << "Training with default parameters: " << std::endl;
        softmaxTrain(dataSet);
        std::cout << "Training procedure finised!" << std::endl;
    }

}


MNISTClassifier::~MNISTClassifier() {
}


void MNISTClassifier::softmaxTrain(const MNISTDataset& dataSet,
    int iteration, double learningRate, int MiniBatchSize, int regularization) {

    // Iterate the image dataset.
    // Reshape the 28x28 image to 1x784 vector,
    // so that it can be used as the input to the softmax classifier
    // Then convert it from int to float to satisfy the requirement of the softmax classifier.
    cv::Mat imagesMat;
    for (auto it = dataSet.images.begin(); it != dataSet.images.end(); it++) {
        imagesMat.push_back((*it).reshape(1,1));
    }
    imagesMat.convertTo(imagesMat, CV_32F);

    // Create a cv::Mat instance to hold the lables
    // Feed cv::Mat constructor a vector<int>, it will construct a one-column matrix automatically.
    // Then convert it from int to float to satisfy the requirement of the softmax classifier.
    cv::Mat labelsMat(dataSet.labels);
    labelsMat.convertTo(labelsMat, CV_32F);


    // Find the regularization method specified by the caller.
    std::string regMethod;
    if (regularization == -1) { regMethod = "Disabled";}
    else if (regularization == 0) {regMethod = "L1 regularization"; }
    else if (regularization == 1) {regMethod = "L2 regularizatoin"; }

    // Print out the parameters used to train the modle.
    std::cout << "A new model is initialized ..." << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Seting for training parameters:" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Training Method: \t" << "MiniBatch Gradient" << std::endl;
    std::cout << "Iteration: \t\t" << iteration << std::endl;
    std::cout << "Learning Rate: \t\t" << learningRate << std::endl;
    std::cout << "MiniBatch Size: \t" << MiniBatchSize << std::endl;
    std::cout << "Regularization: \t" << regMethod << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Please wait for training ...\n\n" << std::endl;

    // Initialize the classifier with the parameters specified by the caller.
    softmaxClassifier = cv::ml::LogisticRegression::create();
    softmaxClassifier->setLearningRate(learningRate);
    softmaxClassifier->setIterations(iteration);
    softmaxClassifier->setRegularization(regularization);
    softmaxClassifier->setTrainMethod(cv::ml::LogisticRegression::BATCH);
    softmaxClassifier->setMiniBatchSize(MiniBatchSize);

    // Train the model.
    // imagesMat holds the images in each row.
    // labelsMat holds the labels in each row, corresponding to the order of images in the imagesMat.
    softmaxClassifier->train(imagesMat, cv::ml::ROW_SAMPLE, labelsMat);

}


int MNISTClassifier::classifyImage(const cv::Mat& sample) {
    // Reshape the 28x28 image to 1x784 vector,
    // so that it can be used as the input to the softmax classifier.
    // Then convert it from int to float.
    cv::Mat inputMat = sample.reshape(1,1);
    inputMat.convertTo(inputMat, CV_32F);

    // Make the prediction and save the predicted label in a cv::Mat called 'classID'.
    cv::Mat classID;
    softmaxClassifier->predict(inputMat, classID);

    // Return the predicted label.
	return classID.at<int>(0);
}


MNISTDataset MNISTClassifier::loadDatasetFromFiles(
		const std::string& labelsFile, const std::string& imagesFile) {
    // Initialize a MNISTDataset instance to store the returned value of this method.
	MNISTDataset data;

    // Initialize a file stream to read the binary file of labels.
	std::fstream labelStream(labelsFile, std::ios::in | std::ios::binary);

	// Read in the magic number in high-endian format.
	// Then reverse it to get the correct value.
	labelStream.read((char *) &data.labelMagicNumber, sizeof(data.labelMagicNumber));
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
    std::fstream imageStream (imagesFile, std::ios::in | std::ios::binary);

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

        // Initialize a temporary cv::Mat to store the i-th image.
        cv::Mat tempMat(data.rowsOfImage, data.ColsOfImage, CV_8UC1);

        for (int row = 0; row < data.rowsOfImage; row++) {
            for (int col = 0; col < data.ColsOfImage; col++) {

                // Read the pixel (row,col) for the i-th image.
                unsigned char tempPixel = 0;
                imageStream.read((char *) &tempPixel, sizeof(tempPixel));

                // Assign it to the corresponding position in the image matrix.
                tempMat.at<uchar>(row,col) = (int) tempPixel;
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
	cv::Mat imagesMat;
    for (auto it = inputDataset.images.begin(); it != inputDataset.images.end(); it++) {
        imagesMat.push_back((*it).reshape(1,1));
    }
    imagesMat.convertTo(imagesMat, CV_32F);

    // Create a cv::Mat instance to hold the lables
    // Feed cv::Mat constructor a vector<int>, it will construct a one-column matrix automatically.
    cv::Mat labelsMat(inputDataset.labels);

    // Initialized a cv::Mat instance to store the predicted labels.
    // Then make predicition use the softmax classifier.
    cv::Mat predictions;
    softmaxClassifier->predict(imagesMat, predictions);

    // Print out the statistics for the performance of the classifier.
    evaluation(labelsMat, predictions);
}


void MNISTClassifier::save(const std::string &fileName) {

    // save the model to '.xml' format, using the save() method of cv::ml::LogisticRegression.
    softmaxClassifier->save(fileName);
    std::cout << "The model is saved to: " << fileName << std::endl;
}


void MNISTClassifier::load(const std::string& fileName) {
    std::cout << "Loading a model from: " << fileName << std::endl;

    // load the model from '.xml' format, using the load() method of cv::ml::StatModel
    softmaxClassifier = cv::ml::StatModel::load<cv::ml::LogisticRegression>(fileName);
}



void MNISTClassifier::evaluation(cv::Mat &labelsMat, cv::Mat &predictions) {

    // Initialize three int arrays to store the ture positive, false positive and false negative for each class
    // For example, truePositive[i] stores the true positive value for label 'i'
    // We fill in these arrays when we traverse the prediction result.
    // In this way, all the three values can be calculated by one pass, with a time complexity of O(n).
    int truePositive[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int falsePositive[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int falseNegative[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Save the number of each label.
    int counter[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Traverse the predicted results.
    for (int i = 0; i < labelsMat.rows; i++) {

        // Get the ground truth and the predicted label
        int original = labelsMat.at<int>(i);
        int predicted = predictions.at<int>(i);
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
    int width = 15;
    std::cout << "Statistics for Classifier Performance:" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(5) << "Label";
    std::cout << std::setw(width) << "Precision";
    std::cout << std::setw(width) << "Recall";
    std::cout << std::setw(width) << "F1 Score";
    std::cout << std::setw(width) << "Support" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    // Iterate the 9 classes to calculate the precision, recall and F1 score;
    for (int i = 0; i < 10; i++) {
        float precision = (float)truePositive[i] / (truePositive[i] + falsePositive[i]);
        float recall = (float)truePositive[i] / (truePositive[i] + falseNegative[i]);
        float f1Score = 2 * precision * recall / (precision + recall);

        std::cout << std::setw(5) << i;
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4);
        std::cout << std::setw(width) << precision;
        std::cout << std::setw(width) << recall;
        std::cout << std::setw(width) << f1Score;
        std::cout << std::setw(width) << counter[i] << std::endl;
    }

    // Calculate total accuracy.
    float totalAccuracy = (float)cv::countNonZero(labelsMat == predictions) / predictions.rows;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    std::cout << "Total Accuracy: \t" << totalAccuracy << std::endl;
    std::cout << "Total ErrorRate:\t" << (1-totalAccuracy) << std::endl;
    std::cout << "----------------------------------------------------------------------------\n\n" << std::endl;
}









int MNISTClassifier::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}
