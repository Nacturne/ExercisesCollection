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
	// TODO Auto-generated stub, complete

	dataSet = trainDataset;

    if (train) {
        std::cout << "Training with default parameters: " << std::endl;
        softmaxTrain(dataSet);
        std::cout << "Training procedure finised!" << std::endl;
    }

}


MNISTClassifier::~MNISTClassifier() {
	// TODO Auto-generated stub, complete
}


void MNISTClassifier::softmaxTrain(const MNISTDataset& dataSet,
    int iteration, double learningRate, int MiniBatchSize, int regularization) {



    cv::Mat imagesMat;
    for (auto it = dataSet.images.begin(); it != dataSet.images.end(); it++) {
        imagesMat.push_back((*it).reshape(1,1));
    }
    imagesMat.convertTo(imagesMat, CV_32F);

    cv::Mat labelsMat(dataSet.labels);
    labelsMat.convertTo(labelsMat, CV_32F);


    std::string regMethod;
    if (regularization == -1) { regMethod = "Disabled";}
    else if (regularization == 0) {regMethod = "L1 regularization"; }
    else if (regularization == 1) {regMethod = "L2 regularizatoin"; }

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

    softmaxClassifier = cv::ml::LogisticRegression::create();
    softmaxClassifier->setLearningRate(learningRate);
    softmaxClassifier->setIterations(iteration);
    softmaxClassifier->setRegularization(regularization);
    softmaxClassifier->setTrainMethod(cv::ml::LogisticRegression::BATCH);
    softmaxClassifier->setMiniBatchSize(MiniBatchSize);

    softmaxClassifier->train(imagesMat, cv::ml::ROW_SAMPLE, labelsMat);

}


int MNISTClassifier::classifyImage(const cv::Mat& sample) {
	// TODO Auto-generated stub, copmlete
    cv::Mat inputMat = sample.reshape(1,1);
    inputMat.convertTo(inputMat, CV_32F);
    cv::Mat classID;
    softmaxClassifier->predict(inputMat, classID);
	return classID.at<int>(0);
}

MNISTDataset MNISTClassifier::loadDatasetFromFiles(
		const std::string& labelsFile, const std::string& imagesFile) {
	// TODO Auto-generated stub, complete
	MNISTDataset data;

	std::fstream labelStream(labelsFile, std::ios::in | std::ios::binary);
	labelStream.read((char *) &data.labelMagicNumber, sizeof(data.labelMagicNumber));
    data.labelMagicNumber = reverseInt(data.labelMagicNumber);
    labelStream.read((char *) &data.numberOfLabels, sizeof(data.numberOfLabels));
    data.numberOfLabels = reverseInt(data.numberOfLabels);
    for (int i = 0; i < data.numberOfLabels; i++) {
        unsigned char tempLabel = 0;
        labelStream.read((char *) &tempLabel, sizeof(tempLabel));
        (data.labels).push_back((int) tempLabel);
    }
    labelStream.close();


    std::fstream imageStream (imagesFile, std::ios::in | std::ios::binary);
	imageStream.read((char *) &data.imageMagicNumber, sizeof(data.imageMagicNumber));
    data.imageMagicNumber = reverseInt(data.imageMagicNumber);
    imageStream.read((char *) &data.numberOfImages, sizeof(data.numberOfImages));
    data.numberOfImages = reverseInt(data.numberOfImages);
	imageStream.read((char *) &data.rowsOfImage, sizeof(data.rowsOfImage));
    data.rowsOfImage = reverseInt(data.rowsOfImage);
    imageStream.read((char *) &data.ColsOfImage, sizeof(data.ColsOfImage));
    data.ColsOfImage = reverseInt(data.ColsOfImage);
    for (int i = 0; i < data.numberOfImages; i++) {
        cv::Mat tempMat(data.rowsOfImage, data.ColsOfImage, CV_8UC1);
        for (int row = 0; row < data.rowsOfImage; row++) {
            for (int col = 0; col < data.ColsOfImage; col++) {
                unsigned char tempPixel = 0;
                imageStream.read((char *) &tempPixel, sizeof(tempPixel));
                tempMat.at<uchar>(row,col) = (int) tempPixel;
            }
        }
        data.images.push_back(tempMat);
    }
    imageStream.close();

	return data;
}


void MNISTClassifier::runTestDatasetAndPrintStats(const MNISTDataset& inputDataset) {
	// TODO Auto-generated stub, complete
	cv::Mat imagesMat;
    for (auto it = inputDataset.images.begin(); it != inputDataset.images.end(); it++) {
        imagesMat.push_back((*it).reshape(1,1));
    }
    imagesMat.convertTo(imagesMat, CV_32F);
    cv::Mat labelsMat(inputDataset.labels);
    cv::Mat predictions;
    softmaxClassifier->predict(imagesMat, predictions);
    evaluation(labelsMat, predictions);
}




void MNISTClassifier::save(const std::string &fileName) {
    softmaxClassifier->save(fileName);
    std::cout << "The model is saved to: " << fileName << std::endl;
}


void MNISTClassifier::load(const std::string& fileName) {
    std::cout << "Loading a model from: " << fileName << std::endl;
    softmaxClassifier = cv::ml::StatModel::load<cv::ml::LogisticRegression>(fileName);
}




void MNISTClassifier::evaluation(cv::Mat &labelsMat, cv::Mat &predictions) {
    int truePositive[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int falsePositive[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int falseNegative[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int counter[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < labelsMat.rows; i++) {
        int original = labelsMat.at<int>(i);
        int predicted = predictions.at<int>(i);
        counter[original] += 1;

        if (original == predicted) {
            truePositive[original] += 1;
        } else {
            falsePositive[predicted] += 1;
            falseNegative[original] += 1;
        }
    }


    int width = 15;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(5) << "Label";
    std::cout << std::setw(width) << "Precision";
    std::cout << std::setw(width) << "Recall";
    std::cout << std::setw(width) << "F1 Score";
    std::cout << std::setw(width) << "Support" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
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

    float totalAccuracy = (float)cv::countNonZero(labelsMat == predictions) / predictions.rows;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    std::cout << "Total Accuracy: " << totalAccuracy << std::endl;
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
