/*
Code Blocks:
#ifndef MNISTCLASSIFIER_H_INCLUDED
#define MNISTCLASSIFIER_H_INCLUDED

#endif // MNISTCLASSIFIER_H_INCLUDED
*/


/*
 * MNISTClassifier.h
 *
 *  Created on: Aug 30, 2016
 *      Author: roy_shilkrot
 */

#ifndef MNISTCLASSIFIER_H_
#define MNISTCLASSIFIER_H_

#include <string>
#include <opencv2/core/core.hpp>



#include <opencv2/opencv.hpp>

struct MNISTDataset {
	//TODO: add data fields as nessecary
	//e.g. std::vector<cv::Mat> images;
	//     std::vector<int>     labels;
    int imageMagicNumber = 0;
    int numberOfImages = 0;
    int rowsOfImage = 0;
    int ColsOfImage = 0;
    std::vector<cv::Mat> images;

    int labelMagicNumber = 0;
    int numberOfLabels = 0;
	std::vector<int> labels;
};

class MNISTClassifier {
public:
	MNISTClassifier(const MNISTDataset& trainDataset, bool train = true);
	virtual ~MNISTClassifier();

	/**
	 * Classify a single sample image.
	 *
	 * @param sample The sample to classify.
	 * @returns A class ID.
	 */
	int classifyImage(const cv::Mat& sample);

	/**
	 * Run the classifier over the test dataset and print out relevant statistics.
	 */
	void runTestDatasetAndPrintStats(const MNISTDataset& inputDataset);

	/**
	 * Load the MNIST dataset from the labels and images files, as described in the bottom of:
	 * http://yann.lecun.com/exdb/mnist/
	 *
	 * @param labelsFile The labels file.
	 * @param imagesFile The images file.
	 * @returns The MNIST dataset struct populated.
	 */
	static MNISTDataset loadDatasetFromFiles(const std::string& labelsFile, const std::string& imagesFile);



	void softmaxTrain(const MNISTDataset &dataSet,
        int iteration = 100, double learningRate = 0.01, int MiniBatchSize = 100, int regularization = 1);

	void save(const std::string& fileName);
	void load(const std::string& fileName);



private:
	//... add member variables here
	MNISTDataset dataSet;
	cv::Ptr<cv::ml::LogisticRegression> softmaxClassifier;

    /**
     * Print out Precision, Recall and F1 score for each class:
     *
     * @param labelsMat The labels of ground truth, taking the form of a column matrix.
     * @param predictions The labels predicted by our model, taking the form of a column matrix.
     */
	void evaluation(cv::Mat &labelsMat, cv::Mat &predictions);

	static int reverseInt(int i);
};

#endif /* MNISTCLASSIFIER_H_ */
