/**
 * @file MNISTClassifier.h
 *
 * @date 2016/11/19
 * @author roy_shilkrot, Xiao Liang
 */

#ifndef MNISTCLASSIFIER_H_
#define MNISTCLASSIFIER_H_

#include <string>
#include <opencv2/opencv.hpp>

struct MNISTDataset {
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

	/**
	 * The constructor will construct a MNISTClassifier from a MNISTDataset instance.
	 * By default, it will train a softmax classifier.
	 *
	 * @param trainDataset The dataset used to construct the instance.
	 * @param train The flag indicating whether to traing thesoftmax classifier. Set it to 'false' when load a pre-trained model outside.
	 */
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


    /**
	 * Train a softmax classifier with user-specified parameters and MiniBatch Gradient method.
	 *
	 * @param dataSet Training dataset, in the format of a MNISTDataset struct.
	 * @param iteration The iteration number.
	 * @param learningRate The learning rate used for training.
	 * @param MiniBatchSize The bach size used for MiniBatch Gradient method.
	 * @param regularization The method for regularization: -1 for disabled; 0 for L1 regularizaton; 1 for L2 regularization.
	 */
	void softmaxTrain(const MNISTDataset &dataSet,
        int iteration = 300, double learningRate = 0.01, int MiniBatchSize = 100, int regularization = 1);

    /**
     * Save the trained model to a file specified by user. Model would be saved in 'xml' format.
     * @param fileName The file name to save the model. The recomended extension is '.xml'.
     */
	void save(const std::string& fileName);

    /**
     * load a pre-trained softmax model.
     * @param fileName The file in which the model is saved.
     */
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

	/**
	 * The MNISTClass image data are stored in a high-endian binary format. To read a 4-byte integer correctly,
	 * the program firstly read from high byte to low byte, then inverse the position of bytes. This method is an auxiliary
	 * function to help us reading the 4-byte integers.
	 *
	 * @param A read-in 4 byte integer from high-endian binary format.
	 * @param The correct value by reversing the bytes of the input integer.
	 */
	static int reverseInt(int i);
};

#endif /* MNISTCLASSIFIER_H_ */
