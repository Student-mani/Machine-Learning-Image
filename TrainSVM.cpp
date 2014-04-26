/********************************************************************************************************************************************


Description: 	The program  reads the file labels.txt which contains the labels of ech train-image. 
		It classifies the images according to the labels. 
		For each label SVM is calculated and is stored.
		.yml files are formed for each label.

		
Compilation: 	1>Below at line 166 provide the path name where images are stored
`		2>In Linux execute: g++ -std=c++0x `pkg-config opencv --cflags` svmtrain.cpp  -o svmtrain `pkg-config opencv --libs`


Execute:        ./svntrain /home/sharmita/Desktop/TagMe/images/labels.txt without_color


Input:		1>Provide the file path to the label.txt which contains only the name of the images along with labels, each in seperate lines.
		2>postfix that is to be added to the output files.


Output:		5 .yml files.


NOTE:		The Gamma and C values are previously computed from the training set by seperate program.


*********************************************************************************************************************************************/







#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// used for claculating the data for diff class and saving the trained svm
void trainSVM(map<string,Mat>& classes_training_data, string& file_postfix, int response_cols, int response_type) {

	//train 1-vs-all SVMs
	vector<string> classes_names;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		classes_names.push_back((*it).first);
	}
	
	string use_postfix = file_postfix;
	for (int i=0;i<classes_names.size();i++) {
		string class_ = classes_names[i];
		//cout << omp_get_thread_num() << " training class: " << class_ << ".." << endl;
		
		Mat samples(0,response_cols,response_type);
		Mat labels(0,1,CV_32FC1);
		
		//copy class samples and label
		cout << "adding " << classes_training_data[class_].rows << " positive" << endl;
		samples.push_back(classes_training_data[class_]);
		Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
		labels.push_back(class_label);
		
		//copy rest samples and label
		for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
			string not_class_ = (*it1).first;
			if(not_class_.compare(class_)==0) continue;
			samples.push_back(classes_training_data[not_class_]);
			class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
			labels.push_back(class_label);
		}
		
		cout << "Train.." << endl;

		CvSVMParams params;
		params.kernel_type=CvSVM::RBF;
		params.svm_type=CvSVM::C_SVC;
		//params.gamma=0.00048215;
		//params.C=0.5;
		params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);

		Mat samples_32f; samples.convertTo(samples_32f, CV_32F);
		if(samples.rows == 0) continue; //phantom class?!
		CvSVM classifier; 
		classifier.train(samples_32f,labels,Mat(),Mat(),params);

		{
			stringstream ss; 
			ss << "SVM_classifier_"; 
			if(file_postfix.size() > 0) ss << file_postfix << "_";
			ss << class_ << ".yml";
			cout << "Save.." << endl;
			classifier.save(ss.str().c_str());
		}
	}
}




int main(int arg,char **argv)
{
	initModule_nonfree();
	cout << "-------- train BOVW SVMs -----------" << endl;
	cout << "read vocabulary form file"<<endl;
	Mat vocabulary;
	FileStorage fsa("newFinal.txt", FileStorage::READ);
	fsa["vocabulary"] >> vocabulary;
	fsa.release();
	
	Ptr<FeatureDetector> featureDetector= Ptr<FeatureDetector>(new SurfFeatureDetector(1000,2,1,true,false));
    	Ptr<DescriptorExtractor>  descriptorExtractor= Ptr<DescriptorExtractor>(new SurfDescriptorExtractor(1000,2,1,true,false));
	//SURF(1000,2,1,true,false);
	//SURF descriptorExtractor(1000,2,1,true,false);
	Ptr<DescriptorMatcher> descriptorMatcher= DescriptorMatcher::create("FlannBased" );
       // Ptr<DescriptorMatcher > descriptorMatcher(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(descriptorExtractor,descriptorMatcher);
	bowide.setVocabulary(vocabulary);
	
	map<string,Mat> classes_training_data; classes_training_data.clear();
	//map<string,int> filename_class;filename_class.clear();

	cout << "look in train data"<<endl;
	Ptr<ifstream> ifs(new ifstream(argv[1]));
	cout << "look in train data"<<endl;
	int total_samples = 0;
	vector<string> classes_names;
	
	char buf[255]; int count = 0;
	vector<string> lines; 
	while(!ifs->eof()) {// && count++ < 30) {
		ifs->getline(buf, 255);
		lines.push_back(buf);
	}
cout<<"Reading all images....";
	for(int i=0;i<lines.size()-1;i++) {
//		printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
//		if(ifs->eof()) break;

		vector<KeyPoint> keypoints;
		Mat response_hist;
		//Mat response_hist = Mat(0, 32, CV_32F);
		Mat img;
		string filepath;

		string line(lines[i]);
		istringstream iss(line);

		iss >> filepath;
		
		string class_; iss >> class_; class_ = "class_" + class_;
		//cout << "  is of " << class_ << endl; 
		if(class_.size() == 0) continue;

		string image;
		image = "/home/manish/Downloads/TagMeData/Train/Images/" + filepath;//provide the path name where images are stored
		//cout <<"the image is  " <<image<<endl; 
		img = imread(image);
		Mat imgl;
		bilateralFilter( img,imgl, 5, 5, 5);
		cvtColor( imgl, imgl, CV_BGR2GRAY );
		equalizeHist( imgl, imgl );
		if(!img.empty()) {
		//imshow("Edge map", img);
		//cout <<"the image is  " <<image<<endl;
		featureDetector->detect(imgl,keypoints);
		//cout <<"the image is  " <<image<<endl;
		bowide.compute(imgl, keypoints, response_hist);//-----------------------------------

		cout << "."; cout.flush();
		{
			//cout << "here" << endl ;
		if(classes_training_data.count(class_) == 0) { //not yet created...
				classes_training_data[class_].create(0,response_hist.cols,response_hist.type());
				classes_names.push_back(class_);
			}
			classes_training_data[class_].push_back(response_hist);
		}
		//if(class_.size() == 0) continue;
		
		}
		total_samples++;
		//else continue;

		
	}
	cout<< "Total number of sample read-----  "<< total_samples<< endl;

	cout << "save to file.."<<endl;
	{
		FileStorage fs("training_samples.txt",FileStorage::WRITE);
		for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
			cout << "save " << (*it).first << endl;
			fs << (*it).first << (*it).second;
		}
	}

	cout << "got " << classes_training_data.size() << " classes." <<endl;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		cout << " class " << (*it).first << " has " << (*it).second.rows << " samples"<<endl;
	}
	
	cout << "train SVMs\n";
	string postfix = argv[2];
	trainSVM(classes_training_data, postfix, bowide.descriptorSize(), bowide.descriptorType());


	return 0;
}
