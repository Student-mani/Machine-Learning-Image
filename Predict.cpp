/********************************************************************************************************************************************


Description: 	The program reads the test images and calculate the descriptors and clustes ans the same way as the train-images.
		Each image is predicted and the score is calculated.
		labels are assigned according to the score.


Compilation: 	In Linux execute: g++ -std=c++0x `pkg-config opencv --cflags` predictfinal.cpp  -o predict `pkg-config opencv --libs`


Execute:        ./predict /home/sharmita/Desktop/Test/Images/imageList.txt 


Input:		1>Provide the file path to the label.txt which contains only the name of the images , each in seperate lines.
		


Output:		ans.txt::this contains each image and its predicted labels(based only on the highest score)


NOTE:		Many times the algorithm provieds two choices when the image is not clear and the score of the 2nd cannot be neglected.

SCOPE OF IMPROVEMENT:	Besides SVM another classifier should be used and both the outputs should be compared to get better results.

*********************************************************************************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <set>
#include <memory>
#include <string>
#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <ml.h>
#include <iostream>
#include <fstream>


//#include <omp.h>

using namespace cv;
using namespace std;


class LablePredictor {
public:	
	LablePredictor();
	void evaluateOneImage(Mat &__img, vector<string> &out_classes);
	map<string,unique_ptr<CvSVM>> & getClassesClassifiers() { return classes_classifiers; }
	

	
private:
	
	void initSVMs();
	void initVocabulary();
	
	bool debug;
	
	Ptr<FeatureDetector > detector;
	Ptr<BOWImgDescriptorExtractor > bowide;
	Ptr<DescriptorMatcher > matcher;
	Ptr<DescriptorExtractor > extractor;
	map<string,unique_ptr<CvSVM>> classes_classifiers;
	Mat vocabulary;
};

LablePredictor::LablePredictor() {
	debug = true;//as said
	initSVMs();
	initVocabulary();
	Ptr<FeatureDetector > _detector= Ptr<FeatureDetector>( new SurfFeatureDetector(1000,2,1,true,false) );
	Ptr<DescriptorMatcher > _matcher= Ptr<DescriptorMatcher>(new FlannBasedMatcher());
	Ptr<DescriptorExtractor > _extractor= Ptr<DescriptorExtractor>(new SurfDescriptorExtractor(1000,2,1,true,false) );
	matcher = _matcher;
	detector = _detector;
	extractor = _extractor;
	bowide = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractor,matcher));
	bowide->setVocabulary(vocabulary);
	//background = imread("background.png");
}




void LablePredictor::initVocabulary() {
	if (debug) cout << "read vocabulary form file"<<endl;
	FileStorage fs("newFinal.txt", FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();	
}


void LablePredictor::initSVMs() {
	string dir, filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	if(debug) cout << "load SVM classifiers" << endl;
	dir = ".";
	dp = opendir( dir.c_str() );
	
	while ((dirp = readdir( dp )))
    {
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		if (filepath.find("SVM_classifier_without_color") != string::npos)
		{
			string class_ = filepath.substr(filepath.rfind('_')+1,filepath.rfind('.')-filepath.rfind('_')-1);
			if (debug) cout << "load " << filepath << ", class: " << class_ << endl;
			
			classes_classifiers.insert(make_pair(class_, unique_ptr<CvSVM>(new CvSVM())));
			classes_classifiers[class_]->load(filepath.c_str());
		}
	}
	closedir(dp);
}

void LablePredictor::evaluateOneImage(Mat &__img, vector<string> &out_classes) {

	vector<Point> check_points;
	//Sliding window approach.. (creating a vector here to ease the OMP parallel for-loop)
	int winsize = 400;
	map<string,pair<int,float> > found_classes;
	for (int x=0; x<__img.cols; x+=winsize/4) {
		for (int y=0; y<__img.rows; y+=winsize/4) {
			check_points.push_back(Point(x,y));
		}
	}
	
	if (debug) cout << "to check: " << check_points.size() << " points"<<endl;
	

	for (int i = 0; i < check_points.size(); i++) {
		int x = check_points[i].x;

		int y = check_points[i].y;
		
		Mat img,response_hist;

		__img(Rect(x-winsize/2,y-winsize/2,winsize,winsize)&Rect(0,0,__img.cols,__img.rows)).copyTo(img);
		vector<KeyPoint> keypoints;
		detector->detect(img,keypoints);
		
										
		bowide->compute(img, keypoints, response_hist);
		if (response_hist.cols == 0 || response_hist.rows == 0) {
 
			continue;
		}
		
		//test vs. SVMs
		
		try {
			float minf = FLT_MAX; string minclass;
			
			for (map<string,unique_ptr<CvSVM>>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
				
				float res = (*it).second->predict(response_hist,true);
			
				if ((*it).first == "misc" && res > 0.9) {
					continue;
				}
				
				if(res > 1.0) continue;
				if (res < minf) {
					minf = res;
					minclass = (*it).first;
					
				}

			
				
			}
			
			float dim = MAX(MIN(minf - 0.8f,0.3f),0.0f) / 0.3f; 
			
			{
			
				found_classes[minclass].first++;
				found_classes[minclass].second += minf;
			}
			
			
		}
		catch (cv::Exception) {
			continue;
		}

			
	}
	
	//if (debug) cout << endl << "found classes: ";
	float max_class_f = FLT_MIN, max_class_f1 = FLT_MIN; string max_class, max_class1;
	vector<float> scores;
	for (map<string,pair<int,float> >::iterator it=found_classes.begin(); it != found_classes.end(); ++it) {
		float score = sqrtf((float)((*it).second.first) * (*it).second.second);
		if (score > 1e+10) {
			continue;	//an impossible score
		}
		scores.push_back(score);
		if (debug) cout << (*it).first << "(" << score << "),"; 
		if(score > max_class_f) { //1st place thrown off
			max_class_f1 = max_class_f;
			max_class1 = max_class;
			
			max_class_f = score;
			max_class = (*it).first;
		} else if (score >  max_class_f1) {	//2nd place thrown off
			max_class_f1 = score;
			max_class1 = (*it).first;
		}
	}
	if (debug) cout << endl;



	Scalar mean_,stddev_;


//	meanStdDev(Mat(scores), mean_, stddev_);
	out_classes.clear();
	out_classes.push_back(max_class);
	cout << max_class << endl;
	if(max_class_f - max_class_f1 < 10) {
		//Forget about it: variance is low (~10), so result is undecicive, we should take both max-classes.
		out_classes.push_back(max_class1);
	}	
	
	//if (debug) cout << "chosen class: " << max_class << ", (" << max_class1 << "?)" << endl;
}

	



int main(int argc, char* argv[]) {
	
	initModule_nonfree();
	LablePredictor predictor;

    string filename=argv[1];//reading the file path.
    cout<<"The filename provided is:: "<<filename<<endl;
    vector<string> names;
    vector<string> filehj;
    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return 0;
    
    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == String::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    string dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;
	cout << "dirName" << "=" <<dirName<< endl;
ofstream fptr("finalans.txt");
        
    while( !file.eof() )
    {
        string str=dirName,str1; getline( file, str1 );
	str=str+str1;

cout<<"processing\n"<<str<<endl;
        Mat __img=imread(str),_img;

	Mat _imgl;
	 bilateralFilter( __img, _imgl, 5, 5, 5);
	cvtColor( _imgl, _imgl, CV_BGR2GRAY );
	equalizeHist( _imgl, _imgl );

	vector<string> max_class;
	predictor.evaluateOneImage(_imgl,max_class);
        

	cout<< "--"<< max_class[0];
	fptr<<str1<<" "<<max_class[0]<<endl;
	
	if (max_class.size()>1) {
		cout << "--" << max_class[1];
	}
       

    }

fptr.close();

	return 0;
}
