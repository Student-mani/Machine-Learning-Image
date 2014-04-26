/********************************************************************************************************************************************


Description: 	The function reads all the train images. Find out the descriptors of each image using SURF after noise elimination and 			scaling. 
		Uses Bag Of Words to cluster the data and store them in a file.


Compilation: 	In Linux execute: g++ -std=c++0x `pkg-config opencv --cflags` test.cpp  -o test `pkg-config opencv --libs`


Execute:        ./test /file /path/to /imagelist.txt

Input:		Provide the file path to the image List which contains only the name of the images in seperate lines.
		NOTE:imagelist should be present in the same directory as the folder containing the train images.



Output:		newfinal.txt::which is the vocabulary formed.



*********************************************************************************************************************************************/












#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;

/*function for extracting the path of the images*/


static void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == String::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;
	cout << dirName << "=" << endl;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

/*function for reading data from the dir*/



static void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,vector<KeyPoint>& trainKeypoints,string& trainDirName,vector<string>& trainImageNames,const string& trainFilename )
{

	cout << "< Reading the images..." << endl;

	readTrainFilenames( trainFilename, trainDirName, trainImageNames );
        if( trainImageNames.empty() )
        {
        	cout << "Train image filenames can not be read." << endl << ">" << endl;
        
        }
        cout << endl << "< Extracting keypoints from images..." << endl;



	SURF descriptorExtractor(1000,2,1,true,false);	//hessian threshold?,octave number?,number of octave?,,whether orientation will be calculated//SURF used to detect the descriptors
	SURF featureDetector(1000,2,1,true,false);

	TermCriteria terminate_criterion;
        terminate_criterion.epsilon = FLT_EPSILON;
        BOWKMeansTrainer bowtrainer( 1000, terminate_criterion, 3, KMEANS_PP_CENTERS );	//Band Of Words //

	int readImageCount = 0;
	

        for( size_t i = 0; i < trainImageNames.size(); i++ )
        {
        	string filename = trainDirName + trainImageNames[i];
		if(trainImageNames[i]==filename)continue;
        	Mat img = imread( filename);
		Mat _img;
		if( img.empty() )
	            cout << "Train image " << filename << " can not be read." << endl;
	        else
		{


			//Reduce noise ad scaling
		        bilateralFilter( img, _img, 5, 5, 5);
			cvtColor( _img, _img, CV_BGR2GRAY );
			equalizeHist( _img, _img );

        
		
	    		Mat descriptor;
			cout << ".";cout.flush();
		        readImageCount++;
	    		featureDetector( _img,Mat(), trainKeypoints );
	    		descriptorExtractor.compute( _img, trainKeypoints, descriptor );
	   
			int descount = descriptor.rows;
			//forr testing purpose
			for(int i=0;i<descount;i++)
			{
				bowtrainer.add(descriptor.row(i));   //is bowtrainer the best 
				}
		}
        }
	cout <<endl;

        if( !readImageCount )
        {
                 cout << "All train images can not be read." << endl << ">" << endl;
        }
        else
		cout << ">" << endl;
	
		Mat vocabulary = bowtrainer.cluster();

		FileStorage fs1("newFinal.txt", FileStorage::WRITE);//the clusters of the train images are stored here......
		fs1 << "vocabulary" << vocabulary;
		fs1.release();
}



int main(int arg,char **argv)
{

	initModule_nonfree();
	string fileWithTrainImages=argv[1];

	Mat queryImage;
	vector<Mat> trainImages;
	vector<string> trainImagesNames;
	string trainDir;
	vector<KeyPoint> queryKeypoints;
    	vector<KeyPoint> trainKeypoints;
	detectKeypoints( queryImage, queryKeypoints,trainKeypoints,trainDir,trainImagesNames,fileWithTrainImages );


return 0; 
	

}
