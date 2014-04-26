Machine-Learning-Image
======================

Read-Learn from a set of images and categorises between different set of images according to the Images learned

This project was created by me and my parten Sharmita Das

The project is where by given a set of images by the user the program learns from the image and eventually train itself.

So, When we give a image for testing it predict as to which class or set the image belonged to.

/************************************************************************************************************************/


Description: 	The function reads all the train images. Find out the descriptors of each image using SURF after noise elimination and 			scaling. 
		Uses Bag Of Words to cluster the data and store them in a file.


Compilation: 	In Linux execute: g++ -std=c++0x `pkg-config opencv --cflags` test.cpp  -o test `pkg-config opencv --libs`


Execute:        ./test /file /path/to /imagelist.txt

Input:		Provide the file path to the image List which contains only the name of the images in seperate lines.
		NOTE:imagelist should be present in the same directory as the folder containing the train images.



Output:		newfinal.txt::which is the vocabulary formed.



*************************************************************************************************************************/

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


