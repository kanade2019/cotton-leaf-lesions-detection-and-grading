These are the main codes of the essay "A two-stage registering method for UAV multiple spectral imagery with application to cotton leaf lesions grading". 
 
We split these codes into three departments: 
  1. registration: 
    a. Coarse registration used SIFT. Input the folder of images and the result folder. This code can output the images which have coarse registration in the result folder. 
    b. Refined registration is template matching. We used a new correlation coefficient and the detail of it can see in the essay. 
  2. detection: 
    use_model.py is coding based on the efficientdet Official code which can be seen in "https://github.com/google/automl.git". 
    Put this file in the master directory, then the codes of efficientdet can be called by C++. 
    We don't change the official codes too much, so uploading them agin is unnecessary. 
  3. grading: 
    There are 2 code files. One is building svm model and the other is using it. 
    The data used for training is organized in Excel format. Each line has 7 data and 1 label.

PS: all codes in this repository are scattered, so they can't be used if there only have them.
