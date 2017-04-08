# Final-year-project-deep-learning-models
Deep learning for freehand sketch object recognition 

## Abstract 
In this project, we propose a novel deep learning architecture that achieves state-of-the-art results in free-hand sketch object recognition. The highly iconic and abstract nature of sketch objects make it hard task for a computer algorithm to recognition them.  As sketch recognition is not a new concept in computer vision, we conducted a detailed study of the previous works related to our project domain. The hand-crafted models failed to capture the iconic nature of sketches. And the existing deep learning architectures are tailored to photo images and do not adopt to the varying levels of abstraction present in sketch objects. This resulted in sketch-A-Net which surpassed human level accuracy. Sketch-A-Net requires stroke order information to accurately recognize sketch objects. The framework only considers real time sketch inputs and cannot handle a large dataset of sketch objects available online.  All the above research discoveries resoundingly stressed to adopt a new deep learning architecture which is tailored to solve sketch recognition.
<br/>

Our model is designed on the Hebbian principle which states that neurons that are coupled together, activate together. We address common issues that are overlooked in previous works regarding a new deep learning model design. We solve overfitting problems of wider network by introducing a sparse structure of convolutional blocks in our model. We engineer the model to solve sketch object iconic and abstract nature by using large number of training samples. Our model is trained on the TU-Berlin sketch dataset which consists of 20,000 objects from 250 categories. We apply data-augmentation techniques on the dataset to elastically increase its size. Our model achieves a ground breaking recognition accuracy of 84.7% which is ~10% more than its predecessors. Then, we deployed our model on a cloud platform and set-up a web application to process sketch recognition requests. Even though our model achieves a high accuracy, it still fails to recognize the intra-class deformations. This points out that our model still has room for improvement.
<br/>

By successfully solving sketch recognition, we can now move towards solving multi-object recognition, sketch object segmentation, image retrieval based on sketch query and the most popular current trend in computer vision, the use of Generative Adversarial Networks to synthesis sketch objects or use a sketch object to synthesis a complete photo realistic image. The possibilities in this domain is endless and we plan to visit and continue our research in deep learning for free-hand sketch objects in the future.     
<br/>

## Condor Job submit

1. Login to GPU cluster and place the condor job files in your public forlder.

2. To run the condor job, use the following command:
  ```
    condor_submit sketch.sub
  ```
3. To check error, output and log of the submitted job, use the following command:
  ```
    cat sketch.out 
    cat sketch.log
    cat skecth.error
  ```
4. To kill job, use the following command:
  ```
    condor_q ## get job ID
    kill job_id
  ```
  
## To start the web application
  Navigate to the web application folder and open terminal and execute the below code:
  ```
    python manage.py runserver
  ```
  ## NGROK tunneling
  
  To set up public IP for the DJANGO application, we open NGROK and excetue the following command for the respective port number(application specific port number):
  
    ```
    ngrok port_number
    
    ```
  


## References


[1] 	C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke and A. Rabinovich, "Going Deeper with Convolutions," The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1-9, 2015. 

[2] 	R. Hua and J. Collomosse, "A Performance Evaluation of Gradient Field HOG Descriptor for Sketch Based," Computer Vision and Image Understanding, vol. Volume 117 , no. 7, pp. 790-806 , 2013. 

[3] 	S. Ouyang, T. Hospedales, Y.-Z. Song and X. Li, "Cross-Modal Face Matching: Beyond Viewed Sketches," Computer Vision -- ACCV 2014, vol. 9004, pp. 210-225, 2014. 

[4] 	R. G. Schneider and T. Tuytelaars, "Sketch classification and classification-driven analysis using Fisher vectors," TOG ACM Trans. Graph. ACM Transactions on Graphics, pp. 1-9, 2014. 

[5] 	Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998. 

[6] 	Yu, Yang, Song, Xiang and Hospedales, "Sketch-a-Net that Beats Humans," Procedings of the British Machine Vision Conference 2015, 2015. 

[7] 	L. T, T. C, S. F and C. S, A new recognition model for, 2015. 

[8] 	J. G, G. MD, H. J and Y.-L. D. E, Computational support for sketching in design a review. Foundation and Trends in Human-Computer Interaction, 2009. 

[9] 	J. MFA, R. MSM, O. NZS and J. Z, "A comparative study on extraction and recognition method of cad data from cad drawings.," in International Conference on Information Management and Engineering, 2009. 

[10] 	Eitz, M. a. Hays, J. a. Alexa and Marc, "How Do Humans Sketch Objects?," ACM Trans. Graph. (Proc. SIGGRAPH), vol. 31, no. 4, pp. 44:1--44:10, 2012. 

[11] 	R. Galiazzi Schneider and T. Tuytelaars, "Sketch classification and classification-driven analysis using fisher vectors," Proceedings of SIGGRAPH Asia 2014, vol. 33, no. 6, pp. 1-9, 2014. 

[12] 	Z. Sun, C. Wang, L. Zhang and L. Zhang, "Free Hand-Drawn Sketch Segmentation," Microsoft Research Asia, Beijing, 2012.

[13] 	A. Krizhevsky, I. Sutskever and G. E. Hinton, "Imagenet classification with deep convolutional neural networks," in Advances in neural information processing systems, 2012. 

[14] 	A. Krizhevsky, I. Sutskever and G. E. Hinton, "ImageNetClassiﬁcationwithDeepConvolutional NeuralNetworks," Conference on Neural Information Processing Systems (NIPS) , 2012. 

[15] 	C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke and A. Rabinovich, "GoingDeeperwithConvolutions," Computer Vision and Patteren Recognition, 2015. 

[16] 	K. Simonyan and A. Zisserman, "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION," International Conference on Learning Representations, 2015. 

[17] 	C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhouck and A. Rabinovich, "Going Deeper with Convolutions," CVPR, 2015. 

[18] 	M.-M. . Poo and R. . Fitzsimonds, "Retrograde Signaling in the Development and Modification of Synapses," Psychological Reviews, vol. , no. , p. , . 

[19] 	S. Arora, A. Bhaskara, R. Ge and T. Ma, "ProvableBoundsforLearningSomeDeepRepresentations," CoRR, 2013.

[20] 	D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in 3rd International Conference for learning representation, San Diego, 2015. 

[21] 	L. Fei-Fei, J. Deng and K. Li, "ImageNet: Constructing a large-scale image database," Journal of Vision, vol. 9, no. 8, pp. 1037-1037, 2010. 

[22] 	P. Dollar, "Fast Edge Detection Using Structured Forests," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 8, pp. 1558-1570, 2015. 

[23] 	Anaconda Software Distribution, Computer software. Vers. 2-2.4.0. Continuum Analytics, 2016. 
[24] 	F. Chollet, keras, \url{https://github.com/fchollet/keras}, 2015. 

[25] 	G. E. Krasner and S. T. Pope, "A cookbook for using the model–view controller user interface paradigm in Smalltalk-80," The Journal of Object Technology, vol. , no. , p. , . 

[26] 	D. L. Parnas and P. C. Clements, "A rational design process: How and why to fake it," Software Engineering, IEEE Transactions, vol. , no. , p. 251–257, .

[27] 	JDONAHUE, JIAYQ, VINYALS, JHOFFMAN, NZHANG, ETZENG and TREVOR, "DeCAF: A Deep Convolutional Activation Feature," 2013. 

[28] 	R. Girshick, J. Donahue, T. Darrell and J. Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation," Tech Report, UC Berkely, Berkely, 2014.

[29] 	J. Uijlings, "Selective Search for Object Recognition," IJCV, Netherlands, 2012.

[30] 	"Global Infrastructure," , . [Online]. Available: https://aws.amazon.com/about-aws/global-infrastructure/. [Accessed 1 4 2017].

[31] 	M. a. H. J. a. A. M. Eitz, "How Do Humans Sketch Objects?," ACM Trans. Graph. (Proc. SIGGRAPH), vol. 31, no. 4, pp. 44:1--44:10, 2012. 

[32] 	Y. . LeCun, "LeNet-5, convolutional neural networks," , . [Online]. Available: http://yann.lecun.com/exdb/lenet/. [Accessed 2 4 2017].

[33] 	H. Li Y, S. TM and G. S. Y, "Freehand sketch recognition by multi-kernel learning," CVIU, 2015.

[34] 	S. Li Y and G. S. Y, "Sketch recognition by ensemble matchning of structured features," BMVC, 2013. 


