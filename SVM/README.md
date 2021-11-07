In this project, I train and test the SVM algorithm on the *Spam vs. Non-spam* and *Atheism vs. Religion* text classification tasks. I also implement an experimental evaluation of SVMs and the perceptron algorithm, with and without kernels, on the problem of classifying images representing digits. Please view sections **2** and **3** in the *SVM.pdf* file for project details and the *Report.pdf* file for implementation results.

### 1\. **Text Classiﬁcation** 

Train and test the SVM algorithm on the *Spam vs. Non-spam* and *Atheism vs. Religion*
classiﬁcation problems, using the datasets provided for the previous assignment. Use a linear
kernel, with the cost parameter C = 5. Report and compare the accuracy of the trained SVM
models with the perceptron and average perceptron accuracies from the [perceptron assignment](https://github.com/taiman9/Machine-learning/tree/master/perceptron).

### 2\. **Digit Recognition** 

In this exercise, you are asked to run an experimental evaluation of SVMs and the perceptron
algorithm, with and without kernels, on the problem of classifying images representing digits.

1\. The UCI Machine Learning Repository at *www.ics.uci.edu/˜mlearn* maintains datasets
for a wide variety of machine learning problems. For this assignment, you are supposed
to work with the Optical Recognition of Handwritten Digits Data Set. The webpage
for this dataset is at:

*http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits*

The actual dataset is located at:

*http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/*

Read the description of the dataset. Download the training set **optdigits.tra** and the
test set **optdigits.tes**. Use the ﬁrst 1000 examples in **optdigits.tra** for development
and the rest of 2823 examples for training. Use all 1797 examples in **optdigits.tes**
for testing. Scale all the features between [0, 1], using the minand max computed over the 
training examples. Create training files for each of the 10 digits, setting the class to 1 
for instances of that digit, and to -1 for instances of other digits, i.e. one-vs-rest scenario.

2\. Train ﬁrst the linear perceptron, with the number of epochs set to *T* ∈ {1, 2, ..., 20}.
After training each linear perceptron, normalize the learned weight vector. Select for
T the value that obtains the best overall accuracy on the development data, and use
this value for the remaining perceptron experiments.

Run experiments with the linear and kernel perceptron algorithms. For the kernel
perceptron, experiment with polynomial kernels ***k***(**x**,**y**) = (1 + **x**<sup>T</sup>**y**)<sup>**d**</sup> with degrees
***d*** ∈ {2, 3, 4, 5, 6}, and with Gaussian kernels ***k***(**x**,**y**) = *exp*(−||**x** − **y**||<sup>2</sup>/2σ<sup>2</sup>) with the
width σ ∈ {0.1, 0.5, 2, 5, 10}. For each hyper-parameter value, you will have trained 10 models, 
one for each digit. In order to compute the label for a test or development example, you will run the 10 trained models and output the label that obtains the highest score. Compute the accuracy on the development data and identify the hyperparameter value that obtains 
the best accuracy. Use the tuned hyper-parameter (*d* for poly-kernel, σ for Gaussian) to compute the overall performance on the test data. For each of the three perceptrons (linear, poly kernel and Gaussian kernel) report the total training time, the overall accuracy, and the number of support vectors. Show and compare the corresponding 4 confusion matrices. Which digit seems to be the hardest
to classify? Which perceptron / kernel combination achieves the best performance? Which algorithms are slower at training time, and why?

3\. Run the same experiments using SVMs instead of perceptrons, i.e. linear SVMs and
SVMs with polynomial and Gaussian kernels. Use the same tuning scenarios for the
hyper-parameters of the polynomial and Gaussian kernels. Use C = 1 in all SVM
experiments. Report the same types of results and analysis as above, and compare
with the perceptron results.

### Obtain Results

Make sure you include a **README.txt** ﬁle explaining how the code is supposed to be used to replicate the results 
included in the report. The screen output produced when running the code should be redirected to (saved into) an **output.txt** ﬁle.
