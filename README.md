Download Link: https://assignmentchef.com/product/solved-cs373-homework-2-decisiontree
<br>
<h1></h1>

<h1>Part 0 Specification</h1>

<h2>Install Python3.7</h2>

You can install anaconda to configure the Python environment for Mac/Win/Linux at https:

//www.anaconda.com/distribution/#download-section. Make sure you install the right version, Python-3.7-64-bit installer is preferred. Then install the related packages, type this command in your terminal:

conda install pandas numpy scipy matplotlib

If you are not fully familiar with Python language, we recommend you to go though online tutorials before doing your homework. Recommended websites for you: https:// www.programiz.com/python-programming/tutorial and https://www.codecademy.com/ learn/learn-python.

You can also try out Jupyter Notebook<sup>1 </sup>for real-time coding, which is quite similar to R.

<h2>Dataset Details</h2>

Find these files:

titanic-train.data, titanic-train.label, titanic-test.data, titanic-test.label

in the HW2.zip file. The overall attributes of dataset is shown in Table. 1. The definition of every feature is:

Survived: dead or survive (Bool);

Pclass: Class of Travel (Int);

Name: name of passenger(String);

Sex: Gender(Bool);

Age: Age of Passengers(Int);

Relatives: Number of relatives (Int);

IsAlone: If he/she has no relatives (Bool);

Ticket: ticket number (String);

Fare: Passenger fare (Float);

Embarked: Port of Embarkation (Int). C = Cherbourg (0), Q = Queenstown(0), S = Southampton(77);

Table 1: Samples in titanic raw data.

https://jupyter.org/install

We consider 7 of the attributes as the input features: Pclass, Sex, Age, Relatives, IsAlone, Fare, Embarked. And consider the first attribute Survived as the class label. As shown in Table. 2, this is what you will read from the csv file.

Table 2: Sample data in the titanic-train.data, titanic-train.label.

In order to read the CSV data and obtain our features and labels with the necessary attributes, you can use the following code:

import pandas as pd

X = pd.read_csv(data_file, delimiter = ‘,’, index_col=None, engine=’python’)

<h2>Input format</h2>

Your python script should take the following arguments:

<ol>

 <li>train-file: path to the training set. (train-file.data, train-file.label)</li>

 <li>test-file: path to the test set. (test-file.data, test-file.label)</li>

 <li>model: model that you want to use. In this case, we will use:</li>

</ol>

− vanilla: the full decision tree.

− depth: the decision tree with static depth.

− min-split: the decision tree with minimum samples to split on.

− prune: the decision tree with post-pruning.

<ol start="4">

 <li>train-set-size: percentage of dataset used for training.</li>

</ol>

Each case may have some additional command-line arguments, which will be mentioned in its format section. Use following examples to get the list of arguments.

import sys for x in sys.argv:

print(‘arg: ‘, x)

Your code should read the training set from train-file, extract the required features, train your decision tree on the training set, and test it on the test set from test-file. Name your file ID3.py.

For debugging purposes, you can use a small fraction of the dataset, for example, by using X[:100] to work with the first 100 data points.

<h2>Model Details</h2>

<strong>Entropy</strong>

To help you build the model easier, we provide sample code for you to calculate the entropy:

import numpy as np def entropy(freqs):

“”” entropy(p) = -SUM (Pi * log(Pi))

&gt;&gt;&gt; entropy([10.,10.])

1.0

&gt;&gt;&gt; entropy([10.,0.])

0

&gt;&gt;&gt; entropy([9.,3.])

0.811278

“”” all_freq = sum(freqs) entropy = 0 for fq in freqs:

prob = ____ * 1.0 / ____ if abs(prob) &gt; 1e-8:

entropy += -____ * np.log2(____)

return entropy where you need to pass the test case:

−1<em>/</em>2 ∗ log(1<em>/</em>2) − 1<em>/</em>2 ∗ log(1<em>/</em>2) = 1

−1 ∗ log(1) − 0 ∗ log(0) = 0

−3<em>/</em>4 ∗ log(3<em>/</em>4) − 1<em>/</em>4 ∗ log(1<em>/</em>4) = 0<em>.</em>811278

<strong>Information gain </strong>we provide sample code for you to calculate the information gain:

def infor_gain(before_split_freqs, after_split_freqs):

“””

gain(D, A) = entropy(D) – SUM ( |Di| / |D| * entropy(Di) )

&gt;&gt;&gt; infor_gain([9,5], [[2,2],[4,2],[3,1]])

0.02922

“””

gain = entropy(____) overall_size = sum(____) for freq in after_split_freqs: ratio = sum(____) * 1.0 / ____ gain -= ratio * entropy(___) return gain

where you need to pass the test case:

entropy(D) = 9<em>/</em>14 ∗ log(9<em>/</em>14) − 5<em>/</em>14 ∗ log(5<em>/</em>14) = 0<em>.</em>9402

entropy(Income=high) = −2<em>/</em>4 ∗ log(2<em>/</em>4) − 2<em>/</em>4 ∗ log(2<em>/</em>4) = 1

entropy(Income=med) = −4<em>/</em>6 ∗ log(4<em>/</em>6) − 2<em>/</em>6 ∗ log(2<em>/</em>6) = 0<em>.</em>91829 entropy(Income=low) = −3<em>/</em>4 ∗ log(3<em>/</em>4) − 1<em>/</em>4 ∗ log(1<em>/</em>4) = 0<em>.</em>81127

Gain(D,Income) = entropy(D)− (4<em>/</em>14 ∗ 1 + 6<em>/</em>14 ∗ 0<em>.</em>91929 + 4<em>/</em>14 ∗ 0<em>.</em>81128) = 1

<strong>Sample decision tree</strong>

Here is the example decision tree by using the sample.data, sample.label:

<strong>Note: The tree building function and prune function should be implemented in a recursive manner, otherwise your solution score will be penalized by 80%.</strong>

<h1>Part 1 Decision Trees</h1>

<strong>Note</strong>: You need to finish only your code as a separate file (ID3.py) for this section. Your algorithm should finish training and testing within 5 Minutes. We will verify your training and testing modules, and you should not include your pre-trained model.

<ol>

 <li>Implement a binary decision tree with no pruning using the ID3 (Iterative Dichotomiser 3) algorithm<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>.</li>

</ol>

Format of calling the function and accuracy you will get after training:

$ python ID3.py ./path/to/train-file ./path/to/test-file vanilla 80

Train set accuracy: 0.9123

Test set accuracy: 0.8123

The fourth argument (80) is the training set percentage. The above example command means we use only the first 80% of the training data from train-file. (We use all of the test data from test-file.)

<ol start="2">

 <li>Implement a binary decision tree with a given maximum depth. Format of calling the function and accuracy you will get after training:</li>

</ol>

$ python ID3.py ./path/to/train-file ./path/to/test-file depth 50 40 14

Train set accuracy: 0.9123

Validation set accuracy: 0.8523

Test set accuracy: 0.8123

The fourth argument (50) is the training set percentage and the fifth argument (40) is the validation set percentage. The sixth argument (14) is the value of maximum depth.

So, for example, the above command would get a training set from the first 50% of train-file and get a validation set from the last 40% of train-file (the two numbers need not add up to 100% because we sometimes use less training data). Finally, we set the maximum depth of the decision tree as 14. As before, we get the full test set from test-file.

<strong>Note: </strong>you have to print the validation set accuracy for this case.

<ol start="3">

 <li>Implement a binary decision tree with a given minimum sample split size. Format of calling the function and accuracy you will get after training:</li>

</ol>

$ python ID3.py ./path/to/train-file ./path/to/test-file min_split 50 40 2

Train set accuracy: 0.9123

Validation set accuracy: 0.8523

Test set accuracy: 0.8123

The sixth argument (2) is the value of minimum samples to split on.

The above example command would get a training set from the first 50% of train-file and get a validation set from the last 40% of train-file (the two numbers need not add up to 100% because we sometimes use less training data). Finally, we set the minimum samples to split on of the decision tree as 2. As before, we get the full test set from test-file.

<strong>Note: </strong>you have to print the validation set accuracy for this case.

<ol start="4">

 <li> Implement a binary decision tree with post-pruning using reduced error pruning.</li>

</ol>

Format of calling the function and accuracy you will get after training:

$ python ID3.py ./path/to/train-file ./path/to/test-file prune 50 40

Train set accuracy: 0.9123

Test set accuracy: 0.8123

The fourth argument (50) is the training set percentage and the fifth argument (40) is the validation set percentage.

So, for example, the above command would get a training set from the first 50% of train-file and get a validation set from the last 40% of train-file. As before, we get the full test set from test-file.

<h1>Part 2 Analysis</h1>

<strong>Note: </strong>You need to submit only your answers in the PDF for this section.

For the following questions, use titanic-train.data, titanic-train.label as the training file and titanic-test.data, titanic-test.label as the test file. You should use numpy, matplotlib, seaborn for plotting the graph, and include your code scripts. Make sure your xaxis, yaxis, title has proper name.

<ol>

 <li> For the full decision tree (vanilla), measure the impact of training set size on the accuracy and size of the tree.</li>

</ol>

Consider training set percentages {40%<em>,</em>60%<em>,</em>80%<em>,</em>100%}.

Plot a graph of test set accuracy and training set accuracy against training set percentage on the same plot.

Plot another graph of number of nodes vs training set percentage.

<ol start="2">

 <li> Repeat the same analysis for the static-depth case (depth).</li>

</ol>

Again, consider values of training set percentage: {40%<em>,</em>50%<em>,</em>60%<em>,</em>70%<em>,</em>80%}. The validation set percentage will remain 20% for all the cases.

Consider values of maximum depth from {5<em>,</em>10<em>,</em>15<em>,</em>20} and pick the best value using the validation set accuracy. The accuracies you report will be the ones for this value of maximum depth. So, for example, if the best value of maximum depth for training set 40% is 5, you will report accuracies for 40% using 5; if for 50% it is 10, you will report accuracies for 50% using 10.

Plot a graph of test set accuracy and training set accuracy against training set percentage on the same plot. Plot another graph of number of nodes vs training set percentage.

Finally, plot the optimal choice of depth against the training set percentage.

<ol start="3">

 <li> Repeat the above analysis for the pruning case (prune).</li>

</ol>

Again, consider values of training set percentage: {40%<em>,</em>50%<em>,</em>60%<em>,</em>70%<em>,</em>80%}. The validation set percentage will remain 20% for all the cases. You will use the validation set when deciding to prune.

Plot a graph of test set accuracy and training set accuracy against training set percentage on the same plot.

Plot another graph of number of nodes vs training set percentage.

<ol start="4">

 <li> Why don’t we prune directly on the test set? Why do we use a separate validation set?</li>

 <li>How would you convert your decision tree (in the depth and prune cases) from a classification model to a ranking model?</li>

</ol>

That is, how would you output a ranking over the possible class labels instead of a single class label?

<a href="#_ftnref1" name="_ftn1">[1]</a> https://en.wikipedia.org/wiki/ID3_algorithm