#!/bin/bash

mkdir bin

make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the featurre files may be built. Note that don't necessarily have to
# use Java for this step.

# cat ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold3  ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > output1
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold3  ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > output2
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2  ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > output3
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2  ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold5 > output4
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2  ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold4 > output5

# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../decision-trees/output1 ./../badges.train1.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../decision-trees/output2 ./../badges.train2.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../decision-trees/output3 ./../badges.train3.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../decision-trees/output4 ./../badges.train4.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../decision-trees/output5 ./../badges.train5.arff


# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.all ./../badges.example.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1 ./../badges.fold1.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2 ./../badges.fold2.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold3 ./../badges.fold3.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold4 ./../badges.fold4.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold5 ./../badges.fold5.arff

# cat ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > train1
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > train2
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold4 ../badges/badges.modified.data.fold5 > train3
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold5 > train4
# cat ../badges/badges.modified.data.fold1 ../badges/badges.modified.data.fold2 ../badges/badges.modified.data.fold3 ../badges/badges.modified.data.fold4 > train5

# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator train1 ./../badges.train1.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator train2 ./../badges.train2.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator train3 ./../badges.train3.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator train4 ./../badges.train4.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator train5 ./../badges.train5.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1 ./../badges.test1.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2 ./../badges.test2.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold3 ./../badges.test3.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold4 ./../badges.test4.arff
# java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold5 ./../badges.test5.arff


# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 
# java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.example.arff

# java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train1.arff ./../badges.fold1.arff
# java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train2.arff ./../badges.fold2.arff
# java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train3.arff ./../badges.fold3.arff
# java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train4.arff ./../badges.fold4.arff
# java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train5.arff ./../badges.fold5.arff

java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train1.arff ./../badges.train2.arff ./../badges.train3.arff ./../badges.train4.arff ./../badges.train5.arff ./../badges.test1.arff ./../badges.test2.arff ./../badges.test3.arff ./../badges.test4.arff ./../badges.test5.arff

