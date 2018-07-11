# Programming Exercise 1: Linear Regression

## Introduction

In this exercise, you will implement linear regression and get to see it work on data. Before starting on this programming exercise, we strongly recom- mend watching the video lectures and completing the review questions for the associated topics.
To get started with the exercise, you will need to download the starter code and unzip its contents to the directory where you wish to complete the exercise. If needed, use the cd command in Octave/MATLAB to change to this directory before starting this exercise.
You can also find instructions for installing Octave/MATLAB in the “En- vironment Setup Instructions” of the course website.

## Files included in this exercise

ex1.m - Octave/MATLAB script that steps you through the exercise ex1 multi.m - Octave/MATLAB script for the later parts of the exercise
ex1data1.txt - Dataset for linear regression with one variable
ex1data2.txt - Dataset for linear regression with multiple variables
submit.m - Submission script that sends your solutions to our servers
[⋆] warmUpExercise.m - Simple example function in Octave/MATLAB
[⋆] plotData.m - Function to display the dataset
[⋆] computeCost.m - Function to compute the cost of linear regression
[⋆] gradientDescent.m - Function to run gradient descent
[†] computeCostMulti.m - Cost function for multiple variables
[†] gradientDescentMulti.m - Gradient descent for multiple variables
[†] featureNormalize.m - Function to normalize features
[†] normalEqn.m - Function to compute the normal equations

⋆ indicates files you will need to complete
† indicates optional exercises

Throughout the exercise, you will be using the scripts ex1.m and ex1 multi.m. These scripts set up the dataset for the problems and make calls to functions that you will write. You do not need to modify either of them. You are only required to modify functions in other files, by following the instructions in this assignment.
For this programming exercise, you are only required to complete the first part of the exercise to implement linear regression with one variable. The second part of the exercise, which is optional, covers linear regression with multiple variables.