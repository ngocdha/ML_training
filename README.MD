# My first Machine Learning training course

This repo contains my solutions to the exercises of the ML course by Andrew Ng on Coursera.

## Personal takeaway

### Exercise 1

- For **gradient descent**: check, double-check, triple-check if the code is really updating theta *simultaneously*. Do NOT partially update theta.

- Always try to *generalize* the code. This means that the code should work for different data sets with different number of features and examples.

- For using theta to make **prediction**: check if data has been normalized or not *prior* to estimating.

### Exercise 4

- The error term (sigma) of a bias unit does **NOT** participate in backpropagation.

- Useful to put the dimensions of the matrices next to the code, in comments. Wow it took me 4 exercises to realize I should develop this habit.

- Make sure the test environment has the correct variables to use for debugging.