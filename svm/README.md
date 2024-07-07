# SVM

## SVM Primal

We use the [Pegasos algorithm](https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf) to solve the SVM Primal Problem.

## SVM Dual

To solve the dual problem, we use the popular optimization package `cvxopt`. The dual problem is rewritten as a quadratic programming problem.

## News data

The `news.mat` dataset is a subset of the [20 news groups dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset).

The optimal weight and bias terms are computed on the `news.mat` data in `pegasos.py` for the primal problem and `svm qp.py` for the dual problem.

The results are fairly similar in both methods. In the Pegasos method, the best test accuracy of 0.729 was obtained with the use of $\lambda = 0.1$. In the dual method, the best test accuracy of 0.713, which is close to one in the Pegasos case, is obtained for $C = 0.01$.

We can also notice that, while for Pegasos, increasing lambda was decreasing the training accuracy, the opposite is the case for this method. This describes the inverse relationship between C and lambda.

## Data

https://drive.google.com/file/d/10hTrfjhbDsXd6Xja9lQ7_c7p20PZKLkT/view?usp=sharing

## Sample Usage

Run the scripts with specific settings for the arguments.

`python pegasos.py --epochs 1000 --lamda 1.0 --batch_size 128`

`python svm_qp.py --C 1.0`

If you run `python pegasos.py` or `python svm_qp.py`, the arg_parse arguments
will be set to their default values.
