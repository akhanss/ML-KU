import numpy as np

def svm_loss(W, X, y, classes, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # compute the loss and the gradient
  # num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # Implementation of a SVM loss, storing the result in loss.                 #
  #############################################################################
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y-1]
  margin = np.transpose(scores) - correct_class_scores + 1 # delta = 1
  margin[y-1, np.arange(num_train)] = 0 

  # values greater than zeros in margin - calculating max(0, margin)
  gt_zero = np.maximum(np.zeros((margin.shape)), margin)

  loss = np.sum(gt_zero)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.  
  loss /= num_train
  # And regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # Implementation the gradient for the SVM loss, storing the result in dW.   #
  #                                                                           #
  #############################################################################

  # classifiers having loss > 0
  gt_zero[gt_zero > 0] = 1

  # Calculating indexes for the necessary subtractions
  images_sum = np.sum(gt_zero, axis = 0)

  # Subtracting the derivative
  gt_zero[y-1, range(num_train)] = -images_sum[range(num_train)]

  # updating the gradients
  dW = np.transpose(gt_zero.dot(X))

  # Normalizing the gradient
  dW /= num_train

  # Adding regularization to the gradieant.
  dW += reg * W

  return loss, dW
