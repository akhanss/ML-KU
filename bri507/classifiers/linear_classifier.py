import numpy as np
from classifiers.linear_svm import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, classes, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=16, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      #                                                                       #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      #########################################################################

      # Get a sample from X by first generating random indices
      X_batch_indices = np.random.choice(num_train, size=batch_size, replace=True)
      # Then index X to get the desired images
      X_batch = X[X_batch_indices, :]  # Should be (dim, batch size)

      # And then get the corresponding labels
      y_batch = y[X_batch_indices]  # should be (batch size, )  

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, classes, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      #                                                                       #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      # gradient descent.
      update = -1 * learning_rate * grad
      new_update = self.W + update
      self.W = new_update

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])
    ###########################################################################
    #                                                                         #
    # Implementation this method. Store the predicted labels in y_pred.       #
    ###########################################################################
    # First, we need to calculate the lables for every training example
    results = np.dot(X, self.W)  # This is done by multiplying the weights with the image features
    best_labels = results.argmax(axis=1)

    # Now, the label selected should be the classifier with the highest score
    y_pred = best_labels

    return y_pred
  
  def loss(self, X_batch, y_batch, classes, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass
    


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, classes, reg):
    return svm_loss(self.W, X_batch, y_batch, classes, reg)

