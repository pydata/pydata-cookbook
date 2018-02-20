:author: Christopher Fonnesbeck
:email: chris.fonnesbeck@vanderbilt.edu
:institution: Vanderbilt University Medical Center

:author: Peadar Coyle
:email: peadarcoyle@googlemail.com

:author: Thomas Wiecki
:email: thomas.wiecki@gmail.com

---------------------------------------
Bayesian Regression Analysis with PyMC3
---------------------------------------

.. class:: abstract

Regression analysis is a fundamental statistical method for relating two sets of variables to one another in order to provide inference or make predictions. It provides estimates of how outcomes of interest vary as the value of specific predictor variables change, along with estimates of uncertainty about this relationship. Adopting the Bayesian approach recasts the regression problem in probabilistic terms, with all model inputs and outputs expressed as probability distributions. This confers a great deal of flexibility to the modeling task, and makes model outputs easier to interpret. However, there has historically been a great deal of computational complexity associated with building and fitting Bayesian models. PyMC3 is a Python package for Bayesian statistical analysis that makes regression analysis easy by providing a high-level language for specifying models and a suite of powerful fitting algorithms for generating output. We present an example-driven guide to Bayesian regression modeling in PyMC3, starting with simple one-line models for linear regression, then expand the simple case to address realistic problems encountered by scientists, including the estimation of non-linear relationships, discrete and binary outcomes, and hierarchical variable relationships. We'll also introduce some of the methods for checking models, since we believe in the old saying that 'all models are wrong, some are useful' it is important to evaluate models before putting them to use.
