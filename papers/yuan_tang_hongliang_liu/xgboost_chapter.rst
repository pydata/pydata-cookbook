:author: Yuan Tang
:email: terrytangyuan@gmail.com
:institution:
:corresponding:

:author: Hongliang Liu
:email: phunter.lau@gmail.com
:institution:
:equal-contributor:

------------------------------------------------

XGBoost: A Portable Distributed Gradient Boosting Library

------------------------------------------------

.. class:: abstract

   XGBoost is a distributed machine learning library under gradient boosting
   framework. It is designed and optimized for high efficiency,
   flexibility and portability. It provides a parallel tree boosting
   implementation (also known as GBDT or GBM) for solving many data science
   problems in a fast and accurate way. Beyond single machine parallelization,
   XGBoost also runs on distributed environments such as Hadoop YARN, Spark,
   and Flink, for distributed model training. It's also optimized on GPUs to
   accelerate the model training. More than half of the winning solutions in
   machine learning challenges hosted at Kaggle and a wide range
   of use cases across industries adopt XGBoost. In this chapter,
   we will briefly walk through main functionalities of XGBoost,
   explain the meanings of selected important parameters, give general guidance
   of parameter tuning, and introduce techniques for running large tasks in
   distributed environments and on GPUs.

.. class:: keywords

   Gradient Boosting, Machine Learning, Predictive Analytics, Data Science


.. include:: papers/yuan_tang_hongliang_liu/intro.txt

.. include:: papers/yuan_tang_hongliang_liu/main_functionalities.txt

.. include:: papers/yuan_tang_hongliang_liu/parameter_explained.txt

.. include:: papers/yuan_tang_hongliang_liu/distributed_and_gpus.txt

.. include:: papers/yuan_tang_hongliang_liu/references.txt
