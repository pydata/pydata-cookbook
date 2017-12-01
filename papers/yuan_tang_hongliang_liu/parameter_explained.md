# Parameter explained: train `xgboost` for the best prediction

There are 3 types of hyper parameters in `xgboost`: 

* General Parameters for general boost function, commonly tree or linear model.
* Booster Parameters for individual booster at each step which determines the model predictive power.
* Learning Task Parameters for the optimizing prediction performance in the learning senarios.

All these parameters are documented in the `xgboost` [document](http://xgboost.readthedocs.io/en/latest/parameter.html).

## General parameters

`booster`, `silent` and `nthread` are important general parameters, while `xgboost` automatically defines `num_pbuffer` and `num_feature`.

* `booster` [default=gbtree]: select boosters in `gbtree`, `gblinear` or `dart` where `gbtree` and `dart` use tree based model and `gblinear` uses linear function.
* `silent` [default=0]: `0` means printing running messages while `1` means silent mode. For better understanding the model, one might want to set it to `0`.
* `nthread` [default to maximum number of threads available in the system if not set]: number of parallel threads used to run xgboost. If ones wants to run on all cores/threads, this value can be left to default. If the system's CPU support hyper-threading, the default number is the number of hyper-threads instead of real cores.

## Booster parameters

We explain most tree model booster parameters in this section since the tree model is most commonly used, and it can outperform the linear booster for most cases. For `dart` and `tweetie regression` parameters, please refer to the `xgboost` [document](http://xgboost.readthedocs.io/en/latest/parameter.html).

*Popular booster parameters*

* `eta` [default=0.3]: the learning rate in GBM model. It is named as `learning_rate` if use the `scikit-learn` interface `XGBClassifier`.
	* the weights on each step is shrinking at each step and the model becomes more stable.
	* Recommened values: 0.01-0.2
	* The smaller `eta`, the slower shrinking, and the more boosting  rounds are needed for reaching the optimal state, where the number of boosting rounds can be defined when initializing `xgboost` by the `n_rounds` parameter.
	* The larger `eta`, the faster shrinking, however, it may risk for not reaching the final optimal state.
* `min_child_weight` [default=1]
	* Defines the minimum sum of weights of all observations required in a child tree.
	* This refers to min “sum of weights” of observations instead of `min_child_leaf ` in GBM.
	* It controls over-fitting: Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. However, a too high value can lead to under-fitting. This parameter should be tuned using cross-validation.

* `max_depth` [default=6]
	* The maximum depth of a tree, same as GBM.
	* Recommended values: 4-8
	* It controls over-fitting: taller trees allows model to learn relations very specific to particular samples, while shorter trees don't have enough depth for learning enough from particular samples. This parameter should be tuned using cross-validation.

* `max_leaf_nodes`: maximum number of terminal nodes or leaves in a tree.
	* It can be used for replacing `max_depth` while a depth of ‘n’ would produce a maximum of `2^n` leaves for binary tree. If this is defined, xgboost will ignore `max_depth` parameter.

* `gamma` [default=0]
	* It specifies the minimum loss reduction required for splitting a node when a node is split only when the resulting split gives a positive reduction in the loss function. 
	* There are no default recommended values, since the values depend on the loss function and should be tuned using cross-validation.

* `subsample` [default=1]
	* it defines the fraction of observations to be randomly samples for each tree. It can mitigate over-fitting for some cases since smallers values gives more conservative decision but it might under-fit if the value is too small.
	* Recommended values: 0.5-1, the smaller value, the more risk of underfitting.
* `colsample_bytree` [default=1]
	* It defines the fraction of columns to be randomly samples for each tree. It may mitigate over-fitting for some cases, since it introduces randomness from the data for preventing the tree learning on too specific samples.
	* Recommended values: 0.5-1, the smaller value, the more risk of underfitting.
* `scale_pos_weight` [default=1]
	* It defines the scale factor of positive vs negative samples in the training dataset. If  the training dataset is highly imbalanced, it can helps faster learning and converging.
* `alpha` [default=0]: it is also named as `reg_alpha` if use the `scikit-learn` interface `XGBClassifier`.
	* It defines L1 regularization term on weight (similar to Lasso regression)
	* If the features are in a very high dimension and has much sparsity, good alpha values can lead to faster algorithm.

*Rarely tuned booster parameters*

`xgboost` also has some parameters which are set to default for most cases.

* `max_delta_step` [default=0]
	* It sets the limit to each tree's weight estimation, if it is default as 0, there is no constraint, otherwise the algorithm becomes more consservative when update steps and has more risk of being underfitting. This parameter might be effective when train a highly imbalanced dataset. 
* `colsample_bylevel` [default=1]
	* It defines the subsample ratio of columns for each split in each level. However, `subsample` and `colsample_bytree` are usually used.
* `lambda` [default=1]: it is also named as `reg_lambda` if use the `scikit-learn` interface `XGBClassifier`.
	* It defines `L2` regularization term on weights (similar to Ridge regression)
	* This used to handle the regularization part of XGBoost. It might mitigate overfitting for some cases.
	
## Learning task parameters

Learning task parameters define the optimization objective metric to be calculated at each step, a.k.a the loss function. For different use cases, different loss functions should be chosen.

* `objective` [default=reg:linear]: the loss function to be minimized. 
	* `binary:logistic` : logistic regression for binary classification, returns predicted probability
	* `multi:softmax` : multiclass classification using the softmax objective, returns predicted class (binary results, not probabilities). `num_class` parameter is needed for defining the number of classes
	* `multi:softprob` : same as softmax, but returns predicted probability of each data point belonging to each class.
* `eval_metric` [ default according to objective ]: The evaluation metric for validation. The default values are `rmse` for regression and `error` for classification.
	* `rmse`: root mean square error
	* `mae`: mean absolute error
	* `logloss`: negative log-likelihood
	* `error`: Binary classification error rate (0.5 threshold)
	* `merror`: Multiclass classification error rate
	* `mlogloss`: Multiclass logloss
	* `auc`: Area under the curve
* `seed` [default=0]: the random number seed
	* maintaining the seeds can generate reproducible results.
	* Using different seeds can introduce model variance. By keeping multiple seeds and their model output, one can ensemble multiple models from multiple seeds and reduce the model variance.
