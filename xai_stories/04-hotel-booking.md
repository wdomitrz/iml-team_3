# Story hotel booking: eXplainable predictions of booking cancellation and guests coming back {#story-hotel-booking}

*Authors: Domitrz Witalis (MIM), Seweryn Karolina (MiNI)*

*Mentors: Jakub Tyrek (Data Scientist), Aleksander Pernach (Consultant)*

## Introduction 

The dataset is downloaded from the Kaggle competition website https://www.kaggle.com/jessemostipak/hotel-booking-demand.
This dataset contains booking information for a city hotel and a resort hotel in Portugal, and includes information such as when the booking was made, length of stay, the number of adults, children, babies, the number of available parking spaces, chosen meals, price etc. There are 32 features and 119 390 observations. The booking website has information about these reservation characteristics and building models can help this company in better offer management. The most important information could be 

* the prediction of booking cancellation, 
* the prediction if client comes back to the hotel,
* the prediction whether client orders additional services (eg. meals),
* customer segmentation.

In this project, we have decided to focus on two first issues.

### Hyperparameter optimization

### Imbalanced dataset

[Put a description of the problem here. indicate the data source. Describe why this problem is important. Indicate the most important literature on the problem.]

## Model 

### Model 1. Booking cancellation

The aim of this model is to predict whether guest cancels reservation and explanation of the reasons. The chosen model is XGBoost with RFE (Recursive Feature Elimination). Bayesian optimisation with TPE tuner has been applied in order to improve model performance. Neural Network Intelligence (NNI) package has been chosen for this task, because it provides user-friendly GUI with summary of experiments.

List of optimized hyperparameters and search space:

1. **max_depth** - the maximum depth of tree.
2. **n_esimators** - the number of trees.
3. **learning_rate** - boosting learning rate
4. **colsample_bytree** - subsample ratio of columns when constructing each tree.


![image](images/hyperparameter_optimization.png)
*Figure details paths of hyperparameters values chosen by algorithm. On the right you can see metric (AUC) of model with those parameters.*

![image](images/ho2.png)
*Figure presents AUC for each experiment. It shows a clear trend in model performence so algorithm is choosing better and better hyperparameters.*


|   |   |   |
|---|---|---|
|AUC train |   |   |
|AUC test   |   |   |


![image](images/roc_curve.png)

### Model 2. Repeated guests

Place a description of the model(s) here. Focus on key information on the design and quality of the model(s) developed.

## Explanations

### Model 1. Booking cancellation

#### dataset

![image](images/feature_importance.png)

![image](images/shap_summary_plot.png)
*.*

Figure above shows SHAP values. There are some interesting findings which are intuitive:

* Clients who canceled some reservations in  the past are more likely to cancel another reservation.
* People who buy refundable option cancel reservations more often than others.
* A lot of days between reservation time and arrival time increases probability of cancelling booking. 
* People who travel with children are more likely to cancel booking.

There are also less intuitive findings:

* People without any special requests cancel reservetion more often than others.
* If trip starts at the end of the week there is higher probability that customers change their minds.
* The bigger number of adults, the highest probability of cancellation.

#### instance

1. The lowest prediction of cancellation probability
![image](images/shap_min.png)
![image](images/min_break_down.png)
2. The highest prediction of cancellation probability
![image](images/shap_max.png)
![image](images/max_break_down.png)
   
### Model 2. Repeated guests


Here, show how XAI techniques can be used to solve the problem.
Will dataset specific or instance specific techniques help more?

Will XAI be useful before (pre), during (in) or after (post) modeling?

What is interesting to learn from the XAI analysis?


## Summary and conclusions 

Here add the most important conclusions related to the XAI analysis.
What did you learn? 
Where were the biggest difficulties?
What else did you recommend?

