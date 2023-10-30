# FB Prophet

FB Prophet is time series forecast implemented in R & Python
Prophet follows sklearn model API by creation of instance of Prophet class and then call its fit and predict methods.
Input Format: A dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.
## Hyper Parameters of Fb Prophet:
- n_changepoints, changepoint_range , changepoint_prior_scale :
- Automatic detection of changepoints: Prophet detects changepoints by first specifying a large number of potential changepoints at which the rate is allowed to change.
- n_changepoints: The number of potential changepoints can be set using this argument, but it’s better tuned by adjusting regularization. Default value is 25 which are uniformly placed in the first 80% of the time series. 
- changepoint_range: This above default range of 80% can be changed using this argument. 
- Specifying the locations of the changepoints: Instead of automatic changepoint detection, Prophet allows to specify the locations of potential changepoints with the changepoints argument.
- changepoint_prior_scale: It adjust the strength of the sparse prior if trend changes are being overfit or underfit using this argument. Default value is 0.05. Increasing it will make the trend more flexible.

## Cesium Time Series Package in Python

Cesium is an end-to-end ML platform for time-series, from calculation of features to model-building to predictions. 
Cesium has two main components - Python library & web API that allows interactive exploration of ML pipelines. 
It take control over the workflow in Python terminal or Jupyter notebook or upload your time-series files then select your machine learning model, and watch Cesium do feature extraction and evaluation right in your browser with the web application.
The overall workflow consists of three steps to apply Cesium:
“featurize” the time series by selecting some set of mathematical functions to apply to each
Build some Regressor/Classification models which use these features to Predict for target variable or distinguish between classes in case of classification model
Validate models by generating predictions for some unseen holdout set and comparing them to the true values.
## Featurization
- Load the time series data and club n-elements of each variable in array format as explained in the code file.
- Then generate features for each time series using the cesium.featurize module. 
- The featurize module includes many built-in choices of features which can be applied for any type of time series
- The output of featurize_time_series is an xarray.Dataset which contains all the feature information needed to train a ML model

## Correlation vs AutoCorrelation

- Correlation is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship. In terms of the strength of relationship, the value of the correlation coefficient varies between +1 and -1.
- A value of ± 1 indicates a perfect degree of association between the two variables. As the correlation coefficient value goes towards 0, the relationship between the two variables will be weaker.
- Auto-correlation refers to the case when your errors are correlated with each other. In layman terms, if the current observation of your dependent variable is correlated with your past observations, you end up in the trap of auto-correlation. 

## Time Series Basics

- Chronological Data
- Cannot be shuffled
- Each row indicate specific time record
- Train – Test split happens chronologically
- Data is analyzed univariately (for given use case)
- Nature of the data represents if it can be predicted or not

## Code Description

    File Name : Engine.py
    File Description : Main class for FBProphet Framework & Cesium Framework
    
    File Name : Featurizing.py
    File Description : Class for Featurizing Framework
    
    File Name : Prophet.py
    File Description : Class for training fbprophet Model
    
    File Name : MLP.py
    File Description : Class for training MLP Model
    
    

### IPython Google Colab

Follow the instructions in the notebook `cesium.ipynb` & `FbProphet.ipynb`



## Execution Instructions ##

Code Tested on Python 3.8.10

To create a virtual environment and install packages from a requirements file in Python 3.8.10, you can follow these execution steps:

Install Python 3.8.10: Download and install Python 3.8.10 from the official Python website (https://www.python.org/downloads/) based on your operating system.

Open a terminal or command prompt: Launch a terminal or command prompt window on your system.

Create a virtual environment: Enter the following command to create a new virtual environment (replace myenv with the desired name for your virtual environment):

```
python3.8 -m venv myenv
```

This command will create a new directory named myenv (or the name you specified) containing the virtual environment files.

Activate the virtual environment: Depending on your operating system, execute the appropriate command to activate the virtual environment:

Windows (Command Prompt):

```
myenv\Scripts\activate.bat
```


Windows (PowerShell):

```
myenv\Scripts\Activate.ps1
```

Unix/Linux:
```
source myenv/bin/activate
```
After activation, you should see the name of your virtual environment in the command prompt or terminal.

Install packages from a requirements file: Assuming you have a requirements.txt file containing the desired packages and their versions, use the following command to install them into the virtual environment:

Go to Src directory with the command: cd Src
then
```
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages and their dependencies into your virtual environment.

Verify installation: You can check if the packages were installed correctly by executing the following command:
```
pip list
```
This will display a list of installed packages within the virtual environment.

Deactivate the virtual environment: When you are done working within the virtual environment, you can deactivate it using the following command:

```
deactivate
```
After deactivation, you'll return to your system's default Python environment.

