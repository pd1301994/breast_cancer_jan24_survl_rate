# Predicting Survival in Breast Cancer Patients: Data Science and Machine Learning Approach
## Description

This repository contains all the necessary data and files clean data a given dataset and to make predictions based on this data on the survival rate of patients with Breast Cancer.
The main goal of this project is compare different machine learning and deep learning algorithms to determine which one will give better results regarding the prediction of the survival status of breast cancer patients. 

## File Organization
This repository contains the following folders: 
- Data
    - dummies: This file contains the data where the following preprocessing was done - label encoder, imputation, reverse label encoder, get_dummies 
    - label_encoder: This file contains the data that was preprocessed with label encoder followed by imputation
    - model_get_dummies_data: Result of running the get_dummies models
    - model_label_data: Result of running the label encoded models
    - no_processed_data: This file contains the raw data
    - processed_data: This file contains the processed data
- Notebooks
    - Modeling.ipynb
    - Preprocessing.ipynb
- stremlit_app
    - assets 
    - tabs
        - Modeling.py
        - Preprocessing.py
        - intro.py
      - app.py
      - config.py
      - member.py
      - requirements.txt
      - style.css
- README.md
- requirements.txt

## Requirements
All the libraries being used in our project can be found in the requirements.txt file to install it use this command in your terminal: 
```
pip install -r requirements.txt
```
# Explanation
For clarification, we will divide this part into two diferent sections: 
## 1. Data preprocessing and modeling
In this section, we will focus on the Data and Notebooks folders. 
The *Data* folder contains our generated data as explained above. 
The *Notebooks* folder contains two notebooks: *Preprocessing.ipynb* and *Modeling.ipynb*.
- **Preprocessing.ipynb**: This notebook contains data cleaning and preprocessing. The input for this notebook will be a csv file containing the raw data, the name of the file is: 'final_Data_merged.csv'.
For importing the data, two paths are specified. One can be accessed from the google drive, for this path you will have to create a folder in your google drive under the name 'Datasets' and import the 'final_Data_merged.csv' raw file into. The second path can be accessed automatically if you download the repository and work on your local machine, the data will be accessed through this path data<no_processed_data<final_Data_merged.csv.
After running all the codes in this notebook, you will have created a total of 8 new processed dataframes:
  - processed_data: This dataframes will produce 2 distinct datasets based on two different thresholds selected for handling NaN values in the raw.
- label_encoder: From the 2 dataframes mentioned above, two separate files will be generated based on the label encoder encoding.
-  dummies: Two separate files will be generated from the 2 dataframes in processed_data, using the get_dummies encoding. Afterwards, the data will be split into data and target for each of the datasets mentioned above, resulting in a total of 4 dataframes. 

  The dataframes available on the repository were created by running the codes in the Preprocessing.ipynb notebook in March 2024. If you would like to create the dataframes yourself based on new parameters follow the instructions under *how to use*.

- **Modeling.ipynb**: This notebook contains the Machine Learning and Deep learning algorithms. The data used in this notebook are the final data created in the Preprocessing.ipynb notebook. For section 5 a user imput will be required. It is asked what oversampling method do you want to use an lists the ones available, the spelling here is important. There  are some cells which take a lot of time to run, be patiente.especially when use gridsearchCV.

 ## 2. Project visualization
For the project visualization, we will use the streamlit_app. To execute this portion of the code, you should first navigate to the streamlit_app folder.
```
cd .\streamlit_app\
```
and then run the following code: 
```
streamlit run app.py
```
The project will run http://localhost:8501. The default browser you are using should open automatically. If it doesn't, you can manually copy and paste the address into the search bar of your web browser.

## How to use the system
1. Download the repository to your local machine
2. The data preprocessed is already saved on the folders, however if you wish to re-write this dataframes they will be overwritten if you run the Preprocessing.ipynb file. **It is very important that you don´t delete the folders as the paths are specified for this specific folders.**
3. For the .zip 'Data' folder, it is also important that you do not duplicate the path. In some OS, when you extract a file it does /data/data/you_files as default. You must avoid this duplication in order that the program works (final path should be (data/your_files)
4. In case you want to use jupyter or colab for the .ipynb file you should create a folder in your drive called 'Datasets'. If you run the Preprocessing.ipynb file all the newly generated dataframes will be saved there. If you do not require the datasets or wish to avoid creating them, you can always download them from this repository and upload them as needed.
