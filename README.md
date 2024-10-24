# NYC_Taxi_Project

This is a repository for the coursework. It performs data analysis on NYC taxi data in a reproducible way. 

## Structure

```
├── data                    # data folder
│── models                  # folder to save models
├── scripts                
│   ├── data_utils.py          # functions for data loading, cleaning, and splitting 
│   ├── model_utils.py         # classes for the dataset and the model
│── train_model.py          # main function
│── requirements.txt        # dependencies
└── ...
```

## Dependencies
To install all the dependencies, check requirements.txt.


## Data Analyze

To reproduce the results, run the following command. You can change the path to save the model as a customized one.
```
python tran_model.py --save_path='./model/nyc_taxi_mlp.pt'
```
