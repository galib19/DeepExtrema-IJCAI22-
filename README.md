# DeepExtrema-IJCAI22

## Data
All data files can be found in the **Data** folder.

## Code
All codes are available in the **Code** folder.

### Baseline Models
- `baseline_models.py`: Contains all baseline models except EVL.

### EVL Baseline
- `extreme_time(EVL).py`: EVL baseline derived from [Forecast package](https://github.com/tymefighter/Forecast).
- `extreme_time_2(EVL).py`: Another version of EVL baseline derived from [Forecast package](https://github.com/tymefighter/Forecast).

### General Utilities
- `general_utilities.py`: Contains general functions for data processing, GEV likelihood estimation, standardization, etc.

### Main Programs
- `main_Ausgrid.py`: Main program for Ausgrid data.
- `main_Hurricanes.py`: Main program for hurricanes data.
- `main_Weather.py`: Main program for weather data.

### DeepExtrema Model
- `model.py`: Contains the DeepExtrema model.

### Training Utilities
- `train_model_gev.py`: Contains the training function.
- `train_utilities.py`: Contains functions for training, including plotting and metrics calculations.
