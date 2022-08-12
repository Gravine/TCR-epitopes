# TCR-epitopes
Use LSTM to predict the binding between TCR and specific epitopes

Run in command prompt:

# model training command 
python TCR-epitopes.py train --model_file=model_file_name

where the model_file_name is the filename of the model to be saved. 
For example:
python TCR-epitopes.py train --model_file=model_0811

# model prediction command
python TCR-epitopes.py predict --model_file=model_file_name --predict_output=predict_output_file_name

where the model_file_name is the filename of the model to be loaded, and the predict_output_file_name if the filename of the csv file of the prediction output to be saved. 
For example:
python TCR-epitopes.py predict --model_file=model_0811 --predict_output=predict_output_0811
