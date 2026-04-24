# Physics-Informed-DoC-Predictor

A collection of ML models that use time, temperature, sensor data, and DoC (CSV files) to learn the relationship between these values to approximate DoC for various resin samples. They are all purely artificially data driven.

MLP Predictor is non physics informed, it is the first iteration in the development of the LSTM model. 

LSTM model is physics informed - uses Friedmann Isoconversion Model to vaildate cure kinetics and isolate ideal S curves from noisy RFID data. 
