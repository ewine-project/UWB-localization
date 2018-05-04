# UWB Data Set

This folder holds the data collected during several measurement campaigns using [SNPN-UWB](http://www.log-a-tec.eu/mtc.html) board with DecaWave [DWM1000](https://www.decawave.com/products/dwm1000-module) UWB pulse radio module.

## Directory Structure

	+ localization
		|___ code
		|___ dataset1
		|___ dataset2
	+ NLOSClassification
		|___ code
		|___ dataset
	
### localization
Contains data sets created for localization evaluation. Data points were collected at predefined points in several indoor environments with recorded positions, actual distances between nodes and all radio channel performance data available for every measurement point.

### NLOSClassification
Contains data set collected in several indoor environments without information of actual positions of nodes. Each data point contains information if the link between the nodes is a LoS link or an NLoS link and all channel performance data available at UWB radio.

## Citation
If you are using our data set in your research, citation of the following paper would be greatly appreciated.

Plain text:

	K. Bregar and M. Mohorčič, "Improving Indoor Localization Using Convolutional Neural Networks on Computationally Restricted Devices," in IEEE Access, vol. 6, pp. 17429-17441, 2018.
	doi: 10.1109/ACCESS.2018.2817800
	keywords: {Computational modeling;Convolutional neural networks;Distance measurement;Estimation;Heuristic algorithms;Performance evaluation;Prediction algorithms;Channel impulse response;convolutional neural network;deep learning;indoor localization;non-line-of-sight;ranging error mitigation;ultra-wide band},
	URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8320781&isnumber=8274985
	

BibTeX: 

	@ARTICLE{8320781,
		author={K. Bregar and M. Mohorčič},
		journal={IEEE Access},
		title={Improving Indoor Localization Using Convolutional Neural Networks on Computationally Restricted Devices},
		year={2018},
		volume={6},
		number={},
		pages={17429-17441},
		keywords={Computational modeling;Convolutional neural networks;Distance measurement;Estimation;Heuristic algorithms;Performance evaluation;Prediction algorithms;Channel impulse response;convolutional neural network;deep learning;indoor localization;non-line-of-sight;ranging error mitigation;ultra-wide band},
		doi={10.1109/ACCESS.2018.2817800},
		ISSN={},
		month={},}
		
## Author and license
Author of UWB data sets and corresponding Python scripts is Klemen Bregar, **klemen.bregar@ijs.si**. 

Data sets are licensed under Creative Commons Attribution Share Alike 4.0 license.

Copyright (C) 2018 SensorLab, Jožef Stefan Institute http://sensorlab.ijs.si

## Acknowledgement
The research leading to these results has received funding from the European Horizon 2020 Programme project eWINE under grant agreement No. 688116.