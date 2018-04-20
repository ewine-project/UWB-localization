# UWB Localization Data Set

Localization data set was created using [SNPN-UWB](http://www.log-a-tec.eu/mtc.html) board with DecaWave [DWM1000](https://www.decawave.com/products/dwm1000-module) UWB pulse radio module.

## Data Set Description
Data set was generated during two measurement campaigns in different office environments. Individual measurement campaigns are covered under [Data Set 1](###Data Set 1) and [Data Set 2](###Data Set 2).

The structure of measurement files is explained under the [File Structure](##File structure) section.

### Data Set 1
This data set was recorded in two adjacent office rooms with the connecting hallway parallel to both the offices. For each file in the dataset1 folder UWB tag device was placed at one fixed position and UWB anchor was consecutively placed throughout the whole location (hallway, room0 and room1).
* tag_room0.csv with measurements for 48 anchor positions with 100 samples for each position (48 x 100 = 4800 measurements)
* tag_room1.csv with measurements for 51 anchor positions with 100 samples for each position (51 x 100 = 5100 measurements) 


### Data Set 2
The second data set was recorded in a different office environment with multiple rooms included. Again, two UWB tag positions were selected and UWB anchor was consecutively placed in several different positions throughout the indoor environment. Measurements can be found in the dataset2 folder, where two separate files are included fo each tag position:
* tag_room0.csv with measurements for 35 anchor positions with 100 samples for each position (35 x 100 = 3500 measurements)
* tag_room1.csv with measurements for 43 anchor positions with 100 samples for each position **but with additional measurements for all 5 available UWB channels** (43 x 5 x 100 = 21600 measurements)
	* this file is split to several smaller files because of its size

## Data Set Structure
Folder with data set is organized as follows:

	+ code
		|__ uwb_dataset.py
	+ dataset1
		|__ tag_room0.csv
		|__ tag_room1.csv
    + dataset2
		|__ tag_room0.csv
		|__ tag_room1
			|__ tag_room1_part0.csv
			|__ tag_room1_part1.csv
			|__ tag_room1_part2.csv
			|__ tag_room1_part3.csv


## File Structure
Each line in data set files represent one measured sample. Elements of each data set sample are:
* TAG x-position
* TAG y-position
* ANCHOR x-position
* ANCHOR y-position
* Measured range (in meters)
* NLOS (1 if NLOS, 0 if LOS)
* TAG ID
* ANCHOR ID
* FP_IDX (index of detected first path element in channel impulse response (CIR) accumulator: in data set it can be accessed by **first_path_index+15**)
* RSS
* RSS of first path signal
* FP_AMP1 (first path amplitude - part1) [look in user manual](http://thetoolchain.com/mirror/dw1000/dw1000_user_manual_v2.05.pdf)
* FP_AMP2 (first path amplitude - part2) [look in user manual](http://thetoolchain.com/mirror/dw1000/dw1000_user_manual_v2.05.pdf) 
* FP_AMP3 (first path amplitude - part3) [look in user manual](http://thetoolchain.com/mirror/dw1000/dw1000_user_manual_v2.05.pdf)
* STDEV_NOISE (standard deviation of noise)
* CIR_PWR (total channel impulse response power)
* MAX_NOISE (maximum value of noise)
* RXPACC (received RX preamble symbols)
* CH (channel number)
* FRAME_LEN (length of frame)
* PREAM_LEN (preamble length)
* BITRATE
* PRFR (pulse repetition frequency rate in MHz)
* 802.15.4 UWB preamble code
* CIR (absolute value of channel impulse response: 1016 points with 1 nanosecond resolution)

## Importing Data Set in Python
To import data set data into Python environment, **uwb_dataset.py** script from folder **code** can be used. The CIR data still needs to be divided by number of acquired RX preamble samples (RX_PACC).

	import uwb_dataset
	
	# import raw data
	data = uwb_dataset.load_data_from_file('../dataset1/tag_room0.csv')
	
	# divide CIR by RX preable count (get CIR of single preamble pulse)
	# item[17] represents number of acquired preamble symbols
	for item in data:
		item[24:] = item[24:]/float(item[17])
	
	print(data)

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
Author of UWB LOS and NLOS Data Set and corresponding Python scripts is Klemen Bregar, **klemen.bregar@ijs.si**. 

Data set is licensed under Creative Commons Attribution Share Alike 4.0 license.

Copyright (C) 2018 SensorLab, Jožef Stefan Institute http://sensorlab.ijs.si

## Acknowledgement
The research leading to these results has received funding from the European Horizon 2020 Programme project eWINE under grant agreement No. 688116.