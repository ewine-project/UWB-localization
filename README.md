# UWB Localization
This repository contains the UWB data sets for localizization and NLoS classification. It contains also implementations of localization algorithms, localization error mitigation algorithms and NLoS classification and error regression building code. Ranging error regression and classification algorithms are implemented using [TensorFlow](https://www.tensorflow.org/) framework.

## Installation
The code needs the following python modules:
	- numpy
	- scikit-learn
	- tensorflow

### Installation on Ubuntu with Virtualenv
Installing Python software with Virtualenv can be useful to separate main system Python installation from separate installation with distinctive set of features and libraries installed. This prevents breaking the existing installation of Python and libraries.

###### 1. Install pip and Virtualenv:	
	$ sudo apt-get install python-pip python-dev python-virtualenv for Python 2.7
	$ sudo apt-get install python3-pip python3-dev python-virtualenv for Python 3.n

###### 2. Create a new virtual environment (in this case it's named 'tf' and is placed in /home/username/tf.):
	$ virtualenv --system-site-packages /home/username/tf # for Python 2.7
	$ virtualenv --system-site-packages -p python3 /home/username/tf # for Python 3.n

###### 3. Activate the new virtual environment 'tf':
	$ source ~/tf/bin/activate

###### 4. Install the latest pip:
	(tf)$ easy_install -U pip

###### 5. Install tensorflow in the activated 'tf' virtual environment:
	(tf)$ pip install --upgrade tensorflow       # Python 2.7
	(tf)$ pip3 install --upgrade tensorflow      # Python 3
	(tf)$ pip install --upgrade tensorflow-gpu   # Python 2.7 with gpu support
	(tf)$ pip3 install --upgrade tensorflow-gpu  # Python 3 with gpu support
	
If the installation process fails, for more details check the official [Tensorflow installation instructions](https://www.tensorflow.org/install/)

###### 6. Install Pandas Python package:
	(tf)$ pip install pandas   # Python 2.7
	(tf)$ pip3 install pandas  # Python 3
	
###### 7. Install scikit-learn Python package:
	(tf)$ pip install sklearn   # Python 2.7
	(tf)$ pip3 install sklearn  # Python 3
	
###### 8. Install scipy Python package:
	(tf)$ pip install scipy   # Python 2.7
	(tf)$ pip3 install scipy  # Python 3
	
### Tensorflow installation 
I prefer installing the TensorFlow using native pip. On 64-bit Ubuntu with no GPU support and Python3 installation goes like:
	
	sudo apt-get install python3-pip python3-dev
	pip3 install tensorflow

For complete installation instructions and installation on other platforms, please check the [Tensorflow installation](https://www.tensorflow.org/install/). 

### numpy installation
	pip3 install numpy

### Scikit-learn installation
	pip3 install sklearn

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
Author of code and data sets in this repository is Klemen Bregar, **klemen.bregar@ijs.si**.

See README.md files in individual sub-directories for details. 

Copyright (C) 2018 SensorLab, Jožef Stefan Institute http://sensorlab.ijs.si

## Acknowledgement
The research leading to these results has received funding from the European Horizon 2020 Programme project eWINE under grant agreement No. 688116.