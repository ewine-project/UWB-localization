# Localization and classification evaluation
This directory holds several Python scripts intended to evaluate localization using and combining several localization error mitigation techniques with classification and regression.

## Scripts
Note: Before running scripts, model builder scripts under NLOSClassificationModel and RangingErrorModel should be run to properly build NLoS classification and ranging error regression models based on included data sets.

### loc_ls_class_eval.py
This script evaluates localization performance using least squares multilateration location estimation algorithm using NLoS classification.

### loc_ls_eval.py
This script evaluates localization performance of least squares multilateration without any error mitigation technique.

### loc_ls_los_eval.py
This script evaluates localization performance of least squares multilateration localization algorithm using only LoS measurements (determined based on a priory knowledge recorded in a dataset).

### loc_ls_regress_eval.py
This script evaluates localization performance of least squares multilateration localization algorithm using ranging error compensation using ranging error regression model.

### loc_wls_eval.py
This script evaluates localization performance of weighted least squares multilateration localization algorithm using estimated ranging errors (error regression model) as range weights.

### loc_wls_regress_eval.py
This script evaluates localization performance of a weighted least squares multilateration localization algorithm using estimated ranging errors as weights and error compensation input.

## Author and licence
Author of demonstration and evaluation scripts is Klemen Bregar, **klemen.bregar@ijs.si**. 

Copyright (C) 2018 SensorLab, Jožef Stefan Institute http://sensorlab.ijs.si

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses