# Using Artificial Intelligence for Estimating People's Perception of the World
Project for M.Sc. Thesis DTU Fall 2020 

## Setup to run code
### Setup
To run the code, the model weight for detected targets needs to be added.
These are not added in the git because they are too large to commit.
The only model weights used for this project is *demo.pt*, which can be found at their git repository: https://github.com/ejcgt/attention-target-detection
The file is to be placed in the following folder *Estimating_individuals_perspectives/detecting_attended_targets/model_weights/*.

### Run
The main function is *estimating_individuals_perspectives.py*, where the parameters for which method to run and what to show are defined in the *__init__()* function.

The directory where the videos are located can be added in the main function.

To analyse the results, run the *Analyse_data/analyse_data.py* and ensure that the input directory is equal to the output from *estimating_individuals_perspectives.py*. 

*The code need some refactoring to be more easily understood, but this has not been done due to time constraints.*

## Test Data and Results
The raw videos and estimated results can be found in *Test_Data_Result*
All videos are compressed into zip files. 
Some of the videos are available on Youtube using the follwoing links: 
* https://youtu.be/94kip8cn4wk
* https://youtu.be/3AOQ5bDbKgI
* https://youtu.be/QcPwwhqfaDg

## Github Repositories
The used github repositories for this project are:

https://github.com/Erkil1452/gaze360
https://github.com/ejcgt/attention-target-detection
https://github.com/ageitgey/face_recognition
https://github.com/facebookresearch/detectron2
