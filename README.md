# Grade Predictor 

This repository and its structure is based on AI-CPS by Marcus Grum. It intends to estimate a student grade based on the following parameters:

1. Socioeconomic Score,
2. Study Hours,
3. Sleep Hours and
4. Attendance (%)

A following estimate of a grade between 0-100 is made by a neural network. 

# Getting Started

This project includes two approaches for grade prediction:

*  OLS Regression (Ordinary Least Squares): Uses traditional statistical regression  
   for prediction
*  Neural Network (AI Model): Uses TensorFlow/Keras for deep learning prediction

First, make sure you have a docker volume named _ai_system_ on your system. If not, create it with the command 
```bash
docker volume create ai_system
```

After changing your students information under '~/images/activationBase_GradePredictor/activation_data.csv', build your current activation image with:
```bash
docker build -t gerhard1132/activationbase-gradepredictor .
```
Finally, run the provided OLS or AI model with the corresponding `docker compose` command (Ensure Docker is installed on your system):

1. OLS model: 
```bash
docker compose -f .\docker-compose.ols.yml up
```

2. AI model:
```bash
docker compose -f .\docker-compose.ai.yml up
```

Every image used in this repository can be found on the dockerhub account [here](https://hub.docker.com/u/gerhard1132).

# Authors
Roman Klinghammer _(rklinghammer@uni-potsdam.de)_
Zunaira Iqbal _(iqbal2@uni-potsdam.de)_

# License 

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](./LICENSE) file for details.


# Status
Finished project! (Project done under module AIBAS at the University of Potsdam)
