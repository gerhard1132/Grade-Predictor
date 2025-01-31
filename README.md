# Grade Predictor 

This repository and its strukture is based on AI-CPS by Marcus Grum. It intends to estimate a student grade based on the following parameters:

1. Socioeconomic Score,
2. Study Hours,
3. Sleep Hours and
4. Attendance (%).

A following estimate of a grade between 0-100 is made by a neural network. 

# Getting Started
Run the provided OLS or AI model with the corresponding `docker compose` command:

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
Zunaira _(iqbal2@uni-potsdam.de)_

# Status
Finished project! (Project done under module AIBAS at the Univserity of Potsdam)
