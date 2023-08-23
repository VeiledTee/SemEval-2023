# SemEval-2023
This repo houses the code submitted to [SemEval 2023 for Task 4: Human Value Detection](https://touche.webis.de/semeval23/touche23-web/index.html). The peer-reviewed paper accosiated with this repo can be found [here](https://aclanthology.org/2023.semeval-1.29/).

### Background
There are four models within this repository, a supervised XGBoost, two supervised Ensemble, and an unsupervised Threshold Comparison model.

### Executing
Executing the ``main,py`` script will run all four models on the training and validation data. Both ``bert-base-uncased`` and ``all-MiniLM-L6-v2`` were
used to generate embeddings for the data provided by the task organizers. The hyperparameters have been tuned to the ``F1-score`` evaluation metric 
used in the task.
