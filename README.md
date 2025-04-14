# About
This repository provides three regression-based approaches (M1, M2, and M3) for melody estimation from polyphonic audios. The repository includes the implementation and pre-trained models.

> Paper: (Link)

# Setup
## Clone the Repository 
```
git clone https://github.com/KavyaRSaxena/me_reg_taslp
cd me_reg_taslp
```

# Usage
To test the different methods, go to the test directory
```
cd testing_codes
```
To test the M1 method:
```
python3 M1.py 
```

To test the M2 method:
```
python3 M2.py 
```

To test the M3 method:
```
python3 M3.py 
```

# Pre-trained Models
The pre-trained models for each method are present in the **model_weights** folder.

# Datasets
The test datasets used are: [ADC2004](http://labrosa.ee.columbia.edu/projects/melody/), [MIREX05](http://labrosa.ee.columbia.edu/projects/melody/), and [HAR](https://zenodo.org/records/8252222)

