# Machine Learning 

Authors: Veronika Krobotová, Jakub Všetečka, Tomáš Hobza

## Environment

Run:

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

If you install any additional packages, please remember to re-freeze the requirements by running:

```bash
pip freeze > requirements.txt
```

## Data
To fetch data run:
```bash
python scripts/fetch_data.py
```

We are using the following datasets:

| Dataset | Features | Samples | Target | Year | Characteristics |
|---------|----------|---------|--------|------|-----------------|
| [Housing Prices](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) | 13 | 545 | price | 2021 | numerical : categorical → 50 : 50 |
| [Spotify Tracks](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | 21 | 114,000 | popularity (0-100) | 2022 | 1 categorical but interesting ranges for numerical |
| [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) | 6 | 10,000 | performance (0-100) | 2023 | 1 categorical but uniform distribution for numeric variables → interesting |