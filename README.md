# birdclef-2024

- https://www.imageclef.org/node/316

## quickstart

Install the pre-commit hooks for formatting code:

```bash
pre-commit install
```

We are generally using a shared VM with limited space.
Install packages to the system using sudo:

```bash
sudo pip install -r requirements.txt
```

We can ignore the following message since we know what we are doing:

```
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

This should allow all users to use the packages without having to install them multiple times.
This is a problem with very large packages like `torch` and `spark`.

Then install the current package into your local user directory, so changes that you make are local to your own users.

```bash
pip install -e .
```

## kaggle

Here are a few resources for running the code on Kaggle:

- https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier
  - This is the google vocalization classifier that we're using as a starting point.
- https://www.kaggle.com/code/acmiyaguchi/dsgt-birdclef-2024-package-sync
  - This is a script that will sync the code from the repo to the Kaggle notebook.
- https://www.kaggle.com/code/acmiyaguchi/dsgt-birdclef-2024-model-sync
  - This is a script that will sync the model from google cloud storage to the Kaggle notebook.
- https://www.kaggle.com/code/acmiyaguchi/dsgt-birdclef-2024-vocalization-inference
  - This is a script that will run the inference on the Kaggle notebook.
