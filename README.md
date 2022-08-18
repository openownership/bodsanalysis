# BODS analysis notebook

This repository contains notebooks and code for analysing data published to the [Beneficial Ownership Data Standard](https://standard.openownership.org/) (BODS). The work contained here has been produced as part of the [Opening Extractives programme](https://www.openownership.org/en/topics/opening-extractives/) which is implemented jointly between the [Extractive Industries Transparency Initiative International Secretariat](https://eiti.org/opening-extractives) and Open Ownership.

The main components are:

- A Python module `qbods.py`, which contains a set of functions for reading, summarising and analysing BODS data. This code is under development and will likely contain bugs.

- An iPython notebook `latvia_demo.ipynb`, which contains code to run a subset of the functions on an [initial dataset](https://data.gov.lv/dati/lv/dataset/plg-bods/resource/19a7d5f5-5586-4de2-a710-fc7145a129f2) released by the Register of Enterprises of the Republic of Latvia, with accompanying text.

Additional notebooks will be added to the repository as this work progresses.

## Running locally

Clone the repository, open the notebook in a suitable program (e.g. [VS code](https://code.visualstudio.com/)), and follow setup instructions within the notebook.

## Running on Deepnote/Google Colab

To run on [Deepnote](https://deepnote.com/), clone this repository, then create a new project and upload the notebook, alongside `qbods.py` and `requirements.txt` as files. Then open the notebook and follow setup instructions.

To run on [Google Colab](https://colab.research.google.com/), clone this repository, then click File > Upload notebook, and upload the notebook. Then in the left hand menu, click on icons for 'Files' then 'Upload Files', then upload the files `qbods.py` and `requirements.txt`. Then open the notebook and follow setup instructions.

## Contributing

Suggestions for new queries and contributions are welcomed via issues and pull requests, respectively