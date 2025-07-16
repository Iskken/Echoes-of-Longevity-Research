from Bio import Entrez
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def yearly_distribution_hist(df):
    df['year'].hist()
    plt.show()