import numpy as np
from numpy import array as Array
import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from typing import List, Tuple


class Calculator:
    def __init__(self, n_samples: int = 50000, digits: int = 2, latent_dim: int = 256, epochs: int = 50):
        self.n_samples = n_samples
        self.digits = digits
        self.latent_dim = latent_dim
        self.epochs = epochs
        
        self.inputs = None
        self.targets = None
        
        print(f"TensorFlow version: {tf.__version__}")
    
    def generate_data(self) -> Tuple[List[str], List[str]]:
        inputs, targets = [], []
        
        for _ in range(self.n_samples):
            a = np.random.randint(0, 10**self.digits)
            b = np.random.randint(0, 10**self.digits)
            
            inp = f"{a:0{self.digits}d}+{b:0{self.digits}d}"
            result = a + b
            out = f"{result:0{self.digits*2}d}"
            
            inputs.append([ord(c) for c in inp])
            targets.append([ord(c) for c in out])
        
        self.inputs = inputs
        self.targets = targets
        
        print(f"\n=== EXEMPLES DE DONNÉES GÉNÉRÉES ===")
        print(f"Premiers inputs (codes ASCII): {inputs[:3]}")
        print(f"Premiers targets (codes ASCII): {targets[:3]}")
        
        for i in range(3):
            inp_str = ''.join([chr(x) for x in inputs[i]])
            tgt_str = ''.join([chr(x) for x in targets[i]])
            print(f"Exemple {i+1}: '{inp_str}' → '{tgt_str}'")
        
        return inputs, targets

