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
        
        self.MAX_LEN_IN = digits * 2 + 1
        self.MAX_LEN_OUT = digits * 2
        
        self.char_to_idx = None
        self.idx_to_char = None
        self.VOCAB_SIZE = None
        
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        
        self.model = None
        self.encoder_inputs = None
        self.encoder_states = None
        self.decoder_inputs = None
        self.decoder_lstm = None
        self.decoder_dense = None
        
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
    
    def build_vocabulary(self):
        all_chars = set()
        for inp in self.inputs:
            all_chars.update([chr(x) for x in inp])
        for tgt in self.targets:
            all_chars.update([chr(x) for x in tgt])
        
        chars = sorted(list(all_chars))
        
        self.char_to_idx = {c: i for i, c in enumerate(['\t'] + chars + ['\n'])}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.VOCAB_SIZE = len(self.char_to_idx)
        
        print(f"\n=== VOCABULAIRE ===")
        print(f"Taille du vocabulaire: {self.VOCAB_SIZE} caractères")
        print(f"Caractères: {chars}")
        print(f"Mapping exemple: '0' → index {self.char_to_idx['0']}")
        print(f"Tokens spéciaux: START='\\t' (idx {self.char_to_idx[chr(9)]}), END='\\n' (idx {self.char_to_idx[chr(10)]})")
    
    def prepare_data(self, seqs: List[str], max_len: int) -> Array:
        X = np.zeros((len(seqs), max_len, self.VOCAB_SIZE), dtype=np.float32)
        
        for i, seq in enumerate(seqs):
            for t, char_code in enumerate(seq):
                if t < max_len:
                    char = chr(char_code)
                    if char in self.char_to_idx:
                        char_idx = self.char_to_idx[char]
                        X[i, t, char_idx] = 1.0
        
        return X
    
    def prepare_training_data(self):
        print("\n=== PRÉPARATION DES DONNÉES ===")
        
        self.encoder_input_data = self.prepare_data(self.inputs, self.MAX_LEN_IN)
        
        decoder_input_seqs = []
        for tgt in self.targets:
            decoder_input = [ord('\t')] + tgt[:-1]
            decoder_input_seqs.append(decoder_input)
        
        self.decoder_input_data = self.prepare_data(decoder_input_seqs, self.MAX_LEN_OUT)
        self.decoder_target_data = self.prepare_data(self.targets, self.MAX_LEN_OUT)
        
        print(f"Shape encoder_input_data: {self.encoder_input_data.shape}")
        print(f"  → (n_samples={self.encoder_input_data.shape[0]}, timesteps={self.encoder_input_data.shape[1]}, vocab={self.encoder_input_data.shape[2]})")
        print(f"Shape decoder_input_data: {self.decoder_input_data.shape}")
        print(f"Shape decoder_target_data: {self.decoder_target_data.shape}")
    
    def build_model(self):
        print("\n=== CONSTRUCTION DU MODÈLE ===")
        
        self.encoder_inputs = Input(shape=(None, self.VOCAB_SIZE), name='encoder_input')
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]
        
        print(f"✓ Encodeur créé: LSTM avec {self.latent_dim} unités")
        
        self.decoder_inputs = Input(shape=(None, self.VOCAB_SIZE), name='decoder_input')
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs, initial_state=self.encoder_states)
        
        self.decoder_dense = Dense(self.VOCAB_SIZE, activation='softmax', name='decoder_output')
        decoder_outputs = self.decoder_dense(decoder_outputs)
        
        print(f"✓ Décodeur créé: LSTM {self.latent_dim} unités + Dense {self.VOCAB_SIZE} sorties")
        
        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs, name='seq2seq_training')
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n=== ARCHITECTURE DU MODÈLE ===")
        self.model.summary()

