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
        
        self.history = None
        
        self.encoder_model = None
        self.decoder_model = None
        
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
    
    def train(self, batch_size: int = 64, validation_split: float = 0.2):
        print("\n=== ENTRAÎNEMENT ===")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            [self.encoder_input_data, self.decoder_input_data],
            self.decoder_target_data,
            batch_size=batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Entraînement terminé!")
    
    def plot_training_history(self):
        print("\n=== VISUALISATION ===")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['loss'], label='Loss (train)', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Loss (validation)', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Évolution de la Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history.history['accuracy'], label='Accuracy (train)', linewidth=2)
        ax2.plot(self.history.history['val_accuracy'], label='Accuracy (validation)', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title("Évolution de l'Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✓ Graphiques sauvegardés dans 'training_history.png'")
        plt.show()
    
    def build_inference_models(self):
        print("\n=== CRÉATION DES MODÈLES D'INFÉRENCE ===")
        
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states, name='encoder_inference')
        print("✓ Modèle encodeur d'inférence créé")
        
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='decoder_state_h')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='decoder_state_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs,
            initial_state=decoder_states_inputs
        )
        
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        
        self.decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states,
            name='decoder_inference'
        )
        
        print("✓ Modèle décodeur d'inférence créé")
    
    def decode_sequence(self, input_seq: Array) -> str:
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        
        target_seq = np.zeros((1, 1, self.VOCAB_SIZE))
        target_seq[0, 0, self.char_to_idx['\t']] = 1.0
        
        stop_condition = False
        decoded_sentence = ''
        
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value,
                verbose=0
            )
            
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.idx_to_char[sampled_token_index]
            
            decoded_sentence += sampled_char
            
            if sampled_char == '\n' or len(decoded_sentence) >= self.MAX_LEN_OUT:
                stop_condition = True
            
            target_seq = np.zeros((1, 1, self.VOCAB_SIZE))
            target_seq[0, 0, sampled_token_index] = 1.0
            
            states_value = [h, c]
        
        return decoded_sentence.replace('\n', '')

