from calculator import Calculator
import numpy as np


if __name__ == "__main__":
    calc = Calculator(n_samples=50000, digits=2, latent_dim=256, epochs=50)
    
    calc.generate_data()
    calc.build_vocabulary()
    calc.prepare_training_data()
    calc.build_model()
    calc.train()
    calc.plot_training_history()
    calc.build_inference_models()
    
    print("\n=== TEST DÉCODAGE ===")
    test_seq = calc.encoder_input_data[0:1]
    result = calc.decode_sequence(test_seq)
    original = ''.join([calc.idx_to_char[np.argmax(x)] for x in test_seq[0] if np.max(x) > 0])
    print(f"Test: '{original}' → '{result}'")

