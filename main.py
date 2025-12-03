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
    
    calc.evaluate_samples(n_samples=20)
    
    print("\n=== TEST SUR DE NOUVELLES ADDITIONS ===")
    test_cases = [
        (15, 27),
        (99, 1),
        (0, 0),
        (50, 50),
        (12, 88),
    ]
    
    for a, b in test_cases:
        calc.test_addition(a, b)
    
    print("\n" + "="*70)
    print("✓ Script terminé avec succès!")
    print("="*70)

