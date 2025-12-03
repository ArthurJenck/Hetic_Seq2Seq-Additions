from calculator import Calculator


if __name__ == "__main__":
    calc = Calculator(n_samples=50000, digits=2, latent_dim=256, epochs=50)
    
    calc.generate_data()
    calc.build_vocabulary()
    calc.prepare_training_data()
    calc.build_model()
    calc.train()
    calc.plot_training_history()

