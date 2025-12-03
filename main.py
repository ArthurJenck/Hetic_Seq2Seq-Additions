from calculator import Calculator


if __name__ == "__main__":
    calc = Calculator(n_samples=50000, digits=2, latent_dim=256, epochs=50)
    
    calc.generate_data()

