from plotter import gen_margin_vs_training_error
from plotter import gen_training_gif

def main():
    print("Entered main")

    # Plot generation functions
    gen_margin_vs_training_error.main()
    gen_training_gif.main()


if __name__ == '__main__':
    main()
