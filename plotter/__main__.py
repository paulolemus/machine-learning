from plotter import gen_training_error_vs_margin
from plotter import gen_training_gif
from plotter import gen_training_error_vs_n
from plotter import gen_training_error_vs_lr

def main():
    print("Entered main")

    # Plot generation functions
    #gen_training_error_vs_margin.main()
    #gen_training_error_vs_n.main()
    gen_training_error_vs_lr.main()
    gen_training_gif.main()


if __name__ == '__main__':
    main()
