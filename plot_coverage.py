import matplotlib.pyplot as plt


def main():
    with open("data/total_coverage.txt", 'r') as f:
        lines = f.readlines()

    data = [int(line.strip()) for line in lines]

    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    main()
