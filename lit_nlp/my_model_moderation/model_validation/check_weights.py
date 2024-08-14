import numpy as np


def read(file_path):
    return np.loadtxt(file_path)


def main():
    file1 = read("pt_deberta_weights.txt")
    file2 = read("tf_deberta_weights.txt")

    differences = np.where(file1 != file2)
    num_differences = len(differences[0])

    if num_differences == 0:
        print("All weights are the same.")
    else:
        print(f"Found {num_differences} differences:")
        for index in zip(*differences):
            print(f"Difference at index {index}: {file1[index]} vs {file2[index]}")


if __name__ == "__main__":
    main()
