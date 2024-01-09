import matplotlib.pyplot as plt

def plot_and_save(values, title, y_label, file_name):
    plt.figure()
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()