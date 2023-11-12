
# main.py
from neural_network import NeuralNetwork
from gui import GUI

def main():
    neural_network = NeuralNetwork()
    gui = GUI(neural_network)
    gui.root.mainloop()

if __name__ == "__main__":
    main()
