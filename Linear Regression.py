import matplotlib.pyplot as plt
import numpy as np

def error(true_y,pred_y):
    cost = np.sum(((true_y-pred_y)**2))/len(true_y)
    return cost
def gd(x,y):
    curr_w = 0.1
    curr_b = 0.01
    iterations = 100
    learning_rate = 0.0001
    n = float(len(x))
    costs = []
    weights = []
    previous_cost = None
    for i in range(iterations):
        pred_y = (curr_w*x)+curr_b
        current_cost = error(y,pred_y)
        if previous_cost and abs(previous_cost-current_cost)<=1e-6:
            break
        previous_cost = current_cost
        costs.append(current_cost)
        weights.append(curr_w)
        weight_derivative = -(2/n)*sum(x*(y-pred_y))
        bias_derivative = -(2/n)*sum(y-pred_y)
        curr_w-=(learning_rate*weight_derivative)
        curr_b-=(learning_rate*bias_derivative)
        print(f"Iterartion {i+1}: Cost {current_cost} Weight {curr_w} Bias {curr_b} ")
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()
    return curr_w,curr_b


def main():
    """
    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
           55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
           45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
           48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
           78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
           55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
           60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
    """
    """
    X = np.array([0,5,10,15,20,25])
    Y = np.array([12,15,17,22,24,30])
    """
    X = np.array([5,10,15,20,25])
    Y = np.array([16,19,23,26,30])
    
    w,b = gd(X,Y)
    pred_y = w*X+b
    print(f"Estimated Weight: {w}\nEstimated bias: {b}")
    
    plt.scatter(X,Y,marker = 'o',color='blue')
    plt.plot(X,pred_y,color = 'red',ls = 'dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()
main()
