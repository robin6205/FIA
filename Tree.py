"""
__author__ = Younggil Chang
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


# slope calculation of linear regression
def estimated_coefficent(x, y):
    n_size = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xx = np.sum(x * x) - n_size * m_x * m_x
    SS_xy = np.sum(y*x) - n_size *m_y*m_x
    slope = SS_xy / SS_xx
    intersect = m_y - slope*m_x
    return(intersect, slope)


def plot_linear_regression_line(x, y, b):
    plt.scatter(x, y, color = "g",marker = "-", s = 30)
    y_predict = b[0] + b[1]*x
    plt.plot(x, y_predict, color = "g")
    plt.title("Linear Regression Plot of ground level")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.savefig("tree depth plot.png")


if __name__ == "__main__":

    #composite image filter
    img = cv2.imread('treedepth.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img,(5,5))
    img = cv2.Canny(img, 100, 200)


    depth = [len(img[0])*[0] ] * len(img) #empty depth data

    # 2d array of depth information

    edge_info = []
    for i in range(len(img[0])):
        for j in range(len(img)):
            if img[j][i] > 50:
                edge_info.append(len(img)-j)
                break
    edge_info = np.array(edge_info)

    avg = np.mean(edge_info, axis=0)
    std = np.std(edge_info, axis=0)

    yaxis = np.array([e for e in edge_info if (avg - 2 * std < e < avg + 2 * std)])  # Remove Outlier
    xaxis = np.linspace(1, len(yaxis), len(yaxis))
    intersection, slope = estimated_coefficent(xaxis, yaxis)
    theta = np.arctan(slope)

    d_list = [] #diameter list
    for i in range(int(0.45 * len(img)), int(0.55 * len(img))):
        diameter = []
        for j in range(len(img[i])):
            ind = j
            while img[i][ind] < 50:
                img[i][ind] = 200
                ind += 1
                if ind >= len(img[i]):
                    break
            if ind > j:
                diameter.append(ind - j)
        d_list.append(diameter)

    result = []
    ind = min([len(x) for x in d_list])
    for j in range(ind):
        temp = 0
        thresh = 0
        for i in range(len(d_list)):
            temp += d_list[i][j]
            thresh += 1
        result.append(np.cos(theta) * np.round(temp/thresh, 4))

    print(result)
    print(theta * 180 / np.pi)
    cv2.imshow("result.jpg", img)