import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


def k_mean(K, X, feature_num, max_iter=100000, center=[]):
    centroids = center

    labels, cluster, centroids_new = add_to_cluster(K, centroids, X, feature_num)
    # print(centroids)
    for n in range(max_iter):
        print(centroids_new)
        # print(centroids_new)
        if centroids == centroids_new:
            # print(np.asarray(labels, dtype=float))
            # print(expect)
            return cluster, labels, centroids
            # return cluster, labels
        else:
            centroids = centroids_new
            labels, cluster, centroids_new = add_to_cluster(K, centroids, X, feature_num)

    return cluster, labels, centroids

# compute within group center(average)
def compute_new_centroids(o_cluster):
    centroids = []
    empty_cluster = []
    sse_max = ([], -1)
    for i in o_cluster:
        if len(o_cluster[i]) != 0:

            mss = np.sum(o_cluster[i], axis=0)

            temp = list(map(lambda x: x / len(o_cluster[i]), mss))
            centroids.append(temp)
            sse = 0
            for point in o_cluster[i]:
                sse += euclidean(point, centroids[-1]) ** 2
            if sse >= sse_max[1]:
                sse_max = (o_cluster[i], sse)
        else:
            centroids.append([])
            empty_cluster.append(i)
    for i in empty_cluster:
        centroids[i - 1] = sse_max[0][np.random.randint(0, len(sse_max[0]))]

    return centroids

# base on distance add to cluster
def add_to_cluster(K, centroids, X, feature_num):
    cluster = {i + 1: [] for i in range(K)}
    labels = []
    a = X.reshape(-1, feature_num)

    for x in a:
        # print(x)
        minimum = (0, 1000000)
        for i in range(len(centroids)):
            dist = euclidean(x, centroids[i])

            if dist <= minimum[1]:
                minimum = (i + 1, dist)
        cluster[minimum[0]].append(x)
        labels.append(minimum[0])

    centroids_new = compute_new_centroids(cluster)
    return labels, cluster, centroids_new

# perform distance calculation
def euclidean(point1, point2):
    dist = 0

    for p1, p2 in zip(point1, point2):
        dist += ((p1 - p2) ** 2)

        # print(dist)
    dist = np.sqrt(dist)

    return dist

# plot image base on center lable and title
def plot_quantization(centers, label, title, h, w):
    label = np.asarray(label)
    label = label.reshape((h, w))

    k1_result = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            k1_result[i, j] = np.asarray(centers[label[i, j] - 1])

    cv2.imwrite(title, k1_result)


if __name__ == '__main__':
    filename = "../data/t3.jpg"
    k = 2
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    if len(sys.argv) == 3:
        filename = sys.argv[1]
        k = int(sys.argv[2])
    img = cv2.imread(filename)
    h, w, c = img.shape
    center1 = [list(img[np.random.randint(0, h), np.random.randint(0, w)]) for i in range(k)]
    cluster, label, centers = k_mean(k, img, 3, 20, center1)
    plot_quantization(centers, label, "output.jpg", h, w)
