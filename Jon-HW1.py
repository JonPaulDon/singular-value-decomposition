import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot as plt

tiger = matplotlib.pyplot.imread("image.jpg")
# print(len(tiger))    # len of tiger
# print(tiger.ndim)

graytiger = np.average(tiger, -1)
"""
U- Left singular vectors
Vh- right singular vectors
S-singular values(ordered by importance) 
"""
u, s, vh = np.linalg.svd(graytiger, full_matrices=False)

img = plt.imshow(graytiger)
img.set_cmap('gray')
plt.title('Original Image')
plt.show()

s = np.diag(s)
for k in (1, 5, 10, 20, 40):
    """
    Multiply the first k columns of U
First k x k block of s
First K columns of vh
"""
    myapproximation = u[:, :k] @ s[0:k, :k] @ vh[:k, :]
    img=plt.imshow(myapproximation)
    img.set_cmap('gray')
    plt.title('k=' + str(k))
    plt.show()
