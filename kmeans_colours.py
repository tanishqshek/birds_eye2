import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import cv2
import numpy as np
from PIL import Image


class KMeansColours:
    COLOR_DICT = {'Black': (0, 0, 0), 'White': (255, 255, 255), 'Red': (255, 0, 0), 'Lime': (0, 255, 0),
                  'Blue': (0, 0, 255), 'Yellow': (255, 255, 0), 'Cyan': (0, 255, 255), 'Magenta': (255, 0, 255),
                  'Silver': (192, 192, 192), 'Gray': (128, 128, 128), 'Maroon': (128, 0, 0), 'Olive': (128, 128, 0),
                  'Green': (0, 128, 0), 'Purple': (128, 0, 128), 'Teal': (0, 128, 128), 'Navy': (0, 0, 128)}
    IMAGE_STRING = None
    IMAGE = None
    CLUSTERS = None
    COLORS = None
    LABELS = None
    SIZE = 128, 128
    COLOR_THRESHOLD = None
    HISTOGRAM = None
    DESIRED_COLOR = None
    TARGET_DIRECTORY = None
    TARGET_DESTINATION = None

    def __init__(self, img, clusters, desired_color, color_threshold=20, file_dir='', file_dest=''):
        self.IMAGE_STRING = img
        self.CLUSTERS = clusters
        self.COLOR_THRESHOLD = color_threshold
        self.DESIRED_COLOR = desired_color
        self.TARGET_DIRECTORY = file_dir
        self.TARGET_DESTINATION = file_dest

    def manhattanDist(self, foo, bar):
        return abs(foo[2] - bar[2]) + abs(foo[1] - bar[1]) + abs(foo[0] - bar[0])

    def downscaleImage(self):
        # print(self.TARGET_DIRECTORY + self.IMAGE_STRING)
        im = Image.open(self.TARGET_DIRECTORY + self.IMAGE_STRING)
        im = im.convert('RGB')
        im.thumbnail(self.SIZE)
        # print(self.TARGET_DESTINATION + "thumbnail_" + self.IMAGE_STRING)
        im.save(self.TARGET_DESTINATION + "thumbnail_" + self.IMAGE_STRING)

    def getImage(self):
        return mpimg.imread(self.TARGET_DESTINATION + "thumbnail_" + self.IMAGE_STRING)

    def displayImage(self):
        plt.imshow(mpimg.imread(self.IMAGE_STRING))

    def processImage(self):
        self.downscaleImage()
        self.IMAGE = self.getImage()
        img_width = self.IMAGE.shape[0]
        img_height = self.IMAGE.shape[1]

        # print(self.IMAGE.shape)
        # print(img_width)
        # print(img_height)
        self.IMAGE = np.reshape(self.IMAGE, (img_width * img_height, 3))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # x = list(self.IMAGE[:, 0])
        # y = list(self.IMAGE[:, 1])
        # z = list(self.IMAGE[:, 2])

        # ax.scatter(x, y, z, color = 'c', marker = 'o')
        # ax.view_init(30, 210)
        # plt.show()

    def computeClusters(self):

        k_means = KMeans(n_clusters=self.CLUSTERS)
        k_means.fit(self.IMAGE)
        self.COLORS = k_means.cluster_centers_
        self.LABELS = k_means.labels_

    def getColors(self):
        return self.COLORS.astype(int)

    def getLabels(self):
        return self.LABELS

    def plotHistogram(self):

        # labels from 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]

        hist = hist[(-hist).argsort()]
        # print(hist*100)

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end

        # Display the histogram
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(chart)
        # plt.show()

        # Save the histogram
        self.HISTOGRAM = hist

        # Update the colors according to dominance
        self.COLORS = colors.astype(int)

    def colorPresent(self):
        """Uses sorted colour dictionary and checks whether any of the top 2 colours by manhattan distance
        match the desired colour and threshold for cluster size."""

        for i in range(len(self.COLORS)):
            # print("{}th color: ".format(i))
            temp = self.getColorDist(self.COLORS[i])
            # print(temp)
            if temp[0] == self.DESIRED_COLOR or temp[1] == self.DESIRED_COLOR:
                if (self.HISTOGRAM[i] * 100) >= self.COLOR_THRESHOLD:
                    return True
        return False

    def getColorDist(self, foo):
        """Accepts given cluster colour as argument and generates dictionary sorted by Manhattan distance
        of given colour with generic colour RGB values."""

        dist_dict = {}
        for color in self.COLOR_DICT:
            dist_dict[color] = self.manhattanDist(foo, self.COLOR_DICT.get(color))
        return sorted(dist_dict, key=dist_dict.get)

    def driver(self):
        self.processImage()
        self.computeClusters()
        self.plotHistogram()
        print("Desired Color - {} for image - {} is present: {}".format(self.DESIRED_COLOR, self.IMAGE_STRING, self.colorPresent()))
        return self.colorPresent()