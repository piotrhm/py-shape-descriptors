import numpy
import math
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from optparse import OptionParser
from random import randrange


def get_coordinates(n, r=20, theta=0, size=60):
    # n = number of vertices in the regular polygon
    # r = radius of the circumscribing circle, a small radius would mean low resolution
    # theta = angle by which the polygon is to be rotated. eg: alpha = 45
    # (x,y) is the center of the polygon

    x = y = size / 2
    coordinates = []
    for i in range(0, n):
        angle = theta * math.pi + (2 * math.pi * i) / n
        coordinates.append((int(round(x + r * math.cos(angle))),
                            int(round(y + r * math.sin(angle)))))
    return coordinates


def gen_polygon_mask(n=3, r=20, theta=0):
    size = 3 * r
    polygon = get_coordinates(n, r, theta, size)

    img = Image.new('L', (size, size), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return numpy.array(img)


parser = OptionParser()
parser.add_option("-f", "--file",
                  dest="filename",
                  help="save polygon(s) under filename_(%d).png")
parser.add_option("-d", "--dim",
                  dest="dim",
                  help="mask dimensions")
parser.add_option("-n", "--num",
                  dest="n",
                  help="number of polygon points")

(options, args) = parser.parse_args()

if options.filename is None:
    filename = "poly"
else:
    filename = options.filename

cwd = os.getcwd()
if not os.path.exists(cwd+'/gen'):
    os.makedirs(cwd+'/gen')

for i in range(10):
    path = cwd+'/gen/'+filename+str(i)+'.png'

    rotation = randrange(180)
    print(rotation)

    gen_poly = gen_polygon_mask(n=int(options.n), r=int(options.dim), theta=rotation/180)
    plt.imshow(gen_poly, cmap='gray')
    plt.imsave(path, gen_poly, cmap='gray')
    plt.show()
