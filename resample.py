import numpy
import cv2
import math


def resampleImage(originalImage, newHeight, newWidth, method):
    oldHeight, oldWidth, c = originalImage.shape
    # the new pixel grid
    resizedGrid = numpy.zeros((newHeight, newWidth, c))
    # horizontal and vertical scales
    widthScale = (oldWidth) / (newWidth) if newHeight != 0 else 0
    heightScale = (oldHeight) / (newHeight) if newWidth != 0 else 0

    for row in range(newHeight):
        for column in range(newWidth):
            # Compute position in relation to N4(p) neighbours.
            x = row * heightScale
            y = column * widthScale

            xFloor = math.floor(x)
            xCeil = min(oldHeight - 1, math.ceil(x))
            yFloor = math.floor(y)
            yCeil = min(oldWidth - 1, math.ceil(y))

            # compute the new pixel value
            if method == "bi-linear":
                # catch pixel aligns
                if (xCeil == xFloor) and (yCeil == yFloor):
                    newPixel = originalImage[int(x), int(y), :]
                elif (xCeil == xFloor):
                    newPixel = originalImage[int(x), int(
                        yFloor), :] * (yCeil - y) + originalImage[int(x), int(yCeil), :] * (y - yFloor)
                elif (yCeil == yFloor):
                    newPixel = (originalImage[int(xFloor), int(
                        y), :] * (xCeil - x)) + (originalImage[int(xCeil), int(y), :] * (x - xFloor))
                else:
                    I1 = originalImage[xFloor, yFloor, :]
                    I2 = originalImage[xCeil, yFloor, :]
                    I3 = originalImage[xFloor, yCeil, :]
                    I4 = originalImage[xCeil, yCeil, :]

                    newPixel = I1 * (xCeil - x)*(yCeil - y) + I2 * (x - xFloor)*(yCeil - y) + I3 * (
                        xCeil - x) * (y - yFloor) + I4 * (x - xFloor) * (y - yFloor)

            elif method == "nearest":
                I1 = originalImage[xFloor, yFloor, :]
                I2 = originalImage[xCeil, yFloor, :]
                I3 = originalImage[xFloor, yCeil, :]
                I4 = originalImage[xCeil, yCeil, :]

                d4 = max((xCeil - x), (yCeil - y))
                d3 = max((x - xFloor), (yCeil - y))
                d2 = max((xCeil - x), (y - yFloor))
                d1 = max((x - xFloor), (y - yFloor))

                nearestIndex = numpy.argmin(
                    numpy.array([d1, d2, d3, d4]))

                newPixel = (I1, I2, I3, I4)[nearestIndex]

            resizedGrid[row, column, :] = newPixel

    return resizedGrid.astype(numpy.uint8)


originalImage = cv2.imread('a.jpg')


newImg = resampleImage(originalImage, 1000, 500, "bi-linear")
# print(newImg)
cv2.imwrite('newBiLinear.jpg', newImg)


newImg = resampleImage(originalImage, 1000, 500, "nearest")
# print(newImg)
cv2.imwrite('newNearest.jpg', newImg)
