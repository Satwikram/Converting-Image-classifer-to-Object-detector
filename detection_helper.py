import imutils

def sliding_window(image, step, ws):
    # Slide a window across a image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yeild the current window
            yield(x, y, image[y: y + ws[1], x: x + ws[0]])

def image_pyramid(image, scale = 1.5, minSize = (224, 224)):
    # Yield the image
    yield image

    # Keep Loop over the image Pyramid
    while True:
        # Compute the dimension of the next image in the pyramid
        w = int(image.shape[1]/scale)
        image = imutils.resize(image, width = w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in  the pyramid
        yield image

