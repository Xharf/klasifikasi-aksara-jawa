from keras.utils import img_to_array
import numpy as np
#


def preprocessing(img):
    width, height = img.size
    img = img.crop(((width - min(img.size)) // 2,
                    (height - min(img.size)) // 2,
                    (width + min(img.size)) // 2,
                    (height + min(img.size)) // 2))
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    images /= 255
    return images
