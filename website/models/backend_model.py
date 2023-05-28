import os, sys
from PIL import Image

# debug mode don't need to load model
is_debug = False
if not is_debug:
    # append muse to path
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from muse import utils_model
else:
    pass

# fake model for testing
def inference(text):
    if is_debug:
        # load images/test_image.png
        image = Image.open("static/images/test_image.png")
    else:
        lst_text = [text]
        images = utils_model.inference(lst_text)
        image = images[0]
    # save to images/gen_image.png
    path_img = os.path.join("static/images/gen_image.png")
    image.save(path_img)
    return path_img
