import random
import torchvision.transforms.functional as TF


# reproducibility
SEED = 42
random.seed(SEED)

# parameters
SIZE = 128


def resize_bbox(bboxes, w_ratio, h_ratio):
    """
    Resize bounding boxes based on width and height ratios.
    """
    return [
        (int(xmin * w_ratio), int(ymin * h_ratio), int(xmax * w_ratio), int(ymax * h_ratio))
        for xmin, ymin, xmax, ymax in bboxes
    ]


def flip_bbox(bboxes, img_width, img_height, flip_type="horizontal"):
    """
    Flip bounding boxes horizontally or vertically.
    """
    if flip_type == "horizontal":
        return [(img_width - xmax, ymin, img_width - xmin, ymax) for xmin, ymin, xmax, ymax in bboxes]
    elif flip_type == "vertical":
        return [(xmin, img_height - ymax, xmax, img_height - ymin) for xmin, ymin, xmax, ymax in bboxes]
    else:
        raise ValueError("Invalid flip_type. Use 'horizontal' or 'vertical'")


def custom_transform(img, bboxes, p=0.5):
    # random horizontal flip
    if random.random() > 0.5:
        img = TF.hflip(img)
        width, height = img.size
        bboxes = flip_bbox(bboxes=bboxes, img_width=width, 
                           img_height=height, flip_type="horizontal")
    # random vertical flip
    if random.random() > 0.3:
        img = TF.vflip(img)
        width, height = img.size
        bboxes = flip_bbox(bboxes=bboxes, img_width=width, 
                           img_height=height, flip_type="vertical")
    # # random rotation
    # if random.random() > 0.4:
    #     angle = random.randint(-30, 30)
    #     img = TF.rotate(img, angle)

    return img, bboxes