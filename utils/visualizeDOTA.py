import collections
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10

STANDARD_COLORS = [
'red','orangered','tomato','lightcoral',
'silver','gold','orange','khaki',
'limegreen', 'forestgreen','springgreen','paleturquoise',
'turquoise','dodgerblue','royalblue','slateblue',
'orchid','crimson','mediumvioletred','pink'
]
'''
STANDARD_COLORS = ['silver',
'lightcoral','royalblue','orange',
'limegreen', 'pink','springgreen',
'turquoise','gold',
'orchid'
]
'''
'''
STANDARD_COLORS = ['silver',
'dodgerblue','lightcoral','darkorange',
'forestgreen', 'crimson','mediumspringgreen',
'darkseagreen','gold',
'magenta'
]
'''



#['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
def visualize_boxes(image, boxes, labels, probs, class_labels):

  category_index = {}
  for id_, label_name in enumerate(class_labels):
    category_index[id_] = {"name": label_name}
  image=visualize_boxes_and_labels_on_image_array(image, boxes, labels, probs, category_index)
  return image

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=3000,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=5,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
          and match the keys in the label map.
      scores: a numpy array of shape [N] or None.    If scores=None, then
          this function assumes that the boxes to be plotted are groundtruth
          boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
          category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width] with
          values ranging between 0 and 1, can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
          with values ranging between 0 and 1, can be None.
      use_normalized_coordinates: whether boxes is to be interpreted as
          normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.    If None, draw
          all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
          class-agnostic mode or not.    This mode will display scores but ignore
          classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
          boxes
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection

  Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)

  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]

  sorted_ind = np.argsort(-scores)
  boxes=boxes[sorted_ind]
  scores=scores[sorted_ind]
  classes=classes[sorted_ind]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100 * scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
            classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    xmin, ymin, xmax, ymax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
        image,
        box_to_instance_masks_map[box],
        color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
        image,
        box_to_instance_boundaries_map[box],
        color='red',
        alpha=1.0
      )
    draw_bounding_box_on_image_array(
      image,
      ymin,
      xmin,
      ymax,
      xmax,
      color=color,
      thickness=line_thickness,
      display_str_list=box_to_display_str_map[box],
      use_normalized_coordinates=use_normalized_coordinates)
  return image


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                                          (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
          ymin, xmin, ymax, xmax as relative to the image.    Otherwise treat
          coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                                          (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
          ymin, xmin, ymax, xmax as relative to the image.    Otherwise treat
          coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  '''
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
      [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                        text_bottom)],
      fill=color)
    draw.text(
      (left + margin, text_bottom - text_height - margin),
      display_str,
      fill='black',
      font=font)
    text_bottom -= text_height - 2 * margin
  '''

def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      mask: a uint8 numpy array of shape (img_height, img_height) with
          values between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
      ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
    np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


if __name__ == '__main__':
  import cv2


  import os
  #cateNames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
  # 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court',
  # 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

  #cateNames = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']

  cateNames = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney',
                'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
                'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']

  path = "D:/PyProjects/visdiortest/savep/"
  #path = "D:/PyProjects/visdronetest/savep/"  # 文件夹目录
  files = os.listdir(path)  # 得到文件夹下的所有文件名称
  txts = []
  #pathimg = "D:/Datasets/VisDrone2019/test/input/JPEGImages/"  # 文件夹目录
  #pathimgsave = "D:/PyProjects/visdronetest/vis_res/"  # 文件夹目录

  pathimg = "D:/Datasets/DIOR/JPEGImages/"  # 文件夹目录
  pathimgsave = "D:/PyProjects/visdiortest/vis_res/"  # 文件夹目录

  for file in files:  # 遍历文件夹
    position = path + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
    #print(position)

    positionimg = pathimg + file.replace('txt','jpg')  # 构造绝对路径，"\\"，其中一个'\'为转义符file
    #print(positionimg)
    positionimgsave = pathimgsave + file.replace('txt', 'jpg')  # 构造绝对路径，"\\"，其中一个'\'为转义符file
    imshow = cv2.imread(positionimg)
    b = []
    l = []
    p = []
    with open(position, "r", encoding='utf-8') as f:  # 打开文件

    #with open('G:/test/vis/1.txt', "r") as f:
      for line in f.readlines():
        data = line.split('\t\n')
        for str1 in data:
          sub_str = str1.split(' ')
        if sub_str:
            l.append(cateNames.index(sub_str[0]))
            p.append(float(sub_str[1]))
            aa = []


            aa.append(float(sub_str[2]))
            aa.append(float(sub_str[3]))
            aa.append(float(sub_str[4]))
            aa.append(float(sub_str[5]))


            b.append(aa)

    _boxes = np.array(b)
    _labels = np.array(l)
    _probs = np.array(p)
    #print('lab', _labels)
    #print('bbo', _boxes)
    #print('pro', _probs.shape)

    visualize_boxes(image=imshow, boxes=_boxes, labels=_labels, probs=_probs, class_labels=cateNames)

    #plt.subplot(111)
    #results = imshow[...,::-1]
    #plt.imshow(results)
    #plt.show()
    #plt.savefig('test2png.jpg', dpi=100)
    cv2.imwrite(positionimgsave, imshow, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
    #, [int(cv2.IMWRITE_PNG_COMPRESSION),0]

