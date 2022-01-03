import numpy as np
import random
import cv2 as cv
import skimage as sk
from skimage import transform
from skimage import exposure
from skimage.util import random_noise
from scipy import ndimage

"""
Data augmentation methods are adopted from the blog post 
https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
by By Ayoosh Kathuria
"""

""" Horizontal flip """
def horizontal_flip(image, bbox):
	bbox_copy = np.copy(bbox)

	image_center = np.array(image.shape[:2])[::-1]/2
	image_center = np.hstack((image_center, image_center))

	image =  image[:,::-1,:]
	bbox_copy[:,[0,2]] += 2*(image_center[[0,2]] - bbox_copy[:,[0,2]])

	box_w = abs(bbox_copy[:,0] - bbox_copy[:,2])
	    
	bbox_copy[:,0] -= box_w
	bbox_copy[:,2] += box_w
	
	return image, bbox_copy

""" Random horizontal shear """
def random_horizontal_shear(image, bbox):
	bbox_copy = np.copy(bbox)

	angle = random.uniform(-0.16,0.16)
	shear_factor = np.tan(angle)

	w,h = image.shape[1], image.shape[0]

	if shear_factor < 0:
	    image, bbox_copy = horizontal_flip(image, bbox_copy)

	M = np.array([[1, abs(shear_factor), 0],[0,1,0]])

	nW =  image.shape[1] + abs(shear_factor*image.shape[0])

	bbox_copy[:,[0,2]] += ((bbox_copy[:,[1,3]]) * abs(shear_factor) ).astype(int) 


	image = cv.warpAffine(image, M, (int(nW), image.shape[0]))

	if shear_factor < 0:
		image, bbox_copy = horizontal_flip(image, bbox_copy)

	image = cv.resize(image, (w,h))

	scale_factor_x = nW / w

	bbox_copy[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 


	return image, bbox_copy


""" Random rotation """
def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]

    return bbox

def rotate_im(image, angle):
    # grab the dimensions of the image and then determine the centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv.warpAffine(image, M, (nW, nH))

#    image = cv.resize(image, (w,h))
    return image

def get_corners(bbox):
    width = (bbox[:,2] - bbox[:,0]).reshape(-1,1)
    height = (bbox[:,3] - bbox[:,1]).reshape(-1,1)
    
    x1 = bbox[:,0].reshape(-1,1)
    y1 = bbox[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bbox[:,2].reshape(-1,1)
    y4 = bbox[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def rotate_box(corners, angle, cx, cy, h, w):
	corners = corners.reshape(-1,2)
	corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

	M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)


	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cx
	M[1, 2] += (nH / 2) - cy
	# Prepare the vector to be transformed
	calculated = np.dot(M,corners.T).T

	calculated = calculated.reshape(-1,8)

	return calculated

def get_enclosing_box(corners):
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

def random_rotate(image, bbox):
	bbox_copy = np.copy(bbox)

	angle = random.uniform(-0.8,0.8)

	w,h = image.shape[1], image.shape[0]
	cx, cy = w//2, h//2

	image = rotate_im(image, angle)

	corners = get_corners(bbox_copy)

	corners = np.hstack((corners, bbox_copy[:,4:]))


	corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)

	new_bbox = get_enclosing_box(corners)


	scale_factor_x = image.shape[1] / w

	scale_factor_y = image.shape[0] / h

	image = cv.resize(image, (w,h))

	new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 

	bbox_copy  = new_bbox

	bbox_copy = clip_box(bbox_copy, [0, 0, w, h], 0.25)

	return image, bbox_copy


""" All other operations """

def change_contrast(image, bbox, percent_change=(0,15)):
    percent_change = random.uniform(percent_change[0], percent_change[1])
    v_min, v_max = np.percentile(image, (0.+percent_change, 100.-percent_change))
    new_image = exposure.rescale_intensity(image, in_range=(v_min,v_max))
    return new_image, bbox

def gamma_correction(image, bbox, gamma_range=(0.7,1.0)):
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    new_image = exposure.adjust_gamma(image, gamma=gamma, gain=random.uniform(0.8,1.0))
    return new_image, bbox

def blur_image(image, bbox):
    new_image = ndimage.uniform_filter(image, size=(5, 5, 1))
    return new_image, bbox

def add_noise(image, bbox):
    new_image = 255 * random_noise(image)
    return new_image, bbox


""" Random combination of operations """ 
def random_operations(image, bbox):
    avail_operations = {7: random_rotate, 
                        4: horizontal_flip, 
                        6: random_horizontal_shear, 
                        1: change_contrast, 
                        2: gamma_correction, 
                        3: blur_image, 
                        5: add_noise}
    chosen_operations = random.sample(list(avail_operations), random.randrange(1, len(avail_operations)))
    chosen_operations.sort()
    new_image = image
    new_bbox = bbox
    for operation in chosen_operations:
    	if operation == 3:
    		pass
    	else:
        	new_image, new_bbox = avail_operations[operation](new_image, new_bbox)
    return new_image, new_bbox