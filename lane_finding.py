import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import camera_calibration
import threshold
import glob
import pickle
from tracker import tracker

# find camera calibration matrix
#mtx, dist = camera_calibration.camera_calibration()
#dist_pickle = {}
#dist_pickle["mtx"] = mtx
#dist_pickle["dist"] = dist
#pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb"))

dist_pickle = pickle.load(open("./calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

pimg = cv2.imread('test_images/test0.jpg')
pimg_size = (pimg.shape[1], pimg.shape[0])
bot_width = 0.76
mid_width = 0.08
height_pct = 0.62
bottom_trim = 0.935
src = np.float32([[pimg_size[0]*(.5-mid_width/2), pimg_size[1]*height_pct],
                  [pimg_size[0]*(.5+mid_width/2), pimg_size[1]*height_pct],
                  [pimg_size[0]*(.5+bot_width/2), pimg_size[1]*bottom_trim],
                  [pimg_size[0]*(.5-bot_width/2), pimg_size[1]*bottom_trim],
                  ])
offset = pimg_size[0] * 0.25
dst = np.float32([[offset,0],
                  [pimg_size[0]-offset, 0],
                  [pimg_size[0]-offset, pimg_size[1]],
                  [offset, pimg_size[1]]])
#print (src)
#print(dst)
def get_perspective_transform_matrix(src,dst):
    return cv2.getPerspectiveTransform(src, dst)

def perspective_transform(M, img):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

M = get_perspective_transform_matrix(src,dst)
Minv = get_perspective_transform_matrix(dst,src)

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0, int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

def process_image(img, idx):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    write_name = 'test_images/tracked_' + str(idx) + '.jpg'
    cv2.imwrite(write_name, img)
    img_size = (img.shape[1], img.shape[0])

    vertices = np.array([[(0.05*img.shape[1],img.shape[0]),(0.4*img.shape[1], 0.6*img.shape[0]),
                          (0.6*img.shape[1], 0.6*img.shape[0]), (0.99*img.shape[1],img.shape[0])]], dtype=np.int32)

    combined = threshold.thresholding(img, idx)
    warped = perspective_transform(M, combined)

    #result = road
    write_name = 'output_images/perspective' + str(idx) + '.jpg'
    cv2.imwrite(write_name, warped)

    window_width = 35
    window_height = 80
    curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=35, My_ym = 30/720, My_xm=4/384,
                            Mysmooth_factor = 15)

    window_centroids=curve_centers.find_window_centroids(warped)
    #print(window_centroids)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    rightx = []
    leftx = []

    # go thru each level and draw the windows
    for level in range(0, len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        l_points[(l_points == 1) | (l_mask == 1)] = 255
        r_points[(r_points == 1) | (r_mask == 1)] = 255

    # draw the results
    template = np.array(r_points+l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    # fit the lane boundries to the left, right center positions found
    #yvals = range(0, warped.shape[0])
    y_vals = np.array(range(0, warped.shape[0]))

    res_yvals = np.arange(warped.shape[0]-(window_height/2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*y_vals*y_vals + left_fit[1]*y_vals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*y_vals*y_vals + right_fit[1]*y_vals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2), axis=0),
                             np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),
                             np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
    middle_marker = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),
                             np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)

    #result = road
    cv2.fillPoly(result, [left_lane], color=[255,0,0])
    cv2.fillPoly(result, [right_lane], color=[0,0,255])
    write_name = 'output_images/transformed_' + str(idx) + '.jpg'
    cv2.imwrite(write_name, result)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255,0,0])
    cv2.fillPoly(road, [right_lane], color=[0,0,255])
    cv2.fillPoly(road_bkg, [left_lane], color=[255,255,255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255,255,255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
    curverad = ((1+(2*curve_fit_cr[0]*y_vals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    #calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff, 3))) + 'm '+side_pos+' of center', (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    write_name='output_images/warped_' + str(idx) + '.jpg'
    cv2.imwrite(write_name, result)

# transformation
# Step through the list and search for chessboard corners
images = glob.glob('test_images/test*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    process_image(img, idx)
