import os
import cv2 as cv
import numpy as np

def sharpen(img_before):
    #----------------------
    # return sharpen image
    #----------------------
    rows,cols,channel=img_before.shape
    #img_before = cv.cvtColor(img_before,cv.COLOR_BGR2RGB)
    kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
    after = cv.filter2D(img_before,-1,kernel)
    #cv.imshow("ab",after)
    after = cv.fastNlMeansDenoisingColored(after,None,10,10,7,21)
    return after

def main():
    original_folder_path = "./rgbd_dataset_freiburg1_room/rgb"
    output_directory = "rgb_preprocessing_output"
    start_index = len(original_folder_path)
    for (root, directories, files) in os.walk(original_folder_path):
        for image in files:
            image_path = os.path.join(root,image)
            #print(image_path[start_index+1:-4])
            #
            img = cv.imread(image_path,cv.IMREAD_COLOR)
            #cv.imshow("test",img)
            #cv.waitKey(1)
            dst = sharpen(img)

            save_path = output_directory+'/'+image_path[start_index+1:]
            #print(save_path)
            cv.imwrite(save_path,dst)
            
    return 1

if __name__ == "__main__":
    main()