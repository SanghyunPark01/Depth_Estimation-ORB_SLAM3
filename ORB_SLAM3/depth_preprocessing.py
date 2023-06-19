import numpy as np
import cv2 as cv
import os
import natsort

def additional(img_before):
    #----------------------
    # return sharpen image
    #----------------------
    rows,cols=img_before.shape
    #after = cv.fastNlMeansDenoising(img_before,None,10,7,20)
    after = cv.medianBlur(img_before, 5)
    #img_before = cv.cvtColor(img_before,cv.COLOR_BGR2RGB)
    kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
    #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    after = cv.filter2D(after,-1,kernel)
    #cv.imshow("ab",after)
    
    return after

def main1():
    input_folder_path = "./../ORB_SLAM3/rgb_p_DE/depth_original"
    input_file_list = natsort.natsorted(os.listdir(input_folder_path))

    GT_folder_path = "./../ORB_SLAM3/rgbd_dataset_freiburg1_xyz/depth"
    GT_file_list = natsort.natsorted(os.listdir(GT_folder_path))
    
    output_directory = "./../ORB_SLAM3/rgb_p_DE/depth"

    print("start preprocessing...")
    for file_name,GT_file_name in zip(input_file_list,GT_file_list):
        #print(file_name)
        file_path = input_folder_path + "/" + file_name
        GT_path = GT_folder_path + "/" + GT_file_name
        #img readp
        img = cv.imread(file_path,cv.IMREAD_UNCHANGED)

        GT_img = cv.imread(GT_path,cv.IMREAD_UNCHANGED)
        #img resize
        dst = cv.resize(img, (640, 480), interpolation=cv.INTER_CUBIC)
        #sharpen
        dst = additional(dst)
        #img encoding uint8->uint16 // Because Ground Truth => uint16
        n_max = (np.max(img) + 1) * 256 - 1
        n_min = (np.min(img) + 1) * 256 - 1
        GT_max = np.max(GT_img)
        rate = GT_max/n_max
        n_min_n = int(n_min*rate)
        img = ((img+1)*256-1)*0.6
        #dst_uint16 = cv.normalize(dst, None, n_min, n_max, cv.NORM_MINMAX, dtype=cv.CV_16U)
        dst_uint16 = img.astype(np.uint16)
        
        #rescaling
        
        #print(n_min_n)
        
        
        #print(rescaling_rate)
        #output_img = dst_uint16*rescaling_rate
        #
        # cv.imshow("tset",dst_uint16)
        # cv.imshow("GT",GT_img)
        # cv.waitKey(1)

        #save image
        save_path = output_directory+"/"+GT_file_name
        #print(save_path)
        cv.imwrite(save_path,dst_uint16)
    print("all data preprocessing complete!")
    print("checking")
    
    OutputLength = natsort.natsorted(os.listdir(output_directory))
    if len(input_file_list) != len(OutputLength):
        print("something wrong")
    else:
        print("check finish! Good!")
    return 1

def main2():
    input_folder_path = "./rgb_original_depth_estimation_dataset/depth_original"
    input_file_list = natsort.natsorted(os.listdir(input_folder_path))
    
    output_directory = "./rgb_original_depth_estimation_dataset/depth"

    for file_name in input_file_list:
        #print(file_name)
        file_path = input_folder_path + "/" + file_name
        
        #img read
        img = cv.imread(file_path,cv.IMREAD_UNCHANGED)

        #img resize
        dst = cv.resize(img, (640, 480), interpolation=cv.INTER_CUBIC)
        #save image
        save_path = output_directory+"/"+file_name
        #print(save_path)
        cv.imwrite(save_path,dst)

    return 1

if __name__ == "__main__":
    main1()
    