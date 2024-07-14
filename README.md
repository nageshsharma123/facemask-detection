import darknet
import cv2
import os
#import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
from google.colab.patches import cv2_imshow


def darknet_helper(img, width, height):
   darknet_image = darknet.make_image(width, height, 3)
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img_resized = cv2.resize(img_rgb, (width, height),interpolation = cv2.INTER_LINEAR)
   img_height, img_width, _ = img.shape
   width_ratio = img_width / width
   height_ratio = img_height / height
   darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
   detections = darknet.detect_image(network, class_names, darknet_image)
   darknet.free_image(darknet_image)
   return detections, width_ratio, height_ratio


if __name__ == "__main__":
   network, class_names, class_colors = darknet.load_network("/content/drive/MyDrive/England/project_mask/yolov4-tiny.cfg", "/content/drive/MyDrive/England/project_mask/obj.data", "/content/drive/MyDrive/England/project_mask/yolov4-tiny_best.weights")
   width = darknet.network_width(network)
   height = darknet.network_height(network)
   
   #font = cv2.FONT_HERSHEY_SIMPLEX
   #fourcc = cv2.VideoWriter_fourcc(*'XVID')
   #loc= "/home/zestiot/darknet/vizag_mar28/April_13_main_entry_16_o.avi"
   #out=cv2.VideoWriter( loc, fourcc, 15, (1280,720), True)
   
   vidObj = cv2.VideoCapture('face_mask.mp4')
   success = 1
   while success:
      success, img = vidObj.read()
      #img = cv2.imread('03.jpg')
      #print("Running")
      image = cv2.resize(img,(1280,720))
      image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      detections, width_ratio, height_ratio = darknet_helper(image, width, height)
      #print(detections)
      if len(detections) >= 0:
         for i1,j1 in enumerate(detections):
            y_res, x_res = image.shape[:2]
            cord1= j1[2]
            conf1 = j1[1]
            cl_name1 = j1[0]
            xco = int(float(cord1[0] - float(cord1[2] / 2)) * float(x_res / 608.0))
            yco = int(float(cord1[1] - float(cord1[3] / 2)) * float(y_res / 608.0))
            x_ext = int(float(cord1[2]) * float(x_res / 608.0))
            y_ext = int(float(cord1[3]) * float(y_res / 608.0))
            #print('name',cl_name1)

            if cl_name1 == 'Face':
               mask_check = 0
               for i,j in enumerate(detections):
                  cord= j[2]
                  conf = j[1]
                  cl_name = j[0]
                  #print('name2',cl_name)
                  if cl_name == 'Mask':
                     mask_check = 1
                     xm=int((cord[0]) * float(x_res/608.0)) 
                     ym=int((cord[1]) * float(y_res/608.0))
                     if (xco < xm < (xco+x_ext)) and (yco < ym < (yco+y_ext)) :
                        image=cv2.rectangle(image,(xco,yco),(xco+x_ext,yco+y_ext),(0,0,255),5)
                        #print('saving')
                        cv2.putText(image,"Mask",(xco-20, yco-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,255), 2)
               if mask_check == 0:
                  image=cv2.rectangle(image,(xco,yco),(xco+x_ext,yco+y_ext),(0,0,255),5)
                  cv2.putText(image,"No Mask",(xco-20, yco-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,255), 2)
      cv2_imshow(image)
      #cv2.imwrite("result_img6.jpg",image)
      #if cv2.waitKey(25) & 0xFF == ord('q'):
      #   break
      
      #img = plt.imshow(image)
      #plt.pause(0.5)
   cap.release()
