import os
import numpy as np
import cv2
import sys
import glob

# toolbar_width = 40

# # setup toolbar
# sys.stdout.write("[%s]" % (" " * toolbar_width))
# sys.stdout.flush()
# sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

# vid_path = 'videos/new_tests/me cnn test.avi' # su cnn test, kingkan cnn test, liu cnn test
# vid_name = 'videos/'+vid_path.split('/')[-1].split('.')[0]

# # create new path if it doesnt exist
# if not os.path.exists(vid_name):
#   os.makedirs(vid_name)

action_id = 'pouring'
save_path = 'data/'+action_id+'/' # su cnn test, kingkan cnn test, liu cnn test
# vid_name = 'videos/'+vid_path.split('/')[-1].split('.')[0]

# create new path if it doesnt exist
if not os.path.exists(save_path):
  os.makedirs(save_path)

print("\n###################################################\n")
print("Annotate frames by entering tags for each frame:")
print("Enter 's' for start of an action,")
# print("Enter 't' action in progress, and+")
print("Enter 'e' for the end of an action.")
print("Enter 'q' to quit.")
print("\n###################################################\n")



# loop through video files in directory
# for names in glob.glob("/media/tjosh/vault/datasets/motion_dataset/mix_and_pour/*.avi"):
for names in glob.glob("data/pouring_videos/*.avi"):
  # print(names.split('/'))
  # vid_id = names.split('/')[-1].split('-')[0  ]
  vid_id = names.split('/')[-1].split('.')[0]
  print("Video Name: ",names)



  # %% loop through videos and annotate:
  # print('saving in: ', vid_name, '....')
  cap = cv2.VideoCapture(names)
  cap_is_open = cap.isOpened()
  if not cap_is_open:
    print("Cap is not opened.")
  frame_no = 0
  while(cap_is_open):
    frame_no += 1

    ret, frame = cap.read()
    if ret==True:

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',gray)

      tag_char = raw_input("Enter a tag for the frame")

      if tag_char == 's':
        start_id = '1'
      elif tag_char == 'e':
        end_id = '1'
      elif tag_char == 'q':
        break
      else:
        start_id = end_id = '0'

      # write frame name:
      # cv2.imwrite(vid_name+'/'+str(frame_no)+'.jpg', frame)
      frame_name =save_path+vid_id+'/'+str(frame_no)+'/'+str(start_id)+'/'+str(end_id)+'.jpg' 
      # cv2.imwrite(vid_name+'/'+str(frame_no)+'.jpg', frame)
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #     break
    else: break

    # # update the bar
    # sys.stdout.write("-")
    # sys.stdout.flush()


  cap.release()
  cv2.destroyAllWindows()