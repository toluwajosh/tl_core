import os
import numpy as np
import cv2
# import sys
import glob


def parabolize(max_x, full=True):
  # max_x is literally the number of steps we want
  if full:
    steps = 20
    y_val = lambda x: 0.01*(x - 10)**2
  else: 
    steps = 10
    y_val = lambda x: 0.01*x**2
  
  x_vals = np.linspace(0, steps, max_x)
  print("x values: ", x_vals)

  y_vals = []
  for i in range(max_x):
    target_val = int(np.floor(y_val(x_vals[i])*100))
    y_vals.append(target_val)

  return y_vals

def save_frames(frame_names_buffer, frames_buffer, full_action):
  # print("Target values: ", parabolize(len(frames_buffer), full_action))
  target_vals = parabolize(len(frames_buffer), full_action)
  # print("Target values: ", target_vals)
  for i in range(len(frame_names_buffer)):
    image_path = frame_names_buffer[i]+'-'+str(target_vals[i])+'.jpg'
    # cv2.imwrite(image_path, frames_buffer[i])
    print("Image path: ", image_path)

  # print("Frames saved...")





action_id = 'pouring'
save_path = 'data/'+action_id+'/' # su cnn test, kingkan cnn test, liu cnn test

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


targets = []

# loop through video files in directory
# for names in glob.glob("/media/tjosh/vault/datasets/motion_dataset/mix_and_pour/*.avi"):
for names in glob.glob("data/pouring_videos/*.avi"):
  # print(names.split('/'))
  # vid_id = names.split('/')[-1].split('.')[0] # on linux
  vid_id = names.split('\\')[-1].split('.')[0]
  print("Video Name: ",names, "\n")

  # %% loop through frames and annotate:
  cap = cv2.VideoCapture(names)
  cap_is_open = cap.isOpened()
  if not cap_is_open:
    print("Cap is not opened.")
  frame_no = 0
  target_counter = 0
  reached_bound = False
  frames_buffer = []
  frame_names_buffer = []
  full_action = False
  while(cap_is_open):
    frame_no += 1
    target_counter +=1
    start_id = tag_char = end_id = 0

    ret, frame = cap.read()
    if ret==True:

      # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)

      key_in = cv2.waitKey()
      if key_in & 0xFF == ord('s'):
        reached_bound = True
        start_id = '1'
      elif key_in & 0xFF == ord('d'):
        end_id = '1'
        reached_bound = True
      elif key_in & 0xFF == ord('f'):
        pass
      elif key_in & 0xFF == ord('q'):
        print("Quiting!!!")
        cap.release()
        cv2.destroyAllWindows()
        exit()
      elif key_in & 0xFF == ord('v'):
        print("Moving to the next video >>")
        break
      else:
        print("You should type in one of the options.  Exiting!!!")
        request = input("Do you want to exit? [y,n]: ")
        if request == 'n':
          print("Contining")
          continue
        else:
          cap.release()
          cv2.destroyAllWindows()
          exit()
      frame_name =save_path+vid_id+'-'+str(frame_no)+'-'+str(start_id)+'-'+str(end_id)#+'.jpg'7
      frame_names_buffer.append(frame_name)
      frames_buffer.append(frame)
      print("Frame name: ", frame_name)
      print("Target Counter: ", len(frames_buffer))
      print("Full Action? ", full_action)
      if reached_bound == True:
        # save all frames in the buffer
        save_frames(frame_names_buffer, frames_buffer, full_action)

        
        full_action = True
        # reset parameters
        frames_buffer = []
        frame_names_buffer = []
        target_counter = 0
        reached_bound = False
    else: break

  # cap is no more open
  # save all frames in the buffer

cap.release()
cv2.destroyAllWindows()