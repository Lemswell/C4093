import cv2
import mediapipe as mp
import csv
import math
import sys
import os
import shutil
# import argparse # import for more comprehensive usage of script

# input path as argument
input_source = sys.argv[1]
if(not os.path.exists(input_source)):
    print("argument is not a path")
    sys.exit()
# define output folder rather than individual files
output_destination = os.path.splitext(input_source)[0] + '_test' # name of output folder
if(os.path.exists(output_destination)): # replace dst dir if already exists
    try:
        shutil.rmtree(output_destination)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
os.mkdir(output_destination)
output_csv = os.path.join(output_destination, os.path.splitext(os.path.basename(input_source))[0] + '.csv')
output_video = os.path.join(output_destination, os.path.splitext(os.path.basename(input_source))[0] + '_landmarks.mp4')

def get_com_from_landmarks(landmarks): # output: tuple (prob shoulda been list but ehh)
    # CENTER OF MASS IS GIVEN BY: HEAD .0681 + TRUNK .4302 + ARMS 2*(UPPERARM .0263 + FOREARM .015 + HAND .00585) + LEGS 2*(THIGH .1447 + SHANK .0457 + FOOT .0133)

    com = (0.0, 0.0, 0.0)
    
    head_coords = ((landmarks[8].x + landmarks[7].x) / 2,
                       (landmarks[8].y + landmarks[7].y) / 2,
                       (landmarks[8].z + landmarks[7].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.0681*x for x in head_coords])))
    
    trunk_coords = ((landmarks[11].x + landmarks[12].x + landmarks[23].x + landmarks[24].x) / 4,
                        (landmarks[11].y + landmarks[12].y + landmarks[23].y + landmarks[24].y) / 4,
                        (landmarks[11].z + landmarks[12].z + landmarks[23].z + landmarks[24].z) / 4)
    com = tuple(map(lambda x, y: x + y, com, tuple([.4302*x for x in trunk_coords])))
    
    l_upperarm_coords = ((landmarks[11].x + landmarks[13].x) / 2,
                         (landmarks[11].y + landmarks[13].y) / 2,
                         (landmarks[11].z + landmarks[13].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.0263*x for x in l_upperarm_coords])))
    
    l_forearm_coords = ((landmarks[15].x + landmarks[13].x) / 2,
                         (landmarks[15].y + landmarks[13].y) / 2,
                         (landmarks[15].z + landmarks[13].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.015*x for x in l_forearm_coords])))
    
    l_hand_coords = ((landmarks[15].x + landmarks[17].x + landmarks[19].x) / 3,
                         (landmarks[15].y + landmarks[17].y + landmarks[19].y) / 3,
                         (landmarks[15].z + landmarks[17].z + landmarks[19].z) / 3)
    com = tuple(map(lambda x, y: x + y, com, tuple([.00585*x for x in l_hand_coords])))
    
    r_upperarm_coords = ((landmarks[12].x + landmarks[14].x) / 2,
                         (landmarks[12].y + landmarks[14].y) / 2,
                         (landmarks[12].z + landmarks[14].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.0263*x for x in r_upperarm_coords])))
    
    r_forearm_coords = ((landmarks[14].x + landmarks[16].x) / 2,
                         (landmarks[14].y + landmarks[16].y) / 2,
                         (landmarks[14].z + landmarks[16].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.015*x for x in r_forearm_coords])))
    
    r_hand_coords = ((landmarks[16].x + landmarks[18].x + landmarks[20].x) / 3,
                         (landmarks[16].y + landmarks[18].y + landmarks[20].y) / 3,
                         (landmarks[16].z + landmarks[18].z + landmarks[20].z) / 3)
    com = tuple(map(lambda x, y: x + y, com, tuple([.00585*x for x in r_hand_coords])))
    
    l_thigh_coords = ((landmarks[25].x + landmarks[23].x) / 2,
                         (landmarks[25].y + landmarks[23].y) / 2,
                         (landmarks[25].z + landmarks[23].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.1447*x for x in l_thigh_coords])))
    
    l_shank_coords = ((landmarks[25].x + landmarks[27].x) / 2,
                         (landmarks[25].y + landmarks[27].y) / 2,
                         (landmarks[25].z + landmarks[27].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.0457*x for x in l_shank_coords])))
    
    l_foot_coords = ((landmarks[27].x + landmarks[31].x + landmarks[29].x) / 3,
                         (landmarks[27].y + landmarks[31].y + landmarks[29].y) / 3,
                         (landmarks[27].z + landmarks[31].z + landmarks[29].z) / 3)
    com = tuple(map(lambda x, y: x + y, com, tuple([.0133*x for x in l_foot_coords])))
    
    r_thigh_coords = ((landmarks[24].x + landmarks[26].x) / 2,
                         (landmarks[24].y + landmarks[26].y) / 2,
                         (landmarks[24].z + landmarks[26].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.1447*x for x in r_thigh_coords])))
    
    r_shank_coords = ((landmarks[26].x + landmarks[28].x) / 2,
                         (landmarks[26].y + landmarks[28].y) / 2,
                         (landmarks[26].z + landmarks[28].z) / 2)
    com = tuple(map(lambda x, y: x + y, com, tuple([.0457*x for x in r_shank_coords])))
    
    r_foot_coords = ((landmarks[28].x + landmarks[32].x + landmarks[30].x) / 3,
                         (landmarks[28].y + landmarks[32].y + landmarks[30].y) / 3,
                         (landmarks[28].z + landmarks[32].z + landmarks[30].z) / 3)
    com = tuple(map(lambda x, y: x + y, com, tuple([.0133*x for x in r_foot_coords])))
    
    return com

def get_weight_distribution(contact_points, CoM): # placeholder weight 1 is given as to a still subject.
    # FINDING ACTUAL DISTRIBUTION: the further away a contact_point is from CoM x and z, the less responsible it is for holding the weight (counteracting the force of gravity). 
        # y value displacement should either have very little to no influence on weight dist (maybe closer horizontally means slightly less weight distributed to it).
        # the weight distribution should be based on the difference between contact point distances from the CoM (comparing CoM-contact_point[0] to CoM-contact_point[1] vectors.

    # problem cases:
        # contact surface: chimneying
        # front lever: should it account for additional force applied during the front lever (where CoM and contact_points are aligned but trunk landmark isn't)

    # ASSUME contact_points is list of list
    result = contact_points

    # get distance between contact_points and CoM (x and z values)
    distance_values = []
    for contact_point in result:
        # distance betwen com and contact_point
        distance = math.sqrt(pow(contact_point[0] - CoM[0], 2) + pow(contact_point[2] - CoM[2], 2))
        distance_values.append(distance)

    # inverse distance values
    for distance in distance_values:
        distance = 1/distance

    # turn inverse distance values to percentage (weight distribution)
    total_distances = sum(distance_values)
    for index, distance in enumerate(distance_values):
        distance_values[index] = distance/total_distances
        result[index].append(distance_values[index])

    return result

def contact_point_to_force_vec(contact_points, CoM):
    # ADDING EXTRA FORCE BASED ON FORCE DIRECTION: produce a vector for each point of contact 
            # assume contact points below CoM push weight up by pushing away from either trunk (11,12,23,24) or CoM, whereas points above pull weight up by pulling towards (trunk or CoM)
            # trunk or CoM depends on if contact point is placed between joining trunk landmark and CoM, if so direction is in relation to CoM, trunk if not.
            # equation: Force * cos(angle from point to trunk) = assined weight distribution

    # problem cases:
        # Friction: imagine scenario where shoulder width square prism volume, to hold oneself up, you'd need to squeeze the volume to make use of the friction. 
        # contact surface: chimneying
    return 0

def add_torque_to_total_force(contact_points, CoM):
    # ADD FORCE FROM TORQUE:
        # should be more applicable when contact points are not on opposite sides (x, z) of CoM
        # equation:
    return 0
        
def add_motion_to_force(contact_points, CoM):
    # ADD FORCE FOR MOVEMENT:
        # TODO account for velocity considering previous *active* frames
    return 0

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    # print(landmarks)
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])

    # ADD CoM TO CSV
    CoM = get_com_from_landmarks(landmarks)
    print(f"{"CENTER_OF_MASS"}: (x: {CoM[0]}, y: {CoM[1]}, z: {CoM[2]})")
    csv_data.append([frame_number, "CENTER_OF_MASS", CoM[0], CoM[1], CoM[2]])

    print("\n")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(input_source)

# Save video data for saving vid
if cap.isOpened():
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    frameSize = (int(width), int(height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
video_out = cv2.VideoWriter(output_video, fourcc, fps, frameSize)

# data for csv
frame_number = 0
csv_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS) # https://developers.google.com/mediapipe/api/solutions/js/tasks-vision.drawingutils
        # TODO add visualisation of CoM
        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data)

    # Display the frame
    # cv2.imshow('MediaPipe Pose', frame)
    
    # save frame to exported video
    video_out.write(frame)
    
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1
    
video_out.release()
cap.release()
cv2.destroyAllWindows()


with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
    csv_writer.writerows(csv_data)