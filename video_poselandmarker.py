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
        print("removing folder: " + output_destination)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
os.mkdir(output_destination)
output_csv = os.path.join(output_destination, os.path.splitext(os.path.basename(input_source))[0] + '.csv')
output_video = os.path.join(output_destination, os.path.splitext(os.path.basename(input_source))[0] + '_landmarks.mp4')

def check_limb_held():
    # how do I get contact points? 
    # stillness aproximates held
    # how to check for stillness?
    # check for only hands and feet (only hands and feet will be called limbs)
    # check each limb for periods of time with low velocity
    # for each frame where acceleration is below certain threshold (threshold tbd)
    # record if average loc remains in a certain radius of this loc (defined by landmarks) (if yes, then this is held period, if no then not held)


    # get velocity per limb (imageunit/frame)
    # find low points of velocity
    # test these points with testing method for stillness refered to above




    return

def get_limb_velocity(landmark_history, fps):
    
    
    # output
    # indicies: lh, rh, lf, rf
    # list[x]: x, y, z, dia, velocity

    if len(landmark_history) < 2:
        return
    
    if (landmark_history[-1][0] - landmark_history[-2][0]) > fps/2:
        return

    persec = (landmark_history[-1][0] - landmark_history[-2][0]) / fps

    limbs_prev = get_limb_coord(landmark_history[-2][2])
    limbs_curr = get_limb_coord(landmark_history[-1][2])
    limb_info = []

    for idx, limb in enumerate(limbs_curr):
        dist = get_distance_between_two_points(limbs_prev[idx], limbs_curr[idx])
        
        limb_info.append(limb.append(dist / persec))
    
    return limb_info

def get_distance_between_two_points(a, b):
    # x,y,z
    # 0,1,2

    distance = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2))
    
    return distance

def get_limb_coord(landmarks): 
    limbs = []

    max = max(get_distance_between_two_points([landmarks[15].x, landmarks[15].y, landmarks[15].z], [landmarks[17].x, landmarks[17].y, landmarks[17].z]),
                get_distance_between_two_points([landmarks[19].x, landmarks[19].y, landmarks[19].z], [landmarks[17].x, landmarks[17].y, landmarks[17].z]),
                get_distance_between_two_points([landmarks[15].x, landmarks[15].y, landmarks[15].z], [landmarks[19].x, landmarks[19].y, landmarks[19].z]))
    l_hand_coords = [(landmarks[15].x + landmarks[17].x + landmarks[19].x) / 3,
                         (landmarks[15].y + landmarks[17].y + landmarks[19].y) / 3,
                         (landmarks[15].z + landmarks[17].z + landmarks[19].z) / 3,
                         max]
    limbs.append(l_hand_coords)

    max = max(get_distance_between_two_points([landmarks[16].x, landmarks[16].y, landmarks[16].z], [landmarks[18].x, landmarks[18].y, landmarks[18].z]),
                get_distance_between_two_points([landmarks[20].x, landmarks[20].y, landmarks[20].z], [landmarks[18].x, landmarks[18].y, landmarks[18].z]),
                get_distance_between_two_points([landmarks[16].x, landmarks[16].y, landmarks[16].z], [landmarks[20].x, landmarks[20].y, landmarks[20].z]))
    r_hand_coords = [(landmarks[16].x + landmarks[18].x + landmarks[20].x) / 3,
                         (landmarks[16].y + landmarks[18].y + landmarks[20].y) / 3,
                         (landmarks[16].z + landmarks[18].z + landmarks[20].z) / 3,
                         max]
    limbs.append(r_hand_coords)

    max = max(get_distance_between_two_points([landmarks[27].x, landmarks[27].y, landmarks[27].z], [landmarks[31].x, landmarks[31].y, landmarks[31].z]),
                get_distance_between_two_points([landmarks[29].x, landmarks[29].y, landmarks[29].z], [landmarks[31].x, landmarks[31].y, landmarks[31].z]),
                get_distance_between_two_points([landmarks[27].x, landmarks[27].y, landmarks[27].z], [landmarks[29].x, landmarks[29].y, landmarks[29].z]))
    l_foot_coords = [(landmarks[27].x + landmarks[31].x + landmarks[29].x) / 3,
                         (landmarks[27].y + landmarks[31].y + landmarks[29].y) / 3,
                         (landmarks[27].z + landmarks[31].z + landmarks[29].z) / 3,
                         max]
    limbs.append(l_foot_coords)

    max = max(get_distance_between_two_points([landmarks[28].x, landmarks[28].y, landmarks[28].z], [landmarks[30].x, landmarks[30].y, landmarks[30].z]),
                get_distance_between_two_points([landmarks[32].x, landmarks[32].y, landmarks[32].z], [landmarks[30].x, landmarks[30].y, landmarks[30].z]),
                get_distance_between_two_points([landmarks[28].x, landmarks[28].y, landmarks[28].z], [landmarks[32].x, landmarks[32].y, landmarks[32].z]))
    r_foot_coords = [(landmarks[28].x + landmarks[32].x + landmarks[30].x) / 3,
                         (landmarks[28].y + landmarks[32].y + landmarks[30].y) / 3,
                         (landmarks[28].z + landmarks[32].z + landmarks[30].z) / 3,
                         max]
    limbs.append(r_foot_coords)
    
    return limbs

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

    # get distance between contact_points and CoM (x and z values)
    distance_values = []
    for contact_point in contact_points:
        # distance betwen com and contact_point
        distance = math.sqrt(pow(contact_point[0] - CoM[0], 2) + pow(contact_point[2] - CoM[2], 2))
        distance_values.append(distance)

    # inverses distance values
    for distance in distance_values:
        distance = 1/distance

    # turn inverse distance values to percentage (weight distribution) and add to return value
    contact_point_weight_distribution = []
    sum_of_inverse_distances = sum(distance_values)
    for inverse_distance in distance_values:
        weight_responsiblity = inverse_distance/sum_of_inverse_distances
        contact_point_weight_distribution.append(weight_responsiblity)

    return contact_point_weight_distribution

def find_angle_threepoints(a, b, c): # given three 3d points a, b, c. find the angle between ab and ac.
    # COULD RESTRUCTURE TO FIND VECTOR!

    # get vectors ab and ac
    ab = (b[0] - a[0], a[1] - b[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])

    # get vector magnitude and normal
    abmag = math.sqrt(pow(ab[0], 2) + pow(ab[1], 2) + pow(ab[3], 2))
    abnorm = (ab[0] / abmag, ab[1] / abmag, ab[2] / abmag)
    acmag = math.sqrt(pow(ac[0], 2) + pow(ac[1], 2) + pow(ac[3], 2))
    acnorm = (ac[0] / acmag, ac[1] / acmag, ac[2] / acmag)

    # calculate dot product
    res = abnorm[0] * acnorm[0] + abnorm[1] * acnorm[1] + abnorm[2] * acnorm[2]

    # find angle
    angle = math.acos(res)

    return angle

def get_force_for_contact_point(contact_points, CoM): # does not include force that accelerates 
    # ADDING EXTRA FORCE BASED ON FORCE DIRECTION: produce a vector for each point of contact 
        # assume contact points below CoM push weight up by pushing away or CoM, whereas points above pull weight up by pulling towards (trunk or CoM)
        # trunk or CoM depends on if contact point is placed between joining trunk landmark and CoM, if so direction is in relation to CoM, trunk if not.
        # equation: Force * cos(angle from point to trunk) = assined weight distribution

    # problem cases:
        # Friction: imagine scenario where shoulder width square prism volume, to hold oneself up, you'd need to squeeze the volume to make use of the friction. 
        # contact surface: chimneying

    weight_dist = get_weight_distribution(contact_points)
    
    result = []

    for index, contact_point in enumerate(contact_points):
        third_point = (contact_point[0], CoM[1], contact_point[2]) 
        angle = find_angle_threepoints(contact_point[:3], CoM, third_point)
        force = weight_dist[index]/math.cos[angle]
        result.append(force)
    
    return result

def add_torque_to_total_force(contact_points, CoM):
    # ADD FORCE FROM TORQUE:
        # should be more applicable when contact points are not on opposite sides (x, z) of CoM
        # equation:
    return 0
        
def add_motion_to_force(contact_points, CoM):
    # ADD FORCE FOR MOVEMENT:
        # TODO account for velocity considering previous *active* frames
    return 0

def write_landmarks_to_csv(landmarks, frame_number, csv_data, CoM):
    print(f"Landmark coordinates for frame {frame_number}:")
    # print(landmarks)
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])

    # ADD CoM TO CSV
    print(f"CENTER_OF_MASS: (x: {CoM[0]}, y: {CoM[1]}, z: {CoM[2]})")
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
# csv_data = []
landmark_history = []
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        # Add the landmark coordinates to image
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS) # https://developers.google.com/mediapipe/api/solutions/js/tasks-vision.drawingutils
        # Add the CoM to to image
        CoM = get_com_from_landmarks(result.pose_landmarks.landmark)
        CoM_coord = (int(CoM[0]*width),int(CoM[1]*height))        
        frame = cv2.circle(frame, CoM_coord, radius=4, color=(0, 255, 0), thickness=-1)
        # add to landmark_history
        landmark_history.append([frame_number, CoM, result.pose_landmarks.landmark])
        limb_velocity = get_limb_velocity(landmark_history, fps)
        if limb_velocity:
            landmark_history[-1].append(limb_velocity)
        # Add the landmark coordinates to the list and print them
        # write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data, CoM)

    # Display the frame
    window_resized = cv2.resize(frame, (int(width/2.3), int(height/2.3))) # Resize image
    cv2.imshow('MediaPipe Pose', window_resized)
    
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
    # csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
    # csv_writer.writerows(csv_data)
    csv_writer.writerows(landmark_history)