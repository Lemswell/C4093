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
output_video = os.path.join(output_destination, os.path.splitext(os.path.basename(input_source))[0] + '_output_video.mp4')

# UTILS
def get_distance_between_two_points(a, b, x_y_scaler): # TODO implement 3d value
    # x,y,z
    # 0,1,2
 

    distance = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1]*x_y_scaler - b[1]*x_y_scaler, 2)) # + math.pow(a[2] - b[2], 2))
    # The last part is commented out because moving reference point for z counts as moving the limb with this interpretation.
    return distance

def get_skull(a, b, x_y_scaler): # version of distance formula that involves normalized z axis
    # x,y,z
    # 0,1,2
 

    distance = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1]*x_y_scaler - b[1]*x_y_scaler, 2) + math.pow(a[2] - b[2], 2))
    
    return distance

def find_angle_threepoints(a, b, c): # given three 3d points a, b, c. find the angle between ab and ac.
    # COULD RESTRUCTURE TO FIND VECTOR!

    # get vectors ab and ac
    ab = (b[0] - a[0], a[1] - b[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])

    # get vector magnitude and normal
    abmag = math.sqrt(pow(ab[0], 2) + pow(ab[1], 2) + pow(ab[2], 2))
    abnorm = (ab[0] / abmag, ab[1] / abmag, ab[2] / abmag)
    acmag = math.sqrt(pow(ac[0], 2) + pow(ac[1], 2) + pow(ac[2], 2))
    acnorm = (ac[0] / acmag, ac[1] / acmag, ac[2] / acmag)

    # calculate dot product
    res = abnorm[0] * acnorm[0] + abnorm[1] * acnorm[1] + abnorm[2] * acnorm[2]

    # find angle
    angle = math.acos(res)

    return angle

# CHECK CONTACT POINT HELD
def get_limb_coord(landmarks, x_y_scaler): # finds limb coords and a diameter of where the limb is in
    limbs = []
    # diameter value is given as a reference to see if a limb is still holding onto a hold (this should be made redundant once object detection is introduced)

    diameter = max(get_distance_between_two_points([landmarks[15].x, landmarks[15].y, landmarks[15].z], [landmarks[17].x, landmarks[17].y, landmarks[17].z], x_y_scaler),
                get_distance_between_two_points([landmarks[19].x, landmarks[19].y, landmarks[19].z], [landmarks[17].x, landmarks[17].y, landmarks[17].z], x_y_scaler),
                get_distance_between_two_points([landmarks[15].x, landmarks[15].y, landmarks[15].z], [landmarks[19].x, landmarks[19].y, landmarks[19].z], x_y_scaler))
    l_hand_coords = ['lh', 
                        [(landmarks[15].x + landmarks[17].x + landmarks[19].x) / 3,
                        (landmarks[15].y + landmarks[17].y + landmarks[19].y) / 3,
                        (landmarks[15].z + landmarks[17].z + landmarks[19].z) / 3 ],
                    diameter]
    limbs.append(l_hand_coords)

    diameter = max(get_distance_between_two_points([landmarks[16].x, landmarks[16].y, landmarks[16].z], [landmarks[18].x, landmarks[18].y, landmarks[18].z], x_y_scaler),
                get_distance_between_two_points([landmarks[20].x, landmarks[20].y, landmarks[20].z], [landmarks[18].x, landmarks[18].y, landmarks[18].z], x_y_scaler),
                get_distance_between_two_points([landmarks[16].x, landmarks[16].y, landmarks[16].z], [landmarks[20].x, landmarks[20].y, landmarks[20].z], x_y_scaler))
    r_hand_coords = ['rh', [(landmarks[16].x + landmarks[18].x + landmarks[20].x) / 3,
                            (landmarks[16].y + landmarks[18].y + landmarks[20].y) / 3,
                            (landmarks[16].z + landmarks[18].z + landmarks[20].z) / 3 ],
                        diameter]
    limbs.append(r_hand_coords)

    diameter = max(get_distance_between_two_points([landmarks[27].x, landmarks[27].y, landmarks[27].z], [landmarks[31].x, landmarks[31].y, landmarks[31].z], x_y_scaler),
                get_distance_between_two_points([landmarks[29].x, landmarks[29].y, landmarks[29].z], [landmarks[31].x, landmarks[31].y, landmarks[31].z], x_y_scaler),
                get_distance_between_two_points([landmarks[27].x, landmarks[27].y, landmarks[27].z], [landmarks[29].x, landmarks[29].y, landmarks[29].z], x_y_scaler))
    l_foot_coords = ['lf', [(landmarks[27].x + landmarks[31].x + landmarks[29].x) / 3,
                            (landmarks[27].y + landmarks[31].y + landmarks[29].y) / 3,
                            (landmarks[27].z + landmarks[31].z + landmarks[29].z) / 3 ],
                        diameter]
    limbs.append(l_foot_coords)

    diameter = max(get_distance_between_two_points([landmarks[28].x, landmarks[28].y, landmarks[28].z], [landmarks[30].x, landmarks[30].y, landmarks[30].z], x_y_scaler),
                get_distance_between_two_points([landmarks[32].x, landmarks[32].y, landmarks[32].z], [landmarks[30].x, landmarks[30].y, landmarks[30].z], x_y_scaler),
                get_distance_between_two_points([landmarks[28].x, landmarks[28].y, landmarks[28].z], [landmarks[32].x, landmarks[32].y, landmarks[32].z], x_y_scaler))
    r_foot_coords = ['rf', [(landmarks[28].x + landmarks[32].x + landmarks[30].x) / 3,
                            (landmarks[28].y + landmarks[32].y + landmarks[30].y) / 3,
                            (landmarks[28].z + landmarks[32].z + landmarks[30].z) / 3 ],
                        diameter]
    limbs.append(r_foot_coords)
    
    return limbs

def get_limb_velocity(landmark_history, fps, x_y_scaler): # finds velocity as (portion of image) per second
    # output
    # indicies: lh, rh, lf, rf
    # list[x]: name, coords, dia, velocity, 

    # make sure there are enough frames in landmark history
    limbs_curr = get_limb_coord(landmark_history[-1][2], x_y_scaler)

    skull = get_skull([landmark_history[-1][2][7].x, landmark_history[-1][2][7].y, landmark_history[-1][2][7].z], [landmark_history[-1][2][8].x, landmark_history[-1][2][8].y, landmark_history[-1][2][8].z], x_y_scaler)

    if len(landmark_history) < 2:
        # print("at frame: " + str(landmark_history[-1][0]))
        # print("there are no previous frames with landmarks")
        return limbs_curr
    
    # make sure gap between current and previous frame (containing landmarks) is less than .5 seconds 
    time_between_frames = (landmark_history[-1][0] - landmark_history[-2][0]) / fps
    if time_between_frames > .5:
        # print("at frame: " + str(landmark_history[-1][0]))
        # print("there are no previous frames with landmarks that are shorter than .5 seconds prior")
        return limbs_curr

    # saving actual landmarks of current frame and previous frame
    limbs_prev = get_limb_coord(landmark_history[-2][2], x_y_scaler)
    
    # a structure for saving limb information
    # indicies: lh, rh, lf, rf
    # list[x]: name, 3dcoords, dia, velocity,
    # where   {name, 3dcoords, dia}                      come from get_limb_coord func
    limb_info = []

    for idx, limb in enumerate(limbs_curr): # cycles through indicies (lh, rh, lf, rf)
        # gets distance two of the same limb[idx] at different frames 
        dist = get_distance_between_two_points(limbs_prev[idx][1], limbs_curr[idx][1], x_y_scaler)
        # appends {name, 3dcoords, dia, velocity} to limb info
        limb.append((dist / skull) / time_between_frames)
        limb_info.append(limb)

    return limb_info

def get_CoM_velocity(landmark_history, fps, x_y_scaler):
    if len(landmark_history) < 2:
        return
    
    # make sure gap between current and previous frame (containing landmarks) is less than .5 seconds 
    time_between_frames = (landmark_history[-1][0] - landmark_history[-2][0]) / fps
    if time_between_frames > .5:
        return
    
    skull = get_skull([landmark_history[-1][2][7].x, landmark_history[-1][2][7].y, landmark_history[-1][2][7].z], [landmark_history[-1][2][8].x, landmark_history[-1][2][8].y, landmark_history[-1][2][8].z], x_y_scaler)
    curr_CoM = landmark_history[-1][1][0]
    prev_CoM = landmark_history[-2][1][0]

    dist = get_distance_between_two_points(list(curr_CoM), list(prev_CoM), x_y_scaler)
        # appends {name, 3dcoords, dia, velocity} to limb info
    return (dist / skull) / time_between_frames

def check_limb_held(landmark_history, fps, x_y_scaler): # TODO
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

    
    velocity_lim = 2 # arbitrary value (TODO)
    # make velocity limit proportional to body length


    limb_info = get_limb_velocity(landmark_history, fps, x_y_scaler)
    if len(limb_info[0]) == 3:
        return limb_info

    

    # identify if helf via length
    for index, limb in enumerate(limb_info):
        if len(landmark_history[-2][3][index]) > 4: # check hold status of previous frame is true
            if get_distance_between_two_points(limb[1], landmark_history[-2][3][index][4][0], x_y_scaler) < landmark_history[-2][3][index][4][1]*1.5: # if still within area of previous holding area
                if limb[3] <= landmark_history[-2][3][index][4][2]: # if curr velocity is less than prev recorded velocity when "held"
                    limb.append([limb[1], limb[2], limb[3]])
                    continue
                limb.append(landmark_history[-2][3][index][4])
                continue
        if limb[3] <= velocity_lim:
            limb.append([limb[1], limb[2], limb[3]]) # appends: {coordinates}, radius, velocity

    # returns list of 4 limbs, each limb is either has information in the form of a list that has either 4 or 5 elements
    # if the list has 5 elements then it is held, these extra two elements show an original diameter and location
    # name, coords, dia(xyz), vel, marker(coords, dia, vel) 

    return limb_info

# IDENTIFTING ACTING FORCES
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

def get_weight_distribution(contact_points, CoM): # output: list of contact points, showing [name, coords, force]
    # FINDING ACTUAL DISTRIBUTION: the further away a contact_point is from CoM x and z, the less responsible it is for holding the weight (counteracting the force of gravity). 
        # y value displacement should either have very little to no influence on weight dist (maybe closer horizontally means slightly less weight distributed to it).
        # the weight distribution should be based on the difference between contact point distances from the CoM (comparing CoM-contact_point[0] to CoM-contact_point[1] vectors.

    # problem cases:
        # contact surface: chimneying
        # front lever: should it account for additional force applied during the front lever (where CoM and contact_points are aligned but trunk landmark isn't)

    # get distance between contact_points and CoM (x and z values)
    result = []
    for contact_point in contact_points:
        # distance betwen com and contact_point
        result.append([contact_point[0], contact_point[1], math.sqrt(pow(contact_point[1][0] - CoM[0], 2) + pow(contact_point[1][2] - CoM[2], 2))])

    # [name, [coords], distance]
    # inverses distance values
    for distance in result:
        distance[2] = 1/distance[2]

    # turn inverse distance values to percentage (weight distribution) and add to return value
    sum_of_inverse_distances = 0 
    for dist in result:
        sum_of_inverse_distances += dist[2]
    for inverse_distance in result:
        # convert inverse distance to weight dist
        inverse_distance[2] = inverse_distance[2]/sum_of_inverse_distances

    return result

def get_force_for_contact_point(contact_points, CoM): # output: list of contact points, showing [name, coords, raw distribution, force after angle consideration]
    # taking into consideratio the direction of force applied 
    # ADDING EXTRA FORCE BASED ON FORCE DIRECTION: produce a vector for each point of contact 
        # assume contact points below CoM push weight up by pushing away or CoM, whereas points above pull weight up by pulling towards (trunk or CoM)
        # trunk or CoM depends on if contact point is placed between joining trunk landmark and CoM, if so direction is in relation to CoM, trunk if not.
        # equation: Force * cos(angle from point to trunk) = assined weight distribution

    # problem cases:
        # Friction: imagine scenario where shoulder width square prism volume, to hold oneself up, you'd need to squeeze the volume to make use of the friction. 
        # contact surface: chimneying

    result = get_weight_distribution(contact_points, CoM)

    for index, contact_point in enumerate(result):
        third_point = (contact_point[1][0], CoM[1], contact_point[1][2]) 

        angle = find_angle_threepoints(contact_point[1], CoM, third_point) 
        force = contact_point[2]/abs(math.cos(angle))
        
        result[index].append(force)
    
    return result

def add_torque_to_total_force(contact_points, CoM): # TODO
    # ADD FORCE FROM TORQUE:
        # should be more applicable when contact points are not on opposite sides (x, z) of CoM
        # equation:
    return
        
def add_motion_to_force(contact_points, CoM): # TODO
    # ADD FORCE FOR MOVEMENT:
        # TODO account for velocity considering previous *active* frames
    return

#OUTPUT
def write_landmarks_to_csv(landmarks, frame_number, csv_data, CoM):
    for idx, landmark in enumerate(landmarks):
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])

    # ADD CoM TO CSV
    csv_data.append([frame_number, "CENTER_OF_MASS", CoM[0], CoM[1], CoM[2]])

def write_velocity_to_csv(landmark_history, csv_data):
    # ADD CoM TO CSV
    # csv_data[4].append([landmark_history[-1][0], "CoM"])
    if len(landmark_history[-1][1]) > 1:
        csv_data[4].append([landmark_history[-1][0], "CoM", landmark_history[-1][1][1]])
    for idx, limb in enumerate(landmark_history[-1][3]):
        if len(limb) > 3:
            csv_data[idx].append([landmark_history[-1][0], limb[0], limb[3]]) # , limb[1]]) # [0], limb[1][1], limb[1][2]])
        # if len(limb) > 4:
        #     csv_data[-1].append(limb[4])

def write_force_to_csv(force_record, frame_number, csv_data):

    for limb in force_record:
        if limb[0] == "lh":
            csv_data[0].append([frame_number, limb[0], limb[2], limb[3]]) #, limb[1]]) # [0], limb[1][1], limb[1][2]])
        if limb[0] == "rh":
            csv_data[1].append([frame_number, limb[0], limb[2], limb[3]]) #, limb[1]]) # [0], limb[1][1], limb[1][2]])
        if limb[0] == "lf":
            csv_data[2].append([frame_number, limb[0], limb[2], limb[3]]) #, limb[1]]) # [0], limb[1][1], limb[1][2]])
        if limb[0] == "rf":
            csv_data[3].append([frame_number, limb[0], limb[2], limb[3]]) #, limb[1]]) # [0], limb[1][1], limb[1][2]])

    # ADD CoM TO CSV

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
force_data = [[],[],[],[]]
velocity_data = [[],[],[],[],[]]
landmark_location_data = []
limb_location_data = [] # TODO
landmark_history = []
fps = cap.get(cv2.CAP_PROP_FPS)
x_y_scaler = 0

while cap.isOpened():
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    x_y_scaler = height/width
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
        landmark_history.append([frame_number, [CoM], result.pose_landmarks.landmark, []])  # list struct (frame, CoM, landmarks, limb_info)
        
        CoM_velocity = get_CoM_velocity(landmark_history, fps, x_y_scaler)
        if CoM_velocity:
            landmark_history[-1][1].append(CoM_velocity)
        
        limb_info = check_limb_held(landmark_history, fps, x_y_scaler)
        if limb_info:
            landmark_history[-1][3] = limb_info

        contact_points = []


        # colour held holds
        for limb in limb_info:
            limb_coord = (int(limb[1][0]*width),int(limb[1][1]*height)) 
            if len(limb) > 4:
                frame = cv2.circle(frame, limb_coord, radius=6, color=(0, 255, 255), thickness=-1)
                contact_points.append([limb[0], limb[1]])
                continue
            frame = cv2.circle(frame, limb_coord, radius=6, color=(255, 0, 0), thickness=-1)
        # Add the landmark coordinates to the list and print them

        frame_force_data = get_force_for_contact_point(contact_points, CoM)
        write_force_to_csv(frame_force_data, frame_number, force_data)
        write_velocity_to_csv(landmark_history, velocity_data)
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, landmark_location_data, CoM)

    font = cv2.FONT_HERSHEY_SIMPLEX 
    frame = cv2.putText(frame,  
                str(frame_number),  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 

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


# making csvs
csv_collection_path = os.path.join(output_destination, os.path.splitext(os.path.basename(input_source))[0] + '_csv_collection')
os.mkdir(csv_collection_path)
output_csv_landmark_locations = os.path.join(csv_collection_path , 'lanmark_location.csv')
output_csv_velocities_lh = os.path.join(csv_collection_path, 'velocities_lh.csv')
output_csv_velocities_rh = os.path.join(csv_collection_path, 'velocities_rh.csv')
output_csv_velocities_lf = os.path.join(csv_collection_path, 'velocities_lf.csv')
output_csv_velocities_rf = os.path.join(csv_collection_path, 'velocities_rf.csv')
output_csv_velocities_CoM = os.path.join(csv_collection_path, 'velocities_CoM.csv')

output_csv_force_lh = os.path.join(csv_collection_path , 'force_lh.csv')
output_csv_force_rh = os.path.join(csv_collection_path , 'force_rh.csv')
output_csv_force_lf = os.path.join(csv_collection_path , 'force_lf.csv')
output_csv_force_rf = os.path.join(csv_collection_path , 'force_rf.csv')

with open(output_csv_landmark_locations, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'name', 'x', 'y', 'z'])
    csv_writer.writerows(landmark_location_data)

with open(output_csv_velocities_lh, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'limbname', 'velocity'])
    csv_writer.writerows(velocity_data[0])
with open(output_csv_velocities_rh, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'limbname', 'velocity'])
    csv_writer.writerows(velocity_data[1])
with open(output_csv_velocities_lf, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'limbname', 'velocity'])
    csv_writer.writerows(velocity_data[2])
with open(output_csv_velocities_rf, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'limbname', 'velocity'])
    csv_writer.writerows(velocity_data[3])

with open(output_csv_velocities_CoM, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'limbname', 'velocity'])
    csv_writer.writerows(velocity_data[4])

with open(output_csv_force_lh, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'raw weight dist'])
    csv_writer.writerows(force_data[0])
with open(output_csv_force_rh, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'raw weight dist'])
    csv_writer.writerows(force_data[1])
with open(output_csv_force_lf, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'raw weight dist'])
    csv_writer.writerows(force_data[2])
with open(output_csv_force_rf, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'raw weight dist'])
    csv_writer.writerows(force_data[3])

graph_collection_path = os.path.join(output_destination, os.path.splitext(os.path.basename(input_source))[0]) + '_graph_collection'
os.mkdir(graph_collection_path)
output_plot_velocities_lh = os.path.join(graph_collection_path, 'velocities_lh.png')
output_plot_velocities_rh = os.path.join(graph_collection_path, 'velocities_rh.png')
output_plot_velocities_lf = os.path.join(graph_collection_path, 'velocities_lf.png')
output_plot_velocities_rf = os.path.join(graph_collection_path, 'velocities_rf.png')
output_plot_velocities_CoM = os.path.join(graph_collection_path, 'velocities_CoM.png')

output_plot_d_lh = os.path.join(graph_collection_path , 'distribution_lh.png')
output_plot_d_rh = os.path.join(graph_collection_path , 'distribution_rh.png')
output_plot_d_lf = os.path.join(graph_collection_path , 'distribution_lf.png')
output_plot_d_rf = os.path.join(graph_collection_path , 'distribution_rf.png')
output_plot_force_lh = os.path.join(graph_collection_path , 'force_lh.png')
output_plot_force_rh = os.path.join(graph_collection_path , 'force_rh.png')
output_plot_force_lf = os.path.join(graph_collection_path , 'force_lf.png')
output_plot_force_rf = os.path.join(graph_collection_path , 'force_rf.png')

import matplotlib.pyplot as plt
import numpy as np
import copy

# populate missing frames with no value with np.nan
def populate_missing(list_rangep, list_valuep):
    # if len(list1) < 2 : return list1

    list_range = copy.deepcopy(list_rangep)
    list_value = copy.deepcopy(list_valuep)

    num = list_range[0]
    i = -1

    while num < list_range[-1]:
        num += 1
        i += 1

        if num in list_range: continue

        if i < len(list_range) - 1:
            list_range.insert(i + 1, num)
            list_value.insert(i + 1, np.nan)
        else:
            list_range.append(num) 
            list_value.append(np.nan)
    
    return [list_range, list_value]



plt.plot([column[0] for column in velocity_data[0]], [column[2] for column in velocity_data[0]])
plt.savefig(output_plot_velocities_lh)
plt.clf()
plt.plot([column[0] for column in velocity_data[1]], [column[2] for column in velocity_data[1]])
plt.savefig(output_plot_velocities_rh)
plt.clf()
plt.plot([column[0] for column in velocity_data[2]], [column[2] for column in velocity_data[2]])
plt.savefig(output_plot_velocities_lf)
plt.clf()
plt.plot([column[0] for column in velocity_data[3]], [column[2] for column in velocity_data[3]])
plt.savefig(output_plot_velocities_rf)
plt.clf()
plt.plot([column[0] for column in velocity_data[4]], [column[2] for column in velocity_data[4]])
plt.savefig(output_plot_velocities_CoM)


plt.clf()
data = populate_missing([column[0] for column in force_data[0]], [column[2] for column in force_data[0]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_d_lh)
plt.clf()
data = populate_missing([column[0] for column in force_data[1]], [column[2] for column in force_data[1]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_d_rh)
plt.clf()
data = populate_missing([column[0] for column in force_data[2]], [column[2] for column in force_data[2]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_d_lf)
plt.clf()
data = populate_missing([column[0] for column in force_data[3]], [column[2] for column in force_data[3]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_d_rf)


plt.clf()
data = populate_missing([column[0] for column in force_data[0]], [column[3] for column in force_data[0]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_force_lh)
plt.clf()
data = populate_missing([column[0] for column in force_data[1]], [column[3] for column in force_data[1]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_force_rh)
plt.clf()
data = populate_missing([column[0] for column in force_data[2]], [column[3] for column in force_data[2]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_force_lf)
plt.clf()
data = populate_missing([column[0] for column in force_data[3]], [column[3] for column in force_data[3]])
plt.plot(data[0], data[1])
plt.savefig(output_plot_force_rf)