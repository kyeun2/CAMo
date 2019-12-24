import numpy as np
import cv2, os
import face_recognition

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
specs_ori = cv2.imread('samples/ryan_transparent.png', -1)


known_face_encodings = []
known_face_names = ["ok"]

image_ok = os.listdir('../static/train/')
for i in image_ok : 
    ok_image = face_recognition.load_image_file("../static/train/" + i)
    ok_face_encoding = face_recognition.face_encodings(ok_image)[0]
    known_face_encodings.append(ok_face_encoding)

print(known_face_encodings)
# Create arrays of known face encodings and their names


cap = cv2.VideoCapture(0) #webcame video
# cap = cv2.VideoCapture('jj.mp4') #any Video file also
cap.set(cv2.CAP_PROP_FPS, 30)
 
 
 
def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src
 


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

 
while 1:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_img = small_img[:, :, ::-1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_img)
        face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = "ok"

            face_names.append(name)

    

    faces=face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))
    for (x, y, w, h), name in zip(faces, face_names):
        
    #for (x, y, w, h), name in zip(face_locations, face_names):
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (h + 6, w - 6), font, 1.0, (0, 0, 225), 1)

        if h > 0 and w > 0 and name != "ok":
            glass_symin = int(y)
            glass_symax = int(y+h)
            sh_glass = (glass_symax - glass_symin)
 
            face_glass_roi_color = img[glass_symin:glass_symax, x:x+w]
 
            specs = cv2.resize(specs_ori, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
            transparentOverlay(face_glass_roi_color,specs)
            
    process_this_frame = not process_this_frame

    cv2.imshow('imoji', img)
 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('img.jpg', img)
        break
 
cap.release()
 
cv2.destroyAllWindows()
