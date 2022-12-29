import numpy as np
import cv2
import time
import datetime
import winsound


def beap():
    # Set frequency to 2000 Hertz
    frequency = 2000
    # Set duration to 1500 milliseconds (1.5 seconds)
    duration = 1500
    # Make beep sound on Windows
    winsound.Beep(frequency, duration)




# Note the starting time
start_time = time.time()

# Initialize these variables for calculating FPS
fps = 0
frame_counter = 0
person_counter = 0
thresh = 1500
# Read the video steram from the camera

cap = cv2.VideoCapture(0)

# Create the background subtractor object
foog = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=200, history=5000)

while (1):

    ret, frame = cap.read()
    if not ret:
        break


    # Apply the background object on each frame
    fgmask = foog.apply(frame)

    # Get rid of the shadows
    ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)


    fgmask = cv2.erode(fgmask, kernel=None, iterations=1)

    # Detect contours in the frame
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if person_counter>=10:
        cv2.imwrite('intruder.jpg',frame)

    if contours:
        # Get the maximum contour
        cnt = max(contours, key=cv2.contourArea)

        # make sure the contour area is somewhat hihger than some threshold to make sure its a person and not some noise.
        if cv2.contourArea(cnt)>thresh:
            print(cv2.contourArea(cnt))

            # We use person counter to identify the movement of person it will also help us to exclude False positive
            person_counter = person_counter + 1
            print(person_counter)

            # Draw a bounding box around the person and label it as person detected
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Intruder Detecded', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)
            beap()

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (500, 400), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                (255, 40, 155), 2)
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))

    # Stack both frames and show the image
    current_time = datetime.datetime.now().strftime("%A, %I:%M:%S %p %d %B %Y")
    cv2.putText(frame,current_time,(100,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),1)
    fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((fgmask_3, frame))
    cv2.imshow('Combined', cv2.resize(stacked, None, fx=0.65, fy=0.65))


    cv2.waitKey(1)



    if 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
