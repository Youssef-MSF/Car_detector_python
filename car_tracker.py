import cv2

video = cv2.VideoCapture('cars.mp4')

# Classifier file
classifier_file = 'car_detector.xml'
# create a classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # Operate the frame
        black_n_white_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    cars = car_tracker.detectMultiScale(black_n_white_img)

    for (x, y, w, h) in cars:
        image = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # display the image
    cv2.imshow('Car detector', frame)
    cv2.waitKey(1)
