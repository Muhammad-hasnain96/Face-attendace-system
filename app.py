import base64
import os
from datetime import date
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 10

imgBackground = cv2.imread("background.png")
if imgBackground is None:
    imgBackground = np.zeros((700, 1000, 3), dtype=np.uint8)

frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


def get_date_strings():
    today = date.today()
    return today.strftime("%m_%d_%y"), today.strftime("%d-%B-%Y")


def get_attendance_path(datetoday=None):
    if datetoday is None:
        datetoday, _ = get_date_strings()
    return os.path.join('Attendance', f'Attendance-{datetoday}.csv')


def ensure_attendance_file(datetoday=None):
    path = get_attendance_path(datetoday)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('Name,Roll,Time')
    return path


def get_display_date():
    return get_date_strings()[1]


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
ensure_attendance_file()

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = frontal_cascade.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
        if len(faces) > 0:
            return faces

        faces = profile_cascade.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
        if len(faces) > 0:
            return faces

        flipped = cv2.flip(gray, 1)
        faces = profile_cascade.detectMultiScale(flipped, 1.2, 5, minSize=(40, 40))
        if len(faces) > 0:
            corrected = []
            for (x, y, w, h) in faces:
                corrected.append((gray.shape[1] - x - w, y, w, h))
            return corrected

        return []
    except Exception:
        return []


def decode_base64_image(data):
    try:
        if ',' in data:
            _, encoded = data.split(',', 1)
        else:
            encoded = data
        binary = base64.b64decode(encoded)
        array = np.frombuffer(binary, dtype=np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def identify_face(face_img):
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray, (100, 100))
        return model.predict(resized_face.reshape(1, -1))[0]
    except Exception:
        return None


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        folder = f'static/faces/{user}'
        for imgname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, imgname))
            if img is None:
                continue
            face_points = extract_faces(img)
            if len(face_points) == 0:
                continue
            x, y, w, h = face_points[0]
            face = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray, (100, 100))
            faces.append(resized_face.ravel())
            labels.append(user)
    if len(faces) > 0:
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    csv_path = ensure_attendance_file()
    df = pd.read_csv(csv_path)
    if df.empty:
        return [], [], [], 0
    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    times = df['Time'].tolist()
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    csv_path = ensure_attendance_file()
    df = pd.read_csv(csv_path)
    if userid in df['Roll'].astype(str).tolist():
        row = df[df['Roll'].astype(str) == userid].index[0]
        df.loc[row, 'Time'] = current_time
        df.loc[row, 'Name'] = username
        df.to_csv(csv_path, index=False)
    else:
        with open(csv_path, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=get_display_date())

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(force=True)
    newusername = data.get('newusername', '').strip()
    newuserid = str(data.get('newuserid', '')).strip()
    images = data.get('images', [])

    if not newusername or not newuserid or not images:
        return jsonify(success=False, message='Name, ID, and captured images are required.')

    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)

    saved_count = 0
    for idx, data_url in enumerate(images):
        img = decode_base64_image(data_url)
        if img is None:
            continue
        faces = extract_faces(img)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        face_crop = img[y:y+h, x:x+w]
        filename = os.path.join(userimagefolder, f'{newusername}_{idx}.jpg')
        cv2.imwrite(filename, face_crop)
        saved_count += 1

    if saved_count == 0:
        return jsonify(success=False, message='No valid face detected in captured images. Please try again.')

    train_model()
    return jsonify(success=True, message=f'Registered {saved_count} images for {newusername}_{newuserid}.')


@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json(force=True)
    data_url = data.get('image', '')
    if not data_url:
        return jsonify(success=False, message='Image data is required.')

    img = decode_base64_image(data_url)
    if img is None:
        return jsonify(success=False, message='Could not decode the camera image.')

    faces = extract_faces(img)
    if len(faces) == 0:
        return jsonify(success=False, message='No face detected. Please position your face clearly in front of the camera.')

    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    identified_person = identify_face(face_crop)
    if identified_person is None:
        return jsonify(success=False, message='No trained model found or face not recognized yet.')

    add_attendance(identified_person)
    return jsonify(success=True, message=f'Attendance marked for {identified_person}.')


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=get_display_date(), mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = frame[y:y+h, x:x+w]
            identified_person = identify_face(face)
            if identified_person:
                add_attendance(identified_person)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
                cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=get_display_date())



@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=get_display_date())

if __name__ == '__main__':
    app.run(debug=True)
