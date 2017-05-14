import cv2
import sys
import time

class MirrorState:
    def __init__(self, openness=0.0, open_time=5.0, close_time=0.5):
        self.openness = openness
        self.close_time = close_time
        self.open_time = open_time
        self.last_update = time.time()

    def Draw(self):
        print "\x1b[2J\x1b[H"
        res  = '|'
        for i in xrange(int(100 * self.openness)):
            res += '-'
        res += '|'
        for i in xrange(100 - int(100 * self.openness)):
            res += '-'
        res += '||'
        for i in xrange(100 - int(100 * self.openness)):
            res += '-'
        res += '|'
        for i in xrange(int(100 * self.openness)):
            res += '-'
        res += '|'
        print res

    def Update(self, do_open):
        time_amt = time.time() - self.last_update
        self.last_update = time.time()
        if do_open:
            d_openness = time_amt / self.open_time
        else:
            d_openness = - time_amt / self.close_time
        self.openness = max(min(self.openness + d_openness, 1.0), 0)
            
        

class SquareState:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.last_seen = time.time()
        self.first_seen = time.time()

    def Update(self, other):
        if self.x1 > other.x2 or other.x1 > self.x2 or self.y1 > other.y2 or other.y1 > self.y2:
            return False
        else:
            self.x1 = max(self.x1, other.x1)
            self.y1 = max(self.y1, other.y1)
            self.x2 = min(self.x2, other.x2)
            self.y2 = min(self.y2, other.y2)
            self.last_seen = other.last_seen
            return True
            
class State:
    def __init__(self, appear_time=1.0, disapear_time=1.0):
        cascPath = "Webcam-Face-Detect/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)        
        self.video_capture = cv2.VideoCapture(0)
        
        self.appear_time = appear_time
        self.disapear_time = disapear_time
        self.squares = []

        self.mirror_state = MirrorState()

    def Update(self, other):
        found = False
        for s in self.squares:
            if s.Update(other):
                found = True
        if not found:
            self.squares.append(other)

    def Prune(self):
        self.squares = [s for s in self.squares if time.time() - s.last_seen < self.disapear_time]

    def VisibleSquares(self):
        for s in self.squares:
            if time.time() - s.first_seen > self.appear_time:
                yield s
        
    def UpdateMirror(self):
        self.mirror_state.Update(len([s for s in self.VisibleSquares()]) == 0)
        self.mirror_state.Draw()
                
        
def CamIteration(state):
    # Capture frame-by-frame
    ret, frame = state.video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = state.faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectan[gle around the faces
    for (x, y, w, h) in faces:
        s = SquareState(x, y, x  + w, y + h)
        state.Update(s)

    state.Prune()

    state.UpdateMirror()
        
    for s in state.VisibleSquares():
        cv2.rectangle(frame, (s.x1, s.y1), (s.x2, s.x2), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True


if __name__ == '__main__':
    state = State()

    while True:
        CamIteration(state)

    # When everything is done, release the capture
    state.video_capture.release()
    cv2.destroyAllWindows()

