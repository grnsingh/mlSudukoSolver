import cv2
import numpy as np
import easyocr


def imgPreprocess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
    return img

def largestContour(contours):
    largest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            possiblePuzzle = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(possiblePuzzle) == 4:
                largest = possiblePuzzle
                max_area = area
    return largest,max_area

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
            
    return boxes

def getPredection(boxes):
    result = []
    for image in boxes:
        
        img = np.asarray(image)
        img = img[10:img.shape[0] - 10, 10:img.shape[1] -10]
        img = cv2.resize(img, (50, 50))

        prediction = reader.readtext(img)
        
        if len(prediction)>0:
            classIndex= [x[1] for x in prediction]
            classIndex=(str(classIndex[0]))
            probabilityValue =[x[2] for x in prediction]
            probabilityValue=float(probabilityValue[0])
        elif len(prediction)==0:
            classIndex=0
            probabilityValue=1

        if classIndex=="{" or classIndex =="}" :
            result.append(7)
        elif classIndex == "y" or classIndex =="Y":
            result.append(4)
        elif probabilityValue > 0:
            result.append((classIndex))
        else:
            result.append(0)
        
            
    return result


def boardArranger(numbers):

    board = np.zeros((9,9),dtype=int)
    k=0
    i=0
    j=0
    while i<9:
      while j<9:
        board[i][j]=numbers[k]
        k=k+1
        j=j+1
      j=0
      i=i+1

    np.set_printoptions(threshold=np.inf)
    
    return board

def boxAuthenticator(board, in0, jn0, guess ):
    for i in range(0,9):
        if board[i][jn0]==guess:
            return False
            
    for j in range(0,9):
        if board[in0][j]==guess:
            return False

    in1 = (in0//3)*3
    jn1 = (jn0//3)*3
 
    for i in range(0,3):
        for j in range(0,3):
            if board[in1+i][jn1+j]==guess:
                return False
    return True


def printBoard(board):
    for i in range(0, 9):
        for j in range(0, 9):
            print(board[i][j], end=" ")
        print()

def boardSolver(board):
    for i in range(0, 9):
        for j in range(0, 9):
            if board[i][j] == 0:
                for val in range(1, 10):
                    if boxAuthenticator(board, i, j, val):
                        board[i][j] = val
                        boardSolver(board)
                        board[i][j] = 0
                return
         
    printBoard(board)

def main():
    imgAddress=input("Enter the suduko file location")
    img=cv2.imread(imgAddress)
    img=cv2.resize(img,(630,630))


    img=imgPreprocess(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

    biggest, maxArea = largestContour(contours)

    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0],[630, 0], [0, 630],[630, 630]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matrix, (630, 630))

    boxes = splitBoxes(img)


    numbers = getPredection(boxes)
    
    board=boardArranger(numbers)
    img=cv2.resize(img,(300,300))
    cv2.imshow(img)
    print("The AI generated suduko board is as follws:")
    printBoard(board)
    userValidation=int(input("Is the suduko board generated correct?\n Enter 1 or 0\n"))
    while userValidation==0:
      address=input("Enter the address of incorrect no. If 6,6 is the location of incorrect no. type 66\nThe Input ranges from 11 to 99  ")
      correctNo=int(input("Enter the correct number\n"))
      board[int(address[0])+1][int(address[1])+1]=correctNo
      print(board)
      userValidation=int(input("Is the suduko board generated correct?"))

    print("The solved SUDUKO is as follows:")
    boardSolver(board)
    








