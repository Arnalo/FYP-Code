# Import the necessary libraries
import os
import sys
import cv2
import warnings
import numpy as np
import pandas as pd
import customtkinter
import tensorflow as tf

warnings.filterwarnings('ignore')

from PIL import ImageTk, Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
from customtkinter import filedialog as fd
from keras.utils import to_categorical

############################################################################################################################################

# Define ShakeShake function as custom object
class ShakeShake(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        if training:
            alpha = K.random_uniform(shape=[tf.shape(inputs[0])[0], 1, 1, 1])
            beta = K.random_uniform(shape=[tf.shape(inputs[0])[0], 1, 1, 1])
        else:
            alpha = 0.5
            beta = 0.5
        return inputs[0] * alpha + inputs[1] * beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = super(ShakeShake, self).get_config()
        return config

############################################################################################################################################

# Get the directory of executable data
def getDataFolder():
    # path of your data in same folder of main .py or added using --add-data
    if getattr(sys, 'frozen', False):
        getDataFolder = sys._MEIPASS
    else:
        getDataFolder = os.path.dirname(
            os.path.abspath(sys.modules['__main__'].__file__)
        )
    return getDataFolder

# Specify the path to the pre-trained model and working directory
firstPath = getDataFolder() + "\CustomCNN1.h5"
secPath = getDataFolder() + "\CustomCNN2.h5"
transferPath = getDataFolder() + "\FinalTransferLearningModel.h5"

direc = os.getcwd() + '\\images'

# Load pre-trained model and labels
cnn1 = load_model(firstPath, custom_objects={'ShakeShake': ShakeShake})
cnn2 = load_model(secPath, custom_objects={'ShakeShake': ShakeShake})
transferCNN = load_model(transferPath, custom_objects={'ShakeShake': ShakeShake})

charLabels = pd.read_csv(getDataFolder() + '\\new_k49_classmap.csv')

# System settings
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

# App GUI
app = customtkinter.CTk()
screenWidth = app.winfo_screenwidth() // 3
screenHeight = app.winfo_screenwidth() // 3

app.title('OCR App')
app.geometry(f"{screenWidth}x{screenHeight}+0+0")
app.resizable(True, True)

# Global variables
newWindow = None
changeText = None
newImg = None
isEnglish = True
currentLabel = 'English'

############################################################################################################################################

def textSwitch(switchVar, selectedImage):
    global isEnglish
    global changeText
    global currentLabel
    
    if switchVar.get() == 'Japanese':
        isEnglish = False
        changeText.deselect()
    else:
        isEnglish = True
        changeText.select()
    
    currentLabel = switchVar.get()
    changeText.configure(text=f"Current language: {currentLabel}")
    refreshImage(selectedImage)
    
############################################################################################################################################

def refreshImage(selectedImage):
    global newImg

    # Perform image preprocessing
    grayImg = cv2.cvtColor(selectedImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
    
    # Find and draw the contours on the image based on the required shape
    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    contours = sorted(contours, key = lambda x: cv2.boundingRect(x)[0])
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 100 and w > 30:
            # Extract the region inside contours
            selectedRegion = selectedImage[y:y+h, x:x+w]
            label = processImg(selectedRegion)
            
            # Insert the bounding box and predicted label into the image
            cv2.rectangle(selectedImage, (x, y), (x+w, y+h), (36, 255, 12), 2)
            selectedImage = putTextWithPIL(selectedImage, label, (x+10, y-30), (36, 255, 50), (x, y, w, h)) # Add condition to swap text
            
    imageContoured = customtkinter.CTkLabel(newWindow)
    imageContoured.grid(row = 1, padx = 28, pady = 28)
    
    # Convert the numpy image to a PIL image
    pilConvertedImage = Image.fromarray(cv2.cvtColor(selectedImage, cv2.COLOR_BGR2RGB))
    
    # Convert the PIL image to a CTkImage for displaying with CTkLabel
    newImg = customtkinter.CTkImage(light_image = pilConvertedImage, size = (screenWidth, screenHeight))
    
    # Create image save button
    imageSave = customtkinter.CTkButton(newWindow, text = 'Save image', corner_radius = 15, hover_color = "#FF0000", command = lambda: saveImage(selectedImage))
    imageSave.grid(column = 0, row = 0, padx = 30, pady = 30)
    
    imageContoured.configure(text = "", image = newImg)
   ############################################################################################################################################

def modelPredict(selectedImage):
    global newWindow
    global changeText
    global currentLabel
    
    # Check if window exists
    if newWindow is not None and newWindow.winfo_exists():
        newWindow.destroy()
    
    newWindow = customtkinter.CTkToplevel()
    newWindow.title('OCR App')
    newWindow.geometry(f"{screenWidth}x{screenHeight}+0+0")
    newWindow.resizable(True, True)
    newWindow.wm_transient(app)
    
    switchVar = customtkinter.StringVar(value = currentLabel)
    
    changeText = customtkinter.CTkSwitch(newWindow, text = f"Current language: {currentLabel}", variable = switchVar, 
                                         onvalue = "English", offvalue = "Japanese",
                                         corner_radius = 15,
                                         command = lambda:textSwitch(switchVar, selectedImage))
    changeText.grid(column = 0, row = 2, padx = 50, pady = 30)
    
    newWindow.grid_columnconfigure(0, weight = 1)
    
    refreshImage(selectedImage)

############################################################################################################################################

def processImg(selectedRegion):
    global isEnglish
    
    # Preprocess the region image for model prediction
    newImg = convertImg(selectedRegion, 28)

    # Run model prediction
    firstPred = cnn1.predict(newImg, verbose = None)
    secPred = cnn2.predict(newImg, verbose = None)
    
    newImg = convertImg(selectedRegion, 64, True)
    thirdPred = transferCNN.predict(newImg, verbose = None)
    
    # Stack the predictions of the models and get the maxsum
    allPreds = np.stack([firstPred, secPred, thirdPred])
    summedPreds = np.sum(allPreds, axis=0)
    
    if isEnglish:
        translatedPreds = charLabels[charLabels['index'] == np.argmax(summedPreds)]['english'].to_string(index=False)
    else:
        translatedPreds = charLabels[charLabels['index'] == np.argmax(summedPreds)]['char'].to_string(index=False)
        
    del firstPred, secPred, newImg, thirdPred, allPreds, summedPreds
    
    return translatedPreds

############################################################################################################################################

def convertImg(img, size, transfer = False):
    if transfer:
        resizedRegion = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to color for transfer model  
    else:
        resizedRegion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
    
    resizedRegion = cv2.resize(resizedRegion, (size, size))  # Resize as needed by the models
    resizedRegion = resizedRegion.astype('float32') / 255.0  # Normalize images to speedup prediction process
    resizedRegion = np.expand_dims(resizedRegion, axis=0)  # Add batch dimension
    
    return resizedRegion

############################################################################################################################################

def putTextWithPIL(image, text, position, color, box_dims):
    # Convert the OpenCV image to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Change font for text
    font = ImageFont.truetype("msgothic.ttc", 32)
    
    # Calculate the size of the text
    text_size = draw.textsize(text, font=font)
    
    # Calculate text position to avoid overlap
    x, y = position
    box_x, box_y, box_w, box_h = box_dims
    
    if y - text_size[1] < box_y:
        # If text above the box overlaps, place text below the box
        y = box_y + box_h + 5
    else:
        # Place text above the box
        y = y - text_size[1] - 5
    
    # Draw the text on the image
    draw.text(position, text, font=font, fill=color)
    
    # Convert the PIL image back to an OpenCV image
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return open_cv_image

############################################################################################################################################

def saveImage(image):
    file_path = fd.asksaveasfilename(defaultextension=".jpg", filetypes = [("JPG files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        cv2.imwrite(file_path, image)
        
############################################################################################################################################

def imageSelector():
    filetypes = (('JPEG images', '*.jpg'),
                 ('PNG images', '*.png'), 
                 ('All files', '*.*'))
    
    imageFile = fd.askopenfilename(filetypes = filetypes, initialdir = direc)
    
    # Process opened image for model prediction
    if imageFile:
        global newImg
        
        img = Image.open(imageFile)
        newImg = img.copy()
        
        selectedImage = customtkinter.CTkImage(light_image = img, size = (screenWidth, screenHeight))
        
        # Convert image to openCV format
        convertedImage = np.array(img)
        
        # Convert RGB image to BGR
        convertedImage = convertedImage[:,:,::-1].copy()

        imageLabel.configure(text = "", image = selectedImage)
        #imageLabel.bind('<Configure>', resizeImage)
        
        # Remove the old button if it exists
        if hasattr(app, 'processFile'):
            app.processFile.destroy()
        
        # Dynamically create the Model Predict button if it doesn't exist
        app.processFile = customtkinter.CTkButton(app, text='Model Predict', corner_radius = 15, hover_color = "#FF0000", command = lambda: modelPredict(convertedImage))
        app.processFile.grid(padx=50, pady=30)

############################################################################################################################################

# Create Select File button
selectFile = customtkinter.CTkButton(app, text = 'Select an image', corner_radius = 15, hover_color = "#FF0000", command = imageSelector)
selectFile.grid(padx = 50, pady = 30)

# Create label where image will be placed
imageLabel = customtkinter.CTkLabel(app, text = "Image will be shown here")
imageLabel.grid(padx = 28, pady = 28)

# Align the widgets based on the grid
app.grid_columnconfigure(0, weight = 1)

app.mainloop()


 
