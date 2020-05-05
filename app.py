import json
import pdb, sys, os, io
import copy

import torch
from tkinter import *
from PIL import ImageTk, Image
from vae import interpolation_experiment, VAE_CNN

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

#########################################################
#########################################################

def init(data):
    data.model_name = './pretrained_models/model.pth.pt'
    data.imgs = []
    data.cur_text = []
    data.image_order = []
    data.error = None
    data.error_opacity = 0

    data.complete = None
    data.complete_opacity = 0

    h = 2048
    z = 128

    # Model
    data.model = VAE_CNN(1, h, z).to(device)
    data.model.to(device)
    data.model.load_state_dict(torch.load('./pretrained_models/model.pth.pt', map_location=torch.device('cpu')))

#########################################################
#########################################################

def show_error(data, error_message):
    data.error = error_message
    data.error_opacity = 20

def show_complete(data, complete_message):
    data.error = None
    data.error_opacity = 0

    data.complete = complete_message
    data.complete_opacity = 100

def save_video(data):
    video_name = ''
    list_of_names = []
    for index in data.image_order:
        utf = data.list_of_names[index]
        video_name += '%s_to_' % utf
        list_of_names.append(utf)
    video_name += '.mp4'
    interpolation_experiment(data.model, list_of_names, video_name)

    show_complete(data, 'Video saved as %s' % video_name)

def keyPressed(event, data):
    # Revert to a previous item
    if event.keysym == 'Return':
        if len(data.image_order) < 2:
            show_error(data, "You need at least two images to interpolate")
        else:
            save_video(data)

def resave_images(data):
    # Reset image lists
    data.imgs = []
    data.cur_text = []

    # Create a set sizes
    sizes = [256, 250, 200, 180, 150, 120, 100, 100, 100, 100, 80, 60, 60, 60, 60, 60]
    data.n_images = len(data.image_order)
    w = sizes[data.n_images - 1]

    for index in data.image_order:
        # Get images
        utf = data.list_of_names[index]
        image_dir = './kmnist/kkanji2/%s/' % utf
        images = sorted(os.listdir(image_dir))
        name = image_dir + images[0]
        image = Image.open(name)
        image = image.resize((w, w), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        # Get names
        data.imgs.append(image)
        data.cur_text.append(utf)

def mousePressed(event, data):
    if len(data.list_box.curselection()) > 0:
        if len(data.list_box.curselection()) <= 16:
            if len(data.image_order) == len(data.list_box.curselection()):
                pass # unchanged
            elif len(data.image_order) > len(data.list_box.curselection()):
                # one removed
                img_order = copy.copy(data.image_order)
                for element in img_order:
                    if element not in data.list_box.curselection():
                        data.image_order.remove(element)
                        break
                resave_images(data)
            elif len(data.image_order) < len(data.list_box.curselection()):
                # one added
                for element in data.list_box.curselection():
                    if element not in data.image_order:
                        data.image_order.append(element)
                        break
                resave_images(data)

        else:
            for element in data.list_box.curselection():
                if element not in data.image_order:
                    data.image_order.append(element)
                    break
            data.list_box.selection_clear(data.image_order[-1])
            data.image_order = data.image_order[:-1]
            show_error(data, "You've reached the max number of selections, press enter to interpolate")

def timerFired(data):
    if data.error_opacity > 0:
        data.error_opacity -= 1

    if data.complete_opacity > 0:
        data.complete_opacity -= 1

#########################################################
#########################################################

def drawInstructions(canvas, data):
    instruct1 = 'Select up to 16 images to interpolate between. Press enter to interpolate. Wait for confirmation at top of screen.'
    canvas.create_text(data.width // 2, data.height - 20, text=instruct1,
                       font="Arial 14", anchor=S, fill='white')

    if data.error_opacity > 0:
        canvas.create_text(data.width // 2, 20, text=data.error,
                       font="Arial 20", anchor=N, fill='red')

    if data.complete_opacity > 0:
        canvas.create_text(data.width // 2, 20, text=data.complete,
                       font="Arial 20", anchor=N, fill='DeepSkyBlue')

def displayImages(canvas, data):
    for i, image in enumerate(data.imgs):
        cur_offset = (i + 1) * (data.width // (data.n_images + 1))
        canvas.create_image(cur_offset, data.height // 4, anchor=N, image=image)

def drawButtons(canvas, data):
    pass

def redrawAll(canvas, data):
    drawInstructions(canvas, data)
    displayImages(canvas, data)
    drawButtons(canvas, data)

####################################
# use the run function as-is
####################################

def run(width, height):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='#23282A', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()

    data.width = width
    data.height = height

    data.timerDelay = 100 # milliseconds
    init(data)

    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=data.width, height=data.height)

    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)

    # Get list of items
    data.list_box = Listbox(root, selectmode=MULTIPLE)
    image_dir = './kmnist/kkanji2/'
    image_names = sorted(os.listdir(image_dir))
    data.list_of_names = []
    i = 1
    for image in image_names:
        if 'U+' in image:
            data.list_of_names.append(image)
            data.list_box.insert(i, image)
            i += 1
    data.list_box.place(x = data.width // 2, y = data.height // 4 + 276, width=256, height=200, anchor=N)

    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

if __name__ == '__main__':
    run(1080, 720)

