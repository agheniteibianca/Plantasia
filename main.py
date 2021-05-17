import PIL
import cv2
import os
import numpy as np
import pandas
import easygui
from PIL import ImageTk, Image
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tkinter import filedialog
import model as bm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from timeit import default_timer as timer
from datetime import timedelta

dataset_path = "Flavia"
selected_img = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/1.jpg'
selected_img1 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/2.jpg'
selected_img2 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/3.jpg'
selected_img3 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/4.jpg'
selected_img4 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/5.jpg'
selected_img5 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/6.jpg'
selected_img6 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/7.jpg'
selected_img7 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/8.jpg'
selected_img8 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/9.jpg'
selected_img9 = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/standardleaves/10.jpg'




img_files = os.listdir(dataset_path)

root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


#binarizare

def Treshhold(path):
    image = cv2.imread(path, 0)
    a = image.copy()
    width, height = image.shape

    for i in range (0,width):
        for j in range (height):
            if a[i][j] >= 125:
                a[i][j] =255
            else:
                a[i][j] =0

    image = image_resize(image, height = 400)
    cv2.imshow("Original " , image)

    a = image_resize(a, height = 400)
    cv2.imshow("Binarizare ", a)
    return a



def create_dataset():
    names = ['area', 'perimeter', 'length', 'width', 'aspect_ratio','mean_r', 'mean_g', 'mean_b', 'className' ]
    dataframe = pandas.DataFrame([], columns=names)

    i = 1000;

    for file in img_files:
        i+=1
        imgpath = dataset_path + "\\" + file
        main_img = cv2.imread(imgpath)



        #preprocessing
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25, 25), 0)
        ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((50, 50), np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)


        #shape features
        contours, hierarchy  = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h


        #color features

        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        #classes for flavia dataset
        if(i >= 1001 and i <= 1059):
            className = 1
        if(i >= 1060 and i <= 1122):
            className = 2
        if(i >= 1552 and i <= 1616):
            className = 3
        if(i >= 1123 and i <= 1194):
            className = 4
        if(i >= 1195 and i <= 1267):
            className = 5
        if(i >= 1268 and i <= 1323):
            className = 6
        if(i >= 1324 and i <= 1385):
            className = 7
        if(i >= 1386 and i <= 1437):
            className = 8
        if(i >= 1497 and i <= 1551):
            className = 9
        if(i >= 1438 and i <= 1496):
            className = 10
        if(i >= 2001 and i <= 2050):
            className = 11
        if(i >= 2051 and i <= 2113):
            className = 12
        if(i >= 2114 and i <= 2165):
            className = 14
        if(i >= 2166 and i <= 2230):
            className = 15
        if(i >= 2231 and i <= 2290):
            className = 16
        if(i >= 2291 and i <= 2346):
            className = 17
        if(i >= 2347 and i <= 2423):
            className = 18
        if(i >= 2424 and i <= 2485):
            className = 19
        if(i >= 2486 and i <= 2546):
            className = 20
        if(i >= 2547 and i <= 2612):
            className = 21
        if(i >= 2616 and i <= 2675):
            className = 22
        if(i >= 3001 and i <= 3055):
            className = 23
        if(i >= 3056 and i <= 3110):
            className = 24
        if(i >= 3111 and i <= 3175):
            className = 25
        if(i >= 3176 and i <= 3229):
            className = 26
        if(i >= 3230 and i <= 3281):
            className = 27
        if(i >= 3282 and i <= 3334):
            className = 28
        if(i >= 3335 and i <= 3389):
            className = 29
        if(i >= 3390 and i <= 3446):
            className = 30
        if(i >= 3447 and i <= 3510):
            className = 31
        if(i >= 3511 and i <= 3563):
            className = 32
        if(i >= 3566 and i <= 3621):
            className = 33


        vector = [area, perimeter, w, h, aspect_ratio,red_mean, green_mean, blue_mean, className ]

        dataframe_buff = pandas.DataFrame([vector], columns=names)
        dataframe = dataframe.append(dataframe_buff)
        print(file)


    print(dataframe.shape)
    print(type(dataframe))
    dataframe.to_csv("Flavia.csv")

    return dataframe


def test_knn():
    dataset =  create_dataset()
    X = dataset.iloc[:, :-1].values

    y = dataset.iloc[:, 8].values
    y=y.astype('int')


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)



    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)


    print(classification_report(y_test, y_pred))



    ##################################### now test one picture #########################################

    main_img = cv2.imread(selected_img)


    names = ['area', 'perimeter', 'length', 'width', 'aspect_ratio','mean_r', 'mean_g', 'mean_b' ]
    test_dataframe = pandas.DataFrame([], columns=names)


    main_img = np.array(main_img)

    #preprocessing
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)


    #shape features
    contours, hierarchy  = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h


    #color features

    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)



    vector = [area, perimeter, w, h, aspect_ratio,red_mean, green_mean, blue_mean ]
    dataframe_buff = pandas.DataFrame([vector], columns=names)
    test_dataframe = test_dataframe.append(dataframe_buff)


    test_dataframe = scaler.transform(test_dataframe)
    y_pred = classifier.predict(test_dataframe)
    print(y_pred)



    #Afiseaza rezultatul predictiei
    if(y_pred[0] == 6):
        result_image = 'C:/Users/aghen/OneDrive/Desktop/ProiectPI/Flavia/1268.jpg'

    img = PIL.Image.open(result_image)
    img = img.resize((250, 250), PIL.Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)

    panel2 = Label(root, image=img)
    panel2.image = img
    panel2.pack(side='left', fill='both', expand=True)


def test_cnn():
    model_ft = bm.basicCNN(8, 32, 100, 100)

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_to_data = './Flavia/'  ## se modifica spre directorul unde sunt datele


    dataset_name = 'flavia'

    data_dir = path_to_data + dataset_name

    # num_classes = 32;
    # classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'
    #              , '27', '28', '29', '30', '31', '32', '33')

    num_classes = 12;
    classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12')

    height = 100
    width = 100

    # Batch size for training (change depending on how much memory you have)
    batch_size = 128

    # Number of epochs to train for
    num_epochs = 20

    image_size = (height, width)
    # data transforms for airplanes and moto dataset
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val', 'test']}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in
        ['train', 'val', 'test']}

    # get some random training images
    dataiter = iter(dataloaders_dict['train'])
    images, labels = dataiter.next()

    # show images
    # evalMetrics.imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%s' % classes[j] for j in range(num_classes)))

    # incarcam un model de retea neuronala definit in fisierul basicModels.py
    nf = 8
    model_ft = bm.basicCNN(nf, num_classes, width, height)

    print(model_ft)
    model_class_name = model_ft.__class__.__name__
    model_ft = model_ft.to(device)

    optimizer = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    start = timer()

    # set eval = True daca dorim sa facem evaluarea modelului
    # eval = False
    # if eval:
    #     ckp_path = 'saved-models/basicCNN-' + dataset_name + '.pth'
    #
    #     model_ft.load_state_dict(torch.load(ckp_path))
    #     model_ft.eval()
    #     # if num_classes == 2:
    #     #    acc, tp, tn  = evalMetrics.confusion_matrix(model_ft, dataloaders_dict, 'test', nf, model_class_name)
    #     # else:
    #     acc = evalMetrics.confusion_matrix_n(model_ft, dataloaders_dict, 'test', nf, model_class_name, num_classes,
    #                                          classes)
    #
    # else:


    epoch = 0
    model_ft, hist = bm.train_model(model_ft, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs,
                                       start_epoch=epoch)

    # PATH = './saved-models/caltech_NF_' + str(nf) + '_' + model_class_name + str(int(acc*100)) + '.pth'
    #PATH = ckp_path = 'saved-models/basicCNN-' + dataset_name + '.pth'

    PATH = 'saved-models/basicCNN-' + dataset_name + '.pth'
    torch.save(model_ft.state_dict(), PATH)

    end = timer()
    print('Ellapsed time ', timedelta(seconds=end - start))



    #test one image using cnn

    global selected_img
    global selected_img1
    global selected_img2
    global selected_img3
    global selected_img4
    global selected_img5
    global selected_img6
    global selected_img7
    global selected_img8
    global selected_img9

    demo = PIL.Image.open(selected_img)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img1)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img2)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)


    demo = PIL.Image.open(selected_img3)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img4)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img5)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img6)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img7)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img8)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])

    print(output)

    demo = PIL.Image.open(selected_img9)
    demo = data_transforms['test'](demo)
    demo = demo.to(device)
    output = model_ft(demo[None, ...])
    print(output)




def apply_filter():
    path = easygui.fileopenbox()
    binarized_image = Treshhold(path)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def openfn():
    selected_img = filedialog.askopenfilename(title='open')
    return selected_img


def open_img():
    global selected_img
    selected_img = openfn()


    img = PIL.Image.open(selected_img)
    img = img.resize((250, 250), PIL.Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack(side='left', fill='both', expand=True)


    w.pack()




if __name__ == '__main__':
    w = Label(root, text="Welcome to my app! Pick a leaf and get a prediction!")
    test_cnn()
    w.pack()

    btn = Button(root, text='open image', command=open_img).pack()
    btnKnn = Button(root, text='Test knn', command=test_knn).pack()
    btnCnn = Button(root, text='Test cnn', command=test_cnn).pack()





    root.mainloop()



