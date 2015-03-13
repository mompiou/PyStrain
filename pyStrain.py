import numpy as np
import scipy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import matplotlib.image as mpimg
import openpiv.tools
import openpiv.process
import openpiv.scaling
from matplotlib import pyplot as plt
import glob
from pyqtgraph.dockarea import *
import re

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('PyStrain')

d1 = Dock(" ", size=(1, 1))     ## give this dock the minimum possible size
d2 = Dock(" ", size=(500,300), closable=True)

area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
area.addDock(d2, 'right')     ## place d2 at right edge of dock area




w1 = pg.LayoutWidget()
### Define a top-level widget to hold everything
#w = pg.GraphicsWindow(size=(1000,800), border=True)
#w.setWindowTitle('pyqtgraph example: ROI Examples')


#imv = pg.ImageView()
#

        
list_of_files = sorted(glob.glob('images/*.tif'))
n=len(list_of_files)
print(n)
n_counter=QtGui.QSpinBox()
n_counter.setMaximum(100000)
n_counter.setMinimum(1)
n_counter.setValue(n)




text2=QtGui.QLabel(" u,v (px):0,0 ")

arr0=openpiv.tools.imread(list_of_files[0])
arr1=openpiv.tools.imread(list_of_files[n-1])
arr0=np.rot90(arr0,k=3)
arr1=np.rot90(arr1,k=3)
## Create a grid layout to manage the widgets size and position

ncount = w1.addWidget(n_counter, row=0, col=0)

button_count=QtGui.QPushButton('update image')
but_count=w1.addWidget(button_count,row=0,col=1)


label2 = w1.addWidget(text2, row=1, col=1)
button=QtGui.QPushButton('PIV')
but=w1.addWidget(button,row=1,col=0)

pos1=50
pos2=50
size1=75
size2=75



button2=QtGui.QPushButton('Set Area')
but2=w1.addWidget(button2,row=3,col=0)


button3=QtGui.QPushButton('Reset')
but3=w1.addWidget(button3,row=4,col=0)


button4=QtGui.QPushButton('Serie')
but4=w1.addWidget(button4,row=5,col=0)

button5=QtGui.QPushButton('Strain')
but5=w1.addWidget(button5,row=6,col=0)

d1.addWidget(w1)

w2 = pg.GraphicsWindow(size=(1000,800), border=True)
d2.addWidget(w2)

v0 = w2.addViewBox(row=1, col=0, lockAspect=True)
v1 = w2.addViewBox(row=1, col=1, lockAspect=True)
vr= w2.addViewBox(row=2, col=0, lockAspect=True)
vrb= w2.addViewBox(row=2, col=1, lockAspect=True)



#
img0 = pg.ImageItem(arr0)
img1 = pg.ImageItem(arr1)
imgroi_a = pg.ImageItem()
imgroi_b = pg.ImageItem()

v0.addItem(img0)
v1.addItem(img1)
vr.addItem(imgroi_a)
vrb.addItem(imgroi_b)
v0.disableAutoRange('xy')
v1.disableAutoRange('xy')
vr.disableAutoRange('xy')

#



rois_a = []
rois_a.append(pg.RectROI([pos1, pos2], [size1, size2], pen=(0,9), parent=v0))
rois_a[-1].addScaleHandle((1,1), (0,0), axes=None, item=None, name=None, lockAspect=True, index=None)
#rois_a[-1].addRotateHandle([1,0], [0.5, 0.5])
roi_t=[]
arrows=[]

######Reset ROI

def reset():
    global D,roi_t,arrows,a,b
    D=np.array([0,0,0,0])
    
    for roi in roi_t:
        v1.removeItem(roi)
    for roi in rois_a:
        v0.removeItem(roi)    
    v0.addItem(rois_a[-1])
    for arrow in arrows:
        v1.removeItem(arrow)
    update_a(rois_a[-1])
    
	
	
def update_img():
	global arr1, Ep
	
	arr1=openpiv.tools.imread(list_of_files[n_counter.value()-1])
	arr1=np.rot90(arr1,k=3)
	img1 = pg.ImageItem(arr1)
	v1.addItem(img1)
	update_a(rois_a[-1])
	Ep=np.zeros((n_counter.value(),4))
	
def update_a(roi_a):
    global a,b,g
    
    imgroi_a.setImage(roi_a.getArrayRegion(arr0, img0), levels=(0, arr0.max()))
    a=roi_a.getArrayRegion(arr0, img0)
    
    vr.autoRange()
    g=roi_a.pos()
    
    return a,g
    
def place():
    global a,b,g,roi_t
    roi_c=pg.RectROI([g[0], g[1]], [a.shape[0], a.shape[1]], pen=(0,9),parent=v1)
    v1.addItem(roi_c)
    roi_t.append(roi_c)
    imgroi_b.setImage(roi_c.getArrayRegion(arr1, img1), levels=(0, arr1.max()))
    b=roi_c.getArrayRegion(arr1, img1)
    update_a(rois_a[-1])
    vrb.autoRange() 
    return b    
P=[]

#####Calculate Strain and draw the stress strain curve from a stress txt file

def epsilon():
    global Ep
    e=(Ep[:,4]-Ep[:,8])/(Ep[:,6]-Ep[:,10])
    e[0]=0
    es=np.cumsum(e)
    print(Ep.shape,Ep,e,es)
    file_f = open("stress.txt", "r")
    x0=[]
    
    for line in file_f:
		x0.append(map(str, line.split()))
    file_f.close()
    file = open("result.txt", "w")
    for k in range(e.shape[0]):
		file.write(str(e[k])+'\n')
    file.close()
    plt.plot(e,x0[0:e.shape[0]],'bo')
    plt.show()

        
   
#def update_b(roi_b):
#    global a,b
#    
#    imgroi_b.setImage(roi_b.getArrayRegion(arr1, img1), levels=(0, arr1.max()))
#    b=roi_b.getArrayRegion(arr1, img1)
#    vrb.autoRange() 
#    
#    return b

D=np.array([0,0,0,0])    
def piv():
    global a,b,D,g,arrows
    a=np.rot90(a,k=1)
    b=np.rot90(b,k=1)
    #print(a.shape,b.shape)
    ov=np.int(a.shape[0]-2)
    ws=np.int(a.shape[0]-1) #taille de la zone de recherche dans la premiere image
    sas=np.int(b.shape[0]-1) #taille de la zone de recherche dans la seconde image
    #img=mpimg.imread('Be1-7b (Channel 1) 010 - 00006.tif')
    
    u, v, sig2noise = openpiv.process.extended_search_area_piv( a.astype(np.int32), b.astype(np.int32), window_size=ws, overlap=ov, dt=1, search_area_size=sas, sig2noise_method='peak2peak' )
    
    
    x, y = openpiv.process.get_coordinates( image_size=a.shape, window_size=ws, overlap=ov )
    
    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold =0 )
    
    #u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    #print(u,v)
    #plt.quiver( x, y, u, v, color='blue' )
    #plt.quiver( x, y, u, v )
    #plt.show()   
     
    D=np.vstack((D,np.array([u[0,0],v[0,0],g[0],g[1]])))
    print(D)
    text2.setText('u,v (px):'+str(np.around((u[0,0]), decimals=2))+','+str(np.around(v[0,0],decimals=2)))
    umean=u[0,0]
    vmean=v[0,0]
    print(umean,vmean)
    arrow=pg.ArrowItem(headlen=5, tailLen=umean, angle=180+np.arctan(-vmean/umean)*180/np.pi)
    
    arrow.setPos(g[0]+a.shape[0]/2, g[1]+a.shape[1]/2)
    arrows.append(arrow)
    v1.addItem(arrow)
    epsx()
#    Dx=x[0,1]-x[0,0]
#    Dy=y[0,0]-y[1,0]
#    print(Dx,Dy)
#    X=x[0,:]
#    Y=y[:,0]
#    U=u[0,:]
    
    #dU=np.diff(u)/(a.shape[1])
    #dV=np.diff(v, axis=0)/(frame_a.shape[1])
    
    #eps=np.sum(dU)/dU.shape[0]
    #print(eps,dU.shape[0])    
    return u,v
    
def epsx():
    global D
    
    E=np.zeros((D.shape[0],D.shape[0]))
    print(E)
    for h in range(1,D.shape[0]):
        for j in range(1,D.shape[0]):
            E[h,j]=(D[h,0]-D[j,0])/(D[h,2]-D[j,2])
    E=np.delete(E,0,0)
    E=np.delete(E,0,1)    
    print(E)

   
ux=0
uy=0


Ep=np.zeros((n_counter.value(),4))


def piv_serie():
    global g,Ep
    
    
    #print(a.shape,b.shape)
    roi_as=[]
    roi_bs=[]
    U=np.array([0,0,0,0])
    
    for i in range(1,n_counter.value()):
        
        
        # read images into numpy arrays
        frame_a  = openpiv.tools.imread( list_of_files[0] )
        frame_b  = openpiv.tools.imread( list_of_files[i] )
        
        frame_a=np.rot90(frame_a,k=3)
        frame_b=np.rot90(frame_b,k=3)
        img_a = pg.ImageItem(frame_a)
        img_b = pg.ImageItem(frame_b)
        v0.addItem(img_a)
        v1.addItem(img_b)
        
        roi_as.append(pg.RectROI([g[0], g[1]], [a.shape[0], a.shape[1]], pen=(0,9),parent=v0))
        roi_bs.append(pg.RectROI([g[0], g[1]], [a.shape[0], a.shape[1]], pen=(0,9),parent=v1))
        v0.addItem(roi_as[i-1])
        v1.addItem(roi_bs[i-1])
        
        a_s=roi_as[i-1].getArrayRegion(frame_a, img_a)
        b_s=roi_bs[i-1].getArrayRegion(frame_b,img_b)
        
        
        
        a_s=np.rot90(a_s,k=1)
        b_s=np.rot90(b_s,k=1)
        ov=np.int(a_s.shape[0]-2)
        ws=np.int(a_s.shape[0]-1) #taille de la zone de recherche dans la premiere image
        sas=np.int(b_s.shape[0]-1)
       
        
        # process image pair with extended search area piv algorithm.
        u, v, sig2noise = openpiv.process.extended_search_area_piv( a_s.astype(np.int32), b_s.astype(np.int32), window_size=ws, overlap=ov, dt=1, search_area_size=sas, sig2noise_method='peak2peak' )
       
        # get window centers coordinates
        x, y = openpiv.process.get_coordinates( image_size=a_s.shape, window_size=ws, overlap=ov )
        
        ux=u[0,0]
        uy=v[0,0]
        
        g[0]=g[0]
        g[1]=g[1]
        U=np.vstack((U,np.array([u[0,0],v[0,0],g[0],g[1]])))
        
        
        v0.removeItem(roi_as[i-1])
        #v1.removeItem(roi_bs[i-2])
        v0.removeItem(img_a)
        v1.removeItem(img_b)
        
       
        print(ux,uy)
    Ep=np.hstack((Ep,U))
    print(Ep)

   
    
for roi in rois_a:
    roi.sigRegionChanged.connect(update_a)
    v0.addItem(roi)
    
update_a(rois_a[-1])




#for roi in rois_b:
#    roi.sigRegionChanged.connect(update_b)
#    v1.addItem(roi)
#    
#update_b(rois_b[-1])   
    
button.clicked.connect(piv)
button_count.clicked.connect(update_img)
button2.clicked.connect(place)
button3.clicked.connect(reset)
button4.clicked.connect(piv_serie)
button5.clicked.connect(epsilon)


## Display the widget as a new window
win.show()

## Start the Qt event loop
app.exec_()



