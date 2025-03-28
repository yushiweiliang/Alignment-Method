# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:49:55 2022

@author: jkf
"""
import numpy as np
import matplotlib.pyplot as plt
import glob,os
from tqdm import tqdm
from skimage import filters
import tools as tool
from skimage.transform import rotate,rescale
from astropy.time import Time


outdir='out_1107'                                               ### Output folder
tool.rmdir(outdir)

list304=sorted(glob.glob(r'aia304/aia.lev1_euv_*304.image_lev1.fits'))
filelist_sut=sorted(glob.glob(r'sdi/sdi_lev10_*_v01.fits.gz'))

Jd304=[]

for i in range(len(list304)):
    tmp=os.path.basename(list304[i])                           ### Get filename
    s=tmp.index('2024-')                                       ### Get the index position of the first match to 2024 in the filename (the 18th)
    tmp=tmp[s:s+17]
    Jd304.append(Time(tmp[:13]+':'+tmp[13:15]+':'+tmp[15:]).jd)### Replace time with Julian day
Jd304=np.array(Jd304)


JDsdi=[]
OBStime=[]
plt.close('all')

tool.mkdir(outdir)
for i in range(len(filelist_sut)):                             ### Ditto for getting sdi Julian day
    tmp=os.path.basename(filelist_sut[i])
    s=tmp.index('lev10_2024')
    tmp=tmp[s+6:s+22]
    OBStime.append(tmp[:4]+'-'+tmp[4:6]+'-'+tmp[6:8]+'T'+tmp[9:11]+':'+tmp[11:13]+':'+tmp[13:15])
    JDsdi.append(Time(tmp[:4]+'-'+tmp[4:6]+'-'+tmp[6:8]+'T'+tmp[9:11]+':'+tmp[11:13]+':'+tmp[13:15]).jd)

JDsdi=np.array(JDsdi)    
result=[]
for i in tqdm(range(len(filelist_sut))):

    k304=np.abs(Jd304-JDsdi[i]).argmin()
    
    if np.abs(Jd304[k304]-JDsdi[i])>0.0007:
                    print('No suitable AIA file')
                    continue
    im304,h304=tool.fitsread(list304[k304])
    h304['AIA304']=os.path.basename(list304[k304])
    h304['SDIname']=os.path.basename(filelist_sut[i])
    
    s304=filters.gaussian(im304,21)
    imsdi,hsdi=tool.fitsread(filelist_sut[i])  
    imsdi,ray=tool.removeray(imsdi,T=0.5)                   ### get rid of hot pixels


    RO=360-hsdi['CROTA2']
    imsdi=rotate(imsdi,RO,order=0)

    ###rescale SDI
    imsdi=rescale(imsdi,5/6)
    scale=imsdi.shape[0]/4608
    imsdi=np.pad(imsdi,128)

    ###Using the cc algorithm to bring the two images close
    c=tool.cc(im304,imsdi)
    imsdi=tool.immove2(imsdi,c[1],c[0])

    im0=np.dstack((s304,imsdi)).transpose(2,0,1)
    im=im0[[0,1]]

    #############align##############r
    align=[[0,1]]
    D=tool.disk(4096,4096,1600)
    w=500

    ####OF alignment
    im0,x,cor=tool.all_align(im[:].copy(),align,mask=D>0,winsize=51,step=50,r_t=5,arrow=0,sample=20)
    ####Output x is 3*4,x[1] is the four parameters of the SDI geometric transformation
    ####x[2] is the error of the four parameters; cor effective points

    im0[0]=im304
    result.append([JDsdi[i],0.5/x[1][0],np.rad2deg(x[1][1]),x[1][2],x[1][3],x[2][0],np.rad2deg(x[2][1]),x[2][2],x[2][3]])
    print(0.5/x[1][0],np.rad2deg(x[1][1]))
    ###save fits file
    file=outdir+'/'+os.path.basename(filelist_sut[i])[:-3]
    tool.fitswrite(file,im0.astype('float32'),h304)

result=np.array(result)
np.save('result1106.npy',result)

plt.figure()
plt.subplot(211)
plt.plot(result[:,0],result[:,2].astype('float32'),'.-')
plt.subplot(212)
plt.plot(result[:,0],result[:,1].astype('float32'),'.-')
plt.show()
