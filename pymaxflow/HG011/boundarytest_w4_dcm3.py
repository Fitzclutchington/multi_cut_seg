import sys
import os
import numpy as np
import dicom
import pylab
import Image
import graphcutseg
import json
import math
import glob

def pyarraytopng(img, size, outfile):
	npimg = np.array(img)
	newpixarray = npimg.astype('B')
	pngimg = Image.fromstring("L",size,newpixarray.tostring())
	pngimg.save(outfile,"PNG")
	return
def blindingouput(seg, imgcolor, size, outfile):
	r,g,b = imgcolor.split()
	ra = [p*0.5+q*0.5 for p,q in zip(seg, r.getdata())] 
	ga = [p*0.5+q*0.5 for p,q in zip(seg, g.getdata())]	
	ba = [p*0.5+q*0.5 for p,q in zip(seg, b.getdata())]
  	r.putdata(ra)
	g.putdata(ga)
	b.putdata(ba)
	newimg = Image.merge('RGB',(r,g,b))
	newimg.save(outfile,"PNG")
	return

if (len(sys.argv) < 2):
	print "Usage [configuration json file]"
	exit()

	
#nclasses = int(sys.argv[1])
#each dir contains dicom images from one modality
f = open(sys.argv[1],'r')
task = json.load(f)
f.close()

training_images = [p for p in task['training_images']]
modalities = [m for m in task['modalities']]
training_image_indices = [m for m in task['training_image_indices']]
case = task['case']
prefix = task['prefix']
outdir =  case + '/' + task['outdir']
dicomdir = task['dicomdir']
groundtruthdir = task['groundtruthdir']


# trainidx = int(sys.argv[1]) #training index starting from 1
# brushimg = Image.open(sys.argv[2])
# segimg= Image.open(sys.argv[3])
# outdir = sys.argv[4]

dirs = [case +"/"+dicomdir+"/" + m for m in modalities]

mmdirs = []

for d in dirs:
	fil = d + '\\*.dcm'
	lis = glob.glob(fil)
	lis.sort()
	mmdirs.append(lis)


fil = case +'\\'+groundtruthdir+'\\*.png'
gtfiles = glob.glob(fil)
gtfiles.sort()

objsamples = []
bcksamples = []
bRegional = 0
boundarysamples = []
nonboundarysamples = []

for t in range(len(training_image_indices)):
	trainidx = training_image_indices[t]
	brushimg = Image.open(training_images[t])
	segimg = Image.open(gtfiles[trainidx)
	#training image files (single slice mulit-modality)
	trainfiles = [m[trainidx-1] for m in mmdirs]
	print trainfiles
	print gtfiles[trainidx]
	trainmmimg = []
	for tf in trainfiles:
			dcmimg  = dicom.read_file(tf)
			dim = dcmimg.pixel_array.shape
			pix_list = list(np.reshape(dcmimg.pixel_array, dim[0]*dim[1])) #convert to 1d list used by this codes
			trainmmimg.append(pix_list)

	ipixels = []
	for i in range(dim[0]*dim[1]):
		ipixels.append([int(pixels[i]) for pixels in trainmmimg])
		
	bpixels = list(brushimg.getdata())
	print len(ipixels)
	print len(bpixels)
	if (len(ipixels) != len(bpixels)):
		print 'brush image size is different than training image size'
		exit()

	iw = segimg.size[0]
	ih = segimg.size[1]

	segdata = segimg.getdata()
	segpixels = [p for p in segdata]
	print len(segpixels)

	if (len(ipixels) != len(segpixels)):
		print 'training ground truth image size is different than training image size'
		exit()
	i = 0
	for p in bpixels:
		if p[0] > 250 and p[1] < 10:
			objsamples.append(ipixels[i])
		if p[0] < 10  and p[2] > 250:
			bcksamples.append(ipixels[i])
		i = i + 1

	outboundary = [0 for i in range(iw*ih)]
	outnonboundary = [0 for i in range(iw*ih)]
	for i in range(iw*ih):
		x = i%iw
		y = i/iw
		if x < iw-1 and segpixels[i] != segpixels[i+1]:
			if segpixels[i] > 100:
				boundarysamples.append([ipixels[i],ipixels[i+1]])
				outboundary[i] = 1
			else:
				boundarysamples.append([ipixels[i+1],ipixels[i]])
				outboundary[i+1] = 1
		else:
			if y < ih-1 and segpixels[i] != segpixels[i+iw]:
				if segpixels[i] > 100:
					boundarysamples.append([ipixels[i],ipixels[i+iw]])
					outboundary[i] = 1
				else:
					boundarysamples.append([ipixels[i+iw],ipixels[i]])
					outboundary[i+iw] = 1
	for i in range(iw*ih):
		x = i%iw
		y = i/iw
		if (i%200 == 0):
			if x < iw-1 and segpixels[i] == segpixels[i+1]:
				nonboundarysamples.append([ipixels[i],ipixels[i+1]])
				outnonboundary[i] = 1
			else:
				if y < ih-1 and segpixels[i] == segpixels[i+iw]:
					nonboundarysamples.append([ipixels[i],ipixels[i+iw]])
					outnonboundary[i] = 1



#print boundarysamples
#print nonboundarysamples
#print len(boundarysamples)
#tt = []
#for s in nonboundarysamples:
#	for k in range(len(s[0])):
#		print type(s[0][k])

[betas, means, variances] = graphcutseg.trainboundary_mm(boundarysamples,nonboundarysamples)
[rbetas, rmeans, rvars] = graphcutseg.trainregional_mm(objsamples,bcksamples)
betas
means
variances
rbetas
rmeans
rvars

graphcutseg.setboundaryparms_mm(betas,means,variances)
graphcutseg.setregionalparms_mm(rbetas,rmeans,rvars)
graphcutseg.setboundaryweight(4)

mmfiles = []
for i in range(0, len(mmdirs[0])):
	mmfiles.append([m[i] for m in mmdirs])

print outdir
cnt = 1
for mmf in mmfiles:
	mmimg = []
	for sf in mmf:
		dcmimg  = dicom.read_file(sf)
		pix_list = map(int, list(np.reshape(dcmimg.pixel_array, dim[0]*dim[1])))
		mmimg.append(pix_list)
		print sf
		print "image size {0}".format(len(pix_list))	
	print "modalities {0}".format(len(mmimg))
	seg = graphcutseg.graphcut_mm3(iw,ih, mmimg, 0, [], [], 0, [], [],[])
	seg2 = [100*p for p in seg]
	outfile = outdir + "/seg_{0:04d}.png".format(cnt)
	cnt = cnt + 1
	print outfile
	pyarraytopng(seg2, [dim[1], dim[0]], outfile)

res = graphcutseg.getnoderesidual()
resmax = max(res)
resmin = min(res)
print
print resmax
print resmin
res2 = [ int((r-resmin)/(resmax-resmin)*255) for r in res]
pyarraytopng(res2, [dim[1], dim[0]],"res.png")
