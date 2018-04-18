import csv
import numpy as np
import os


#annotation file
filename = 'train.csv'
with open(filename) as f:
	reader = csv.reader(f)
	rows = [row for row in reader]
print 'dataset length:', len(rows)
def get_iterm(rows,index):
	keypoint = []
	for i, j in enumerate(rows[index]):
		if i == 0:
			im = j
		if i == 1:
			classes = j
		if i >1:
			temp = map(eval,j.split('_'))
			keypoint.append(temp[0:2])
	keypoints = np.array(keypoint).ravel()
	#keypoints = points.reshape(-1)
	#mask = np.append(points[:,2].reshape(-1,1),points[:,2].reshape(-1,1),1)
	#mask = mask.reshape(-1)
	return im, classes,keypoints
im, classes,keypoints = get_iterm(rows,26921)
#blouse
#keypoints = np.append(keypoints[0:14],keypoints[18:30])
#skirt
#keypoints = keypoints[30:38]
#outwear
#keypoints = np.append(keypoints[0:4],keypoints[6:30])
#dress
#keypoints = np.append(keypoints[0:26],keypoints[34:38])
#trousers
#keypoints = np.append(keypoints[30:34],keypoints[38:48])
#print keypoints.shape, keypoints


class1 = open('train_blouse.txt','w')
class2 = open('train_skirt.txt','w')
class3 = open('train_outwear.txt','w')
class4 = open('train_dress.txt','w')
class5 = open('train_trousers.txt','w')

root = 'train/'
for i in range(1, 31632): #31632
	im, classes ,keypoints= get_iterm(rows,i)
	#print len(rows[i]),rows[i]
	#print im
	#print points.shape,points
	#print mask
	if cmp(classes,'blouse') == 0:
		points = np.append(keypoints[0:14],keypoints[18:30])
		class1.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points).strip() + ',\n') #
		#class1.writelines([root+im,'\n'])
		#class1.writelines([str(points),'\n'])
	elif cmp(classes, 'skirt') == 0:
		points = keypoints[30:38]
		class2.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + '\n')
	elif cmp(classes,'outwear') == 0:
		points = np.append(keypoints[0:4],keypoints[6:30])
		class3.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + '\n')
	elif cmp(classes, 'dress') == 0:
		points = np.append(keypoints[0:26],keypoints[34:38])
		class4.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + 'n')
	elif cmp(classes,'trousers') == 0:
		points = np.append(keypoints[30:34],keypoints[38:48])
		class5.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + '\n')
	
class1.close()
class2.close()
class3.close()
class4.close()
class5.close()
print('finish.....')


class1 = open('train_outwear.txt','r')
ff = class1.readlines()
ss = ff[100].split(',')
pp = np.array(map(eval,ss[1].split()))
print len(ff)
print ss[0]
print pp.shape, pp

class1.close()

'''

im, classes ,points= get_iterm(rows,1)
print len(rows[1]),rows[1]
print im
print points.shape,points
print points[17],points[18],points[19]
print type(classes)

root = 'train/'
for i in range(1, 31632): #31632
	im, classes ,points= get_iterm(rows,i)
	#print len(rows[i]),rows[i]
	#print im
	#print points.shape,points
	#print mask
	if cmp(classes,'blouse') == 0:
		class1.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points).strip() + ',\n') #
		#class1.writelines([root+im,'\n'])
		#class1.writelines([str(points),'\n'])
	elif cmp(classes, 'skirt') == 0:
		class2.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + '\n')
	elif cmp(classes,'outwear') == 0:
		class3.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + '\n')
	elif cmp(classes, 'dress') == 0:
		class4.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + 'n')
	elif cmp(classes,'trousers') == 0:
		class5.write(root + im + ',' + ' '.join(map(str, points)) + ',\n') #str(points) + '\n')

class1.close()
class2.close()
class3.close()
class4.close()
class5.close()


print('finish.....')

class1 = open('train_trousers.txt','r')
ff = class1.readlines()
ss = ff[0].split(',')
pp = np.array(map(eval,ss[1].split()))
print len(ff)
print ss[0]
print pp.shape, pp

class1.close()
'''