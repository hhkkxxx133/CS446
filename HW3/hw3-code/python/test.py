import math
import numpy

input = raw_input("Enter the position of stumps follewed by the position of the dinasour:")
input_list = input.split()
input_list = [float(item) for item in input_list] 

y_target = input_list.pop()
x_target = input_list.pop()

num = len(input_list)
barrier = []
for i in range(num/2):
  barrier.append([input_list[2*i],input_list[2*i+1]])




#画原始图
def drawGraph(x,y):
  pl.title("The Convex Hull")
  pl.xlabel("x axis")
  pl.ylabel("y axis")
  pl.plot(x,y,'ro')
#画凸包
def drawCH(x,y):
  #pl.plot(x,y1,label='DP',linewidth=4,color='red')
  pl.plot(x,y,color='blue',linewidth=2)
  pl.plot(x[-1],y[-1],x[0],y[0])
  lastx=[x[-1],x[0]]
  lasty=[y[-1],y[0]]
  pl.plot(lastx,lasty,color='blue',linewidth=2)
#存点的矩阵,每行一个点,列0->x坐标,列1->y坐标,列2->代表极角
def matrix(rows,cols):
  cols=3
  mat = [[0 for col in range (cols)]
        for row in range(rows)]
  return mat
#返回叉积
def crossMut(stack,p3):
  p2=stack[-1]
  p1=stack[-2]
  vx,vy=(p2[0]-p1[0],p2[1]-p1[1])
  wx,wy=(p3[0]-p1[0],p3[1]-p1[1])
  return (vx*wy-vy*wx)
#Graham扫描法O(nlogn),mat是经过极角排序的点集
def GrahamScan(mat):
  #print mat
  points=len(mat) #点数
  """
 for k in range(points):
  print mat[k][0],mat[k][1],mat[k][2]
 """
  stack=[]
  stack.append((mat[0][0],mat[0][1])) #push p0
  stack.append((mat[1][0],mat[1][1])) #push p1
  stack.append((mat[2][0],mat[2][1])) #push p2
  for i in range(3,points):
    #print stack
    p3=(mat[i][0],mat[i][1])
    while crossMut(stack,p3)<0:stack.pop()
    stack.append(p3)
  return stack


mat = matrix(num,3) 
mininum = 1000
for i in range(num):	
  mat[i][0] = barrier[i][0]
  mat[i][1] = barrier[i][1]
  if barrier[i][1]<mininum: 
    mininum=barrier[i][1]
    idx = i

d = {}  
for i in range(num):
  if (mat[i][0],mat[i][1])==(x[idx],y[idx]):mat[i][2]=0
  else:mat[i][2]=math.atan2((mat[i][1]-y[idx]),(mat[i][0]-x[idx]))
  d[(mat[i][0],mat[i][1])]=mat[i][2]
lst=sorted(d.items(),key=lambda e:e[1]) 

for i in range(num):	
  ((x,y),eth0)=lst[i]
  mat[i][0],mat[i][1],mat[i][2]=x,y,eth0

stack=GrahamScan(mat)

x,y=[],[]
for item in stack:
  x.append(item[0])
  y.append(item[1])


