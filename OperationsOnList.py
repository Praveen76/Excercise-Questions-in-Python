#Question1:- Impute negative values with 0 in the list
mylist = [1,-2,3,-4,5,6,-7,8,9]
for i in range(len(mylist)):
    if mylist[i]<0:
        mylist[i]=0 
        
        
#Question2:- Give me a list with alternate sign of elements(ie.difference between 
#next element and current element)

b=[]
mylist = [3,2,4,1,15,6,7,5,2]
sign = lambda a: 1 if a>0 else -1 if a<0 else 0

for i in range (0,len(mylist)-1):
    prev = mylist[i-1]
    cur = mylist[i]
    nxt = mylist[i+1]

    result=nxt-cur
    result1=cur-prev
    print(i)
    if (i==0):
        b.append(result)
    else:
        if (sign(result1 ) != 
