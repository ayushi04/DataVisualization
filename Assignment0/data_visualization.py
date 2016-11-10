import csv,datetime
from PIL import Image
import numpy as np
filepath='C:/Users/Ayushi/Desktop/1 sem/data visualisation/P00000001-ALL/P00000001-ALL.csv'
reader1 = csv.DictReader(open(filepath, 'r'))
reader2 = csv.DictReader(open(filepath, 'r'))

w, h = 256,256 
#matrix = [[0 for x in range(w)] for y in range(h)] 
matrix = np.zeros(shape=(w,h))
i=0;
for row1 in reader1:
    j=0;    
    for row2 in reader2:
        bit=0;
        name1 = row1['cand_nm']
        amount1 = row1['contb_receipt_amt']
        datestr = row1['contb_receipt_dt']
        date1 = datetime.datetime.strptime(datestr, '%d-%b-%y')
        occ1=row1['contbr_occupation']
        emp1=row1['contbr_employer']
        tp1=row1['form_tp']       
        city1=row1['contbr_city']
        state1=row1['contbr_st']
        
        name2 = row2['cand_nm']
        amount2 = row2['contb_receipt_amt'] 
        datestr = row2['contb_receipt_dt']
        date2 = datetime.datetime.strptime(datestr, '%d-%b-%y')
        occ2=row2['contbr_occupation']
        emp2=row2['contbr_employer']
        tp2=row2['form_tp']     
        city2=row1['contbr_city']
        state2=row1['contbr_st']
        
        #condition1
        if(name1==name2) :
            bit = bit*10 +1
        else :
            bit = bit*10
        #condition2
        if (amount1>amount2):
            bit = bit*10 + 1
        else :
            bit = bit *10
        #condition3
        if(date1>date2):
            bit = bit*10 + 1
        else :
            bit = bit *10
        #condition4 : contbr_occupation
        if(occ1==occ2):
            bit = bit*10 + 1
        else :
            bit = bit *10
        #condition5 : contbr_employer
        if(emp1==emp2):
            bit = bit*10 + 1
        else :
            bit = bit *10
        #condition6 : form_tp
        if(tp1==tp2):
            bit = bit*10 + 1
        else :
            bit = bit *10    
        #condition7 : contbr_city
        if(city1==city2):
            bit = bit*10 + 1
        else :
            bit = bit *10    
        #condition8 : contbr_st
        if(state1==state2):
            bit = bit*10 + 1
        else :
            bit = bit *10    
            
            
        # werite all conditions
        num=int(str(bit),2)
        matrix[i][j]=num
        if(j>=255):
            break
        j+=1
    if i>=255:
        break  
    i=i+1      
print ('hello')
print (matrix)
#img = Image.fromarray(matrix, 'RGB')
#img.save('img.png')
#img.show()
img1 = Image.fromarray(matrix)
#img1.save('img1.bmp')
img1.show()
