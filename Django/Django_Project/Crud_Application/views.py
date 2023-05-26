from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import Datas

# Create your views here.

def home(request):
   mydata=Datas.objects.all()
   if(mydata!=""):
        return render(request,"home.html",{'datas': mydata })

   else:
        return render(request,"home.html")

def addData(request):
    if request.method=='POST':
       name=request.post['name']
       age=request.post['age']
       address=request.post['address']
       contact=request.post['contact']
       mail=request.post['mail']

       obj=Datas
       obj.Name=name
       obj.Age=age
       obj.Address=address 
       obj.Contact=contact
       obj.Mail=mail
       obj.save()
       mydata=Datas.objects.all()
       return redirect('home')
    return render(request,"home.html")   

def updateData(request,id):   
    mydata=Datas.objects.get(id=id) 
    if request.method=='POST':
       name=request.post['name']
       age=request.post['age']
       address=request.post['address']
       contact=request.post['contact']
       mail=request.post['mail']

       mydata.Name=name
       mydata.Age=age
       mydata.Adderss=Address
       mydata.Contact=contact
       mydata.Mail=mail
       mydata.save()
       return redirect('home')
    return render(request,'update.html',{'data':mydata})

def deleteData(request,id):
    mydata=Datas.objects.get(id=id)
    mydata.data.delete()
    return redirect('home')