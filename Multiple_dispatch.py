# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:43:14 2021

@author: learningmachine
"""

class Pet:
    pass

class Dog(Pet):
    def __init__(self,name):
        self.Name = name
        
    def encounter(self,pet):
        if isinstance(pet,Cat):
            print(f"{self.Name} meets {pet.Name} and chases")
        else:
            print(f"{self.Name} meets {pet.Name} and sniffs")
            
        
class Cat(Pet):
    def __init__(self,name):
        self.Name = name 
        
    def encounter(self,pet):
        if isinstance(pet,Cat):
            print(f"{self.Name} meets {pet.Name} and slinks")
        else:
            print(f"{self.Name} meets {pet.Name} and hisses")
        
        
fido = Dog("Fido")
rex = Dog("Rex")
whiskers = Cat("Whiskers")
spots = Cat("Spots")

fido.encounter(rex)
fido.encounter(whiskers)
whiskers.encounter(rex)
whiskers.encounter(spots)

