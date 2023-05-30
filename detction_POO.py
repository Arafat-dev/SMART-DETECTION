class Comptage:
   


    def __init__(self):
        self.nb_person = 0
        self.nb_caisse = 0
        self.execution = 0

    

    def afficher(self):
        print(self.nb_person)
        print(self.nb_caisse)
    
    def setNbperson(self,nb):
        self.nb_person = nb

     
    def getNbperson(self):
        return self.nb_person

obj1 = Comptage()

obj1.afficher()
obj1.setNbperson(5)

v = obj1.getNbperson()
print(v)





