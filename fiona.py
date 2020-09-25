
from People.Person import Favorite
def class Fiona(Favorite):
    def __init__(self, relationShip = None):
        super(Fiona, self).__init__(relationship = 'Casper van Elteren')

        self.properties = dict(cooking = True,\
                            cleaning = False,\
                            sports = True,\
                            tea = True,\
                            mice_inside_house = False,\
                            family = True )
        # init fun stuff
        while True:
            self.adventure_time(self._relationship)

    def adventure_time(self, withPerson = None):
        #
        for things in self.list_fun_things(time = now()):
            if things is "cuddle":
                self.cuddle(self._relationship)
            elif things == "cooking":
                self.make_food_for_Fiona()
            elif things == "need hugs":
                self.cheerup(by = self._relationship)
            #....
